//! The auto-tuner: for every distinct matmul in a plan, generates schedule
//! variants (register tile, K blocking, how tasks split the rows and
//! columns), compiles them, times each on the actual operands with the
//! actual epilogue, and keeps the fastest. Results are cached per shape,
//! vector width, thread count and CPU model.

use crate::codegen::{self, Arg, Params};
use crate::fuse::{Kernel, MatmulK, Plan, Step};
use crate::jit::{self, KernelFn};
use crate::pool::Pool;
use crate::runtime::matmul_signature;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
        .replace(' ', "_")
}

pub fn cache_key(sig: &str, w: usize, threads: usize) -> String {
    format!("{sig}|w{w}|t{threads}|{}", cpu_model())
}

fn cache_path() -> PathBuf {
    jit::cache_dir().join("tune.txt")
}

pub fn load_cache() -> HashMap<String, Params> {
    let mut m = HashMap::new();
    if let Ok(s) = std::fs::read_to_string(cache_path()) {
        for l in s.lines() {
            let f: Vec<&str> = l.split_whitespace().collect();
            if f.len() == 7 {
                let n: Vec<usize> = f[1..].iter().filter_map(|x| x.parse().ok()).collect();
                if n.len() == 6 {
                    m.insert(
                        f[0].to_string(),
                        Params {
                            mr: n[0],
                            nr: n[1],
                            mc: n[2],
                            ncp: n[3],
                            kc: n[4],
                            pf: n[5],
                        },
                    );
                }
            }
        }
    }
    m
}

fn save_cache(m: &HashMap<String, Params>) {
    let mut lines: Vec<String> = m
        .iter()
        .map(|(k, p)| format!("{k} {} {} {} {} {} {}", p.mr, p.nr, p.mc, p.ncp, p.kc, p.pf))
        .collect();
    lines.sort();
    let _ = std::fs::create_dir_all(jit::cache_dir());
    let _ = std::fs::write(cache_path(), lines.join("\n") + "\n");
}

/// Register tiles and K blocks to try.
fn tiles(mk: &MatmulK, w: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let n_pad = mk.n.div_ceil(w) * w;
    for mr in [4, 6, 8, 12] {
        for nv in 1..=4 {
            let nr = nv * w;
            if nr > n_pad.max(w) || mr * nv > 28 || mr > mk.m.max(4) {
                continue;
            }
            let mut kcs = vec![mk.k];
            for kc in [128, 256] {
                if kc < mk.k {
                    kcs.push(kc);
                }
            }
            for kc in kcs {
                out.push((mr, nr, kc));
            }
        }
    }
    out
}

/// Ways to split the work into tasks, for a given tile.
fn layouts(mk: &MatmulK, mr: usize, nr: usize, kc: usize, threads: usize) -> Vec<Params> {
    let np = mk.n.div_ceil(nr);
    let mall = mk.m.div_ceil(mr) * mr;
    let mut v = Vec::new();
    // Split the columns: every task takes all rows of some panels.
    for per in [threads, 2 * threads, 4 * threads] {
        v.push(Params {
            mr,
            nr,
            mc: mall,
            ncp: np.div_ceil(per).max(1),
            kc,
            pf: 0,
        });
    }
    // Split the rows: every task takes all panels for some rows.
    for per in [threads, 2 * threads, 4 * threads] {
        let mc = mk.m.div_ceil(per).div_ceil(mr).max(1) * mr;
        v.push(Params {
            mr,
            nr,
            mc,
            ncp: np,
            kc,
            pf: 0,
        });
    }
    // Both.
    v.push(Params {
        mr,
        nr,
        mc: mk.m.div_ceil(2).div_ceil(mr).max(1) * mr,
        ncp: np.div_ceil(threads).max(1),
        kc,
        pf: 0,
    });
    v.sort_by_key(|p| (p.mc, p.ncp));
    v.dedup();
    // And software prefetching of B, for each split.
    let base = v.clone();
    for pf in [8, 32] {
        v.extend(base.iter().map(|p| Params { pf, ..*p }));
    }
    v
}

/// Buffers for timing a kernel. Weights (constants and packed panels)
/// come in enough copies to exceed the last-level cache, rotated between
/// runs, so each run reads them cold from memory as inference does.
struct Harness {
    sets: Vec<Vec<Vec<f32>>>,
}

const COLD_BYTES: usize = 96 << 20;

/// Buffers for a kernel's arguments: real constants, random activations.
fn harness(plan: &Plan, args: &[Arg]) -> Harness {
    let g = &plan.g;
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
    };
    let weight_bytes: usize = args
        .iter()
        .map(|a| match a {
            Arg::Value(v) if g.konst(*v).is_some() => 4 * crate::tensor::numel(g.shape(*v)),
            Arg::Packed { k, n, nr, .. } => 4 * n.div_ceil(*nr) * k * nr,
            _ => 0,
        })
        .sum();
    let copies = COLD_BYTES.div_ceil(weight_bytes.max(1)).clamp(1, 64);
    let activations: Vec<Option<Vec<f32>>> = args
        .iter()
        .map(|a| match a {
            Arg::Value(v) if g.konst(*v).is_none() => Some(
                (0..crate::tensor::numel(g.shape(*v)))
                    .map(|_| rnd())
                    .collect(),
            ),
            _ => None,
        })
        .collect();
    let mut sets = Vec::new();
    for _ in 0..copies {
        let set = args
            .iter()
            .zip(&activations)
            .map(|(a, act)| match (a, act) {
                (_, Some(v)) => v.clone(),
                (Arg::Value(v), None) => g.konst(*v).unwrap().as_f32().to_vec(),
                (Arg::Packed { k, n, nr, .. }, None) => {
                    (0..n.div_ceil(*nr) * k * nr).map(|_| rnd()).collect()
                }
                (Arg::PackedHalf { .. }, None) => unreachable!("fp16 operands are GPU-only"),
            })
            .collect();
        sets.push(set);
    }
    Harness { sets }
}

struct SendPtr(*const *mut f32);
unsafe impl Sync for SendPtr {}

fn time_kernel(pool: &Pool, f: KernelFn, h: &mut Harness, tasks: usize) -> f64 {
    let ptrs: Vec<Vec<*mut f32>> = h
        .sets
        .iter_mut()
        .map(|s| s.iter_mut().map(|b| b.as_mut_ptr()).collect())
        .collect();
    let chunks = tasks.min(pool.threads() * 4).max(1);
    let run = |set: usize| {
        let p = SendPtr(ptrs[set].as_ptr());
        pool.run(chunks, &|c| {
            let a = (c * tasks / chunks) as i64;
            let b = ((c + 1) * tasks / chunks) as i64;
            let p = &p;
            // SAFETY: harness buffers sized for this kernel's arguments.
            unsafe { f(p.0, a, b) };
        })
    };
    let n_sets = ptrs.len();
    run(0);
    // The median of runs, each on the next copy of the weights.
    let mut ts = Vec::new();
    let start = Instant::now();
    let mut i = 1;
    while ts.len() < 5 || (start.elapsed().as_secs_f64() < 0.05 && ts.len() < 40) {
        let t = Instant::now();
        run(i % n_sets);
        ts.push(t.elapsed().as_secs_f64());
        i += 1;
    }
    ts.sort_by(f64::total_cmp);
    ts[ts.len() / 2]
}

/// Compiles `sources` (name, body) into one library per chunk, in parallel.
fn compile_all(
    w: usize,
    bodies: &[(String, String)],
) -> Result<Vec<(String, jit::Library)>, String> {
    let parts = 4.min(bodies.len()).max(1);
    let chunks: Vec<Vec<&(String, String)>> = (0..parts)
        .map(|p| bodies.iter().skip(p).step_by(parts).collect())
        .collect();
    let results: Vec<Result<(Vec<String>, jit::Library), String>> = std::thread::scope(|s| {
        let hs: Vec<_> = chunks
            .iter()
            .map(|c| {
                s.spawn(move || {
                    let mut src = codegen::prelude(w);
                    for (name, body) in c {
                        src.push_str(&format!("\nvoid {name}{body}"));
                    }
                    jit::compile(&src).map(|l| (c.iter().map(|x| x.0.clone()).collect(), l))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = Vec::new();
    for r in results {
        let (names, lib) = r?;
        let lib = std::sync::Arc::new(lib);
        for n in names {
            out.push((n, ArcLib(lib.clone())));
        }
    }
    Ok(out.into_iter().map(|(n, l)| (n, l.into_lib())).collect())
}

struct ArcLib(std::sync::Arc<jit::Library>);

impl ArcLib {
    fn into_lib(self) -> jit::Library {
        // Libraries stay loaded for the life of the process; tuning only
        // needs the function pointers, so a shared handle is fine.
        match std::sync::Arc::try_unwrap(self.0) {
            Ok(l) => l,
            Err(a) => jit::Library::alias(&a),
        }
    }
}

/// Tunes every distinct matmul of the plan not already in `cache`;
/// returns schedules by signature. Logs progress when `log`.
pub fn tune(plan: &Plan, threads: usize, log: bool) -> Result<HashMap<String, Params>, String> {
    let w = jit::vector_width();
    let mut cache = load_cache();
    let mut out = HashMap::new();
    let mut todo: Vec<&MatmulK> = Vec::new();
    for s in &plan.steps {
        if let Step::Kernel(Kernel::Matmul(mk)) = s {
            let sig = matmul_signature(mk);
            if out.contains_key(&sig) || todo.iter().any(|m| matmul_signature(m) == sig) {
                continue;
            }
            match cache.get(&cache_key(&sig, w, threads)) {
                Some(p) => {
                    out.insert(sig, *p);
                }
                None => todo.push(mk),
            }
        }
    }
    if todo.is_empty() {
        return Ok(out);
    }
    let pool = Pool::new(threads);
    let t0 = Instant::now();
    // Stage 1: register tile and K block, with a default split; stage 2:
    // the split, for the best tile.
    let mut best: Vec<(Params, f64)> = vec![
        (
            Params {
                mr: 0,
                nr: 0,
                mc: 0,
                ncp: 0,
                kc: 0,
                pf: 0
            },
            f64::MAX
        );
        todo.len()
    ];
    for stage in 0..2 {
        let mut bodies = Vec::new();
        let mut meta = Vec::new();
        for (i, mk) in todo.iter().enumerate() {
            let cands: Vec<Params> = if stage == 0 {
                tiles(mk, w)
                    .into_iter()
                    .map(|(mr, nr, kc)| {
                        let mut p = codegen::default_params(mk, w, threads);
                        let np = mk.n.div_ceil(nr);
                        p.mr = mr;
                        p.nr = nr;
                        p.kc = kc;
                        p.mc = p.mc.div_ceil(mr) * mr;
                        p.ncp = p.ncp.min(np).max(1);
                        p
                    })
                    .collect()
            } else {
                let b = best[i].0;
                layouts(mk, b.mr, b.nr, b.kc, threads)
            };
            for p in cands {
                let src = codegen::matmul_kernel(&plan.g, mk, p, w);
                let name = format!("t{i}_{}", bodies.len());
                bodies.push((name, src.body.clone()));
                meta.push((i, p, src.args, src.tasks));
            }
        }
        let libs = compile_all(w, &bodies)?;
        let fns: HashMap<String, KernelFn> = libs
            .iter()
            .map(|(n, l)| Ok((n.clone(), l.kernel(n)?)))
            .collect::<Result<_, String>>()?;
        std::mem::forget(libs);
        for ((name, _), (i, p, args, tasks)) in bodies.iter().zip(meta) {
            let mut h = harness(plan, &args);
            let t = time_kernel(&pool, fns[name], &mut h, tasks);
            if t < best[i].1 {
                best[i] = (p, t);
            }
        }
    }
    for (mk, (p, t)) in todo.iter().zip(&best) {
        let sig = matmul_signature(mk);
        let flops =
            2.0 * (mk.m * mk.n * mk.k * mk.batch.iter().map(|b| b.1).product::<usize>()) as f64;
        if log {
            eprintln!(
                "tuned {sig}: MR {} NR {} KC {} MC {} panels/task {}: {:.3} ms, {:.0} GFLOPS",
                p.mr,
                p.nr,
                p.kc,
                p.mc,
                p.ncp,
                t * 1e3,
                flops / t * 1e-9
            );
        }
        cache.insert(cache_key(&sig, w, threads), *p);
        out.insert(sig, *p);
    }
    save_cache(&cache);
    if log {
        eprintln!(
            "tuned {} matmul shapes in {:.1} s",
            todo.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(out)
}

/// Cached schedules for the plan's matmuls (no tuning).
pub fn cached(plan: &Plan, threads: usize) -> HashMap<String, Params> {
    let w = jit::vector_width();
    let cache = load_cache();
    let mut out = HashMap::new();
    for s in &plan.steps {
        if let Step::Kernel(Kernel::Matmul(mk)) = s {
            let sig = matmul_signature(mk);
            if let Some(p) = cache.get(&cache_key(&sig, w, threads)) {
                out.insert(sig, *p);
            }
        }
    }
    out
}
