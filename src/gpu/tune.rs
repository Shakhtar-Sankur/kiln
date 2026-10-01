//! The GPU auto-tuner: for each distinct matmul (shape and operand kind),
//! every schedule of the search space is compiled into one module and
//! timed alone on the device with CUDA events; the fastest is cached per
//! GPU model in `gtune.txt`.

use super::attention::{self, AttnParams};
use super::codegen::{self, MmParams};
use super::driver::{Cuda, Nvrtc};
use super::{Device, compile_cubin, mm_key};
use crate::codegen::Arg;
use crate::fuse::{Kernel, LoopK, MatmulK, Plan, Step};
use crate::runtime::{matmul_signature, pack, pack_half_t};
use crate::tensor::numel;
use std::collections::HashMap;

fn cache_path() -> std::path::PathBuf {
    crate::jit::cache_dir().join("gtune.txt")
}

fn key(dev: &str, sig: &str) -> String {
    format!("{}|{sig}", dev.replace(' ', "_"))
}

/// Lines `key bm bn bk tm tn tc ks vec` (older formats are ignored and
/// re-tuned).
fn load() -> HashMap<String, MmParams> {
    let mut out = HashMap::new();
    let Ok(s) = std::fs::read_to_string(cache_path()) else {
        return out;
    };
    for line in s.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 9 {
            continue;
        }
        let n: Vec<usize> = f[1..6].iter().filter_map(|x| x.parse().ok()).collect();
        if let (5, Ok(tc), Ok(ks), Ok(vec)) = (n.len(), f[6].parse(), f[7].parse(), f[8].parse()) {
            let p = MmParams {
                ks,
                vec,
                ..MmParams::new(n[0], n[1], n[2], n[3], n[4], tc)
            };
            if p.valid() {
                out.insert(f[0].to_string(), p);
            }
        }
    }
    out
}

fn save(all: &HashMap<String, MmParams>) {
    let mut lines: Vec<String> = all
        .iter()
        .map(|(k, p)| {
            format!(
                "{k} {} {} {} {} {} {} {} {}",
                p.bm, p.bn, p.bk, p.tm, p.tn, p.tc, p.ks, p.vec
            )
        })
        .collect();
    lines.sort();
    let _ = std::fs::create_dir_all(crate::jit::cache_dir());
    let _ = std::fs::write(cache_path(), lines.join("\n") + "\n");
}

fn matmuls(plan: &Plan) -> Vec<&MatmulK> {
    let mut seen = std::collections::HashSet::new();
    plan.steps
        .iter()
        .filter_map(|s| match s {
            Step::Kernel(Kernel::Matmul(mk)) if seen.insert(matmul_signature(mk)) => Some(&**mk),
            _ => None,
        })
        .collect()
}

/// Cached schedules for this plan's matmuls on the current GPU.
pub fn cached(plan: &Plan, device: Device, half: bool) -> HashMap<String, MmParams> {
    if device != Device::Cuda {
        return HashMap::new();
    }
    let Ok(c) = Cuda::open() else {
        return HashMap::new();
    };
    let all = load();
    let mut out = HashMap::new();
    for mk in matmuls(plan) {
        let sig = mm_key(mk, half);
        if let Some(p) = all.get(&key(&c.name, &sig)) {
            out.insert(sig, *p);
        }
    }
    out
}

/// Device buffers for a matmul's arguments (the same for every schedule
/// of one precision): constants uploaded, activations filled, packed
/// weights packed, the workspace sized for `ws` floats.
fn matmul_buffers(
    c: &Cuda,
    g: &crate::graph::Graph,
    mk: &MatmulK,
    args: &[Arg],
    ws: usize,
    bufs: &mut HashMap<String, u64>,
) -> Result<Vec<u64>, String> {
    let mut out = Vec::new();
    for a in args {
        let key = format!("{a:?}");
        if let Some(&p) = bufs.get(&key)
            && !matches!(a, Arg::Workspace)
        {
            out.push(p);
            continue;
        }
        let p = match a {
            Arg::Workspace => {
                let p = c.alloc(ws.max(1) * 4)?;
                out.push(p);
                continue;
            }
            Arg::Value(v) => match g.konst(*v) {
                Some(t) => {
                    let p = c.alloc(t.len() * 4)?;
                    c.upload(p, &t.to_f32())?;
                    p
                }
                None => {
                    let n = numel(g.shape(*v));
                    let p = c.alloc(n * 4)?;
                    c.upload(p, &vec![0.01f32; n])?;
                    p
                }
            },
            Arg::Packed {
                value,
                lin,
                k: kk,
                n,
                nr,
            } => {
                let data = pack(g.konst(*value).unwrap(), lin, *kk, *n, *nr, mk.vk, mk.vj);
                let p = c.alloc(data.len() * 4)?;
                c.upload(p, &data)?;
                p
            }
            Arg::PackedHalf {
                value,
                lin,
                k: kk,
                n,
            } => {
                let h = pack_half_t(g.konst(*value).unwrap(), lin, *kk, *n, mk.vk, mk.vj);
                let words: Vec<f32> = h
                    .chunks(2)
                    .map(|c| {
                        f32::from_bits(u32::from(c[0]) | (u32::from(*c.get(1).unwrap_or(&0)) << 16))
                    })
                    .collect();
                let p = c.alloc(words.len() * 4)?;
                c.upload(p, &words)?;
                p
            }
        };
        bufs.insert(key, p);
        out.push(p);
    }
    Ok(out)
}

/// Milliseconds per run of each schedule (median of 5 x 10 runs).
fn time_schedules(
    c: &Cuda,
    nv: &Nvrtc,
    g: &crate::graph::Graph,
    mk: &MatmulK,
    space: &[MmParams],
) -> Result<Vec<f64>, String> {
    let arch = (c.cc.0 * 10 + c.cc.1) as u32;
    let mut src = String::from(codegen::PRELUDE);
    let mut kernels = Vec::new();
    let mut names = Vec::new();
    for (i, p) in space.iter().enumerate() {
        let gk = codegen::matmul_kernel(mk, *p);
        src.push('\n');
        src.push_str(&gk.src.replace("KNAME", &format!("t{i}")));
        names.push(format!("t{i}"));
        if gk.then.is_some() {
            names.push(format!("t{i}_r"));
        }
        kernels.push(gk);
    }
    let (bin, _) = compile_cubin(nv, &src, arch)?;
    let fs = c.load(&bin, &names)?;
    let mut bufs = HashMap::new();
    let mut out = Vec::new();
    let mut fi = 0;
    for gk in &kernels {
        let args = matmul_buffers(c, g, mk, &gk.args, gk.ws, &mut bufs)?;
        let (f, fthen) = (fs[fi], gk.then.map(|_| fs[fi + 1]));
        fi += if gk.then.is_some() { 2 } else { 1 };
        if gk.smem > 48 * 1024 {
            c.allow_smem(f, gk.smem)?;
        }
        let launch = || -> Result<(), String> {
            c.launch(f, gk.grid, gk.block, gk.smem, &args)?;
            if let (Some(ft), Some((grid, block))) = (fthen, gk.then) {
                c.launch(ft, grid, block, 0, &args)?;
            }
            Ok(())
        };
        launch()?;
        c.sync()?;
        let mut reps = Vec::new();
        for _ in 0..5 {
            let ms = c.time(&mut || {
                for _ in 0..10 {
                    launch()?;
                }
                Ok(())
            })? as f64
                / 10.0;
            reps.push(ms);
        }
        reps.sort_by(f64::total_cmp);
        out.push(reps[reps.len() / 2]);
    }
    Ok(out)
}

/// Times every schedule of each matmul not yet tuned on this GPU, then
/// split-K versions of the four fastest, and keeps the fastest overall.
pub fn tune(
    plan: &Plan,
    device: Device,
    half: bool,
    log: bool,
) -> Result<HashMap<String, MmParams>, String> {
    if device != Device::Cuda {
        return Ok(HashMap::new());
    }
    let c = Cuda::open()?;
    let nv = Nvrtc::open()?;
    let mut all = load();
    let mut out = HashMap::new();
    let g = &plan.g;
    for mk in matmuls(plan) {
        let sig = mm_key(mk, half);
        let k = key(&c.name, &sig);
        if let Some(p) = all.get(&k) {
            out.insert(sig, *p);
            continue;
        }
        let t0 = std::time::Instant::now();
        let space = codegen::mm_space(half);
        let times = time_schedules(&c, &nv, g, mk, &space)?;
        let mut ranked: Vec<(f64, MmParams)> =
            times.into_iter().zip(space.iter().copied()).collect();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Split-K for the four fastest, where the reduction is long enough.
        let mut split = Vec::new();
        for &(_, p) in ranked.iter().take(4) {
            for ks in [2, 3, 4] {
                if mk.k >= ks * p.bk * 2 {
                    split.push(MmParams { ks, ..p });
                }
            }
        }
        if !split.is_empty() {
            let st = time_schedules(&c, &nv, g, mk, &split)?;
            ranked.extend(st.into_iter().zip(split.iter().copied()));
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        let best = ranked[0];
        if log {
            let flops = 2.0
                * (mk.m * mk.n * mk.k) as f64
                * mk.batch.iter().map(|b| b.1).product::<usize>() as f64;
            eprintln!(
                "tuned {sig}: {:?} {:.3} ms ({:.0} GFLOPS) of {} schedules in {:.1} s",
                best.1,
                best.0,
                flops / best.0 * 1e-6,
                space.len() + split.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        all.insert(k, best.1);
        save(&all);
        out.insert(sig, best.1);
    }
    Ok(out)
}

fn attn_cache_path() -> std::path::PathBuf {
    crate::jit::cache_dir().join("gtune_attn.txt")
}

fn load_attn() -> HashMap<String, Option<AttnParams>> {
    let mut out = HashMap::new();
    let Ok(s) = std::fs::read_to_string(attn_cache_path()) else {
        return out;
    };
    for line in s.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            [k, "unfused"] => {
                out.insert(k.to_string(), None);
            }
            [k, bm, nt] => {
                if let (Ok(bm), Ok(nt)) = (bm.parse(), nt.parse()) {
                    out.insert(k.to_string(), Some(AttnParams { bm, nt }));
                }
            }
            _ => {}
        }
    }
    out
}

fn save_attn(all: &HashMap<String, Option<AttnParams>>) {
    let mut lines: Vec<String> = all
        .iter()
        .map(|(k, p)| match p {
            Some(p) => format!("{k} {} {}", p.bm, p.nt),
            None => format!("{k} unfused"),
        })
        .collect();
    lines.sort();
    let _ = std::fs::create_dir_all(crate::jit::cache_dir());
    let _ = std::fs::write(attn_cache_path(), lines.join("\n") + "\n");
}

/// The attention patterns of a plan, one per tuning key.
fn patterns(plan: &Plan, half: bool) -> Vec<(String, &MatmulK, &LoopK, &MatmulK)> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for [a, b, c] in attention::find(plan) {
        if let (
            Step::Kernel(Kernel::Matmul(m1)),
            Step::Kernel(Kernel::Loop(lk)),
            Step::Kernel(Kernel::Matmul(m3)),
        ) = (&plan.steps[a], &plan.steps[b], &plan.steps[c])
        {
            let k = attention::key(m1, m3, half);
            if seen.insert(k.clone()) {
                out.push((k, &**m1, lk, &**m3));
            }
        }
    }
    out
}

/// Cached fusion choices for this plan's attention patterns on this GPU.
pub fn cached_attention(
    plan: &Plan,
    device: Device,
    half: bool,
) -> HashMap<String, Option<AttnParams>> {
    let mut out = HashMap::new();
    if device != Device::Cuda {
        return out;
    }
    let Ok(c) = Cuda::open() else {
        return out;
    };
    let all = load_attn();
    for (k, ..) in patterns(plan, half) {
        if let Some(p) = all.get(&key(&c.name, &k)) {
            out.insert(k, *p);
        }
    }
    out
}

/// For each attention pattern, times every fused shape against the three
/// unfused kernels (with the matmul schedules in `mm`) and keeps the
/// fastest.
pub fn tune_attention(
    plan: &Plan,
    device: Device,
    half: bool,
    mm: &HashMap<String, MmParams>,
    log: bool,
) -> Result<HashMap<String, Option<AttnParams>>, String> {
    let mut out = HashMap::new();
    if device != Device::Cuda {
        return Ok(out);
    }
    let c = Cuda::open()?;
    let nv = Nvrtc::open()?;
    let arch = (c.cc.0 * 10 + c.cc.1) as u32;
    let mut all = load_attn();
    let g = &plan.g;
    for (k, m1, lk, m3) in patterns(plan, half) {
        let ck = key(&c.name, &k);
        if let Some(p) = all.get(&ck) {
            out.insert(k, *p);
            continue;
        }
        let sched = |m: &MatmulK| {
            mm.get(&mm_key(m, half))
                .copied()
                .filter(|p| p.tc == half)
                .unwrap_or_else(|| codegen::default_mm(m, c.sms, half))
        };
        // Candidate 0: unfused (three kernels); then each fused shape.
        let mut cands: Vec<(Option<AttnParams>, Vec<codegen::GpuKernel>)> = vec![(
            None,
            vec![
                codegen::matmul_kernel(m1, sched(m1)),
                codegen::loop_kernel(lk, None),
                codegen::matmul_kernel(m3, sched(m3)),
            ],
        )];
        for p in attention::space() {
            if let Some(kern) = attention::attention_kernel(m1, lk, m3, p) {
                cands.push((Some(p), vec![kern]));
            }
        }
        let mut src = String::from(codegen::PRELUDE);
        let mut names = Vec::new();
        for (ci, (_, ks)) in cands.iter().enumerate() {
            for (ki, kern) in ks.iter().enumerate() {
                let n = format!("c{ci}_{ki}");
                src.push('\n');
                src.push_str(&kern.src.replace("KNAME", &n));
                if kern.then.is_some() {
                    names.push(format!("{n}_r"));
                }
                names.push(n);
            }
        }
        let (bin, _) = compile_cubin(&nv, &src, arch)?;
        let fs = c.load(&bin, &names)?;
        // One device buffer per value any candidate touches.
        let mut bufs: HashMap<usize, u64> = HashMap::new();
        // Per candidate: (function, second function, arguments) per kernel.
        type Launches = Vec<Vec<(usize, Option<usize>, Vec<u64>)>>;
        let mut launches: Launches = Vec::new();
        let mut fi = 0;
        for (_, ks) in &cands {
            let mut l = Vec::new();
            for kern in ks {
                let mut args = Vec::new();
                for a in &kern.args {
                    let v = match a {
                        Arg::Value(v) => v,
                        Arg::Workspace => {
                            args.push(c.alloc(kern.ws.max(1) * 4)?);
                            continue;
                        }
                        _ => return Err("attention operands are activations".into()),
                    };
                    let p = match bufs.get(v) {
                        Some(&p) => p,
                        None => {
                            let p = match g.konst(*v) {
                                Some(t) => {
                                    let p = c.alloc(t.len() * 4)?;
                                    c.upload(p, &t.to_f32())?;
                                    p
                                }
                                None => {
                                    let n = numel(g.shape(*v));
                                    let p = c.alloc(n * 4)?;
                                    c.upload(p, &vec![0.01f32; n])?;
                                    p
                                }
                            };
                            bufs.insert(*v, p);
                            p
                        }
                    };
                    args.push(p);
                }
                // Names were pushed as [<n>_r,] <n>.
                let (f, ft) = if kern.then.is_some() {
                    (fi + 1, Some(fi))
                } else {
                    (fi, None)
                };
                if kern.smem > 48 * 1024 {
                    c.allow_smem(fs[f], kern.smem)?;
                }
                l.push((f, ft, args));
                fi += if kern.then.is_some() { 2 } else { 1 };
            }
            launches.push(l);
        }
        let mut best = (f64::INFINITY, None);
        let mut report = Vec::new();
        for ((choice, ks), l) in cands.iter().zip(&launches) {
            let run = || -> Result<(), String> {
                for (kern, (f, ft, args)) in ks.iter().zip(l) {
                    c.launch(fs[*f], kern.grid, kern.block, kern.smem, args)?;
                    if let (Some(ft), Some((grid, block))) = (ft, kern.then) {
                        c.launch(fs[*ft], grid, block, 0, args)?;
                    }
                }
                Ok(())
            };
            run()?;
            c.sync()?;
            let mut reps = Vec::new();
            for _ in 0..5 {
                let ms = c.time(&mut || {
                    for _ in 0..10 {
                        run()?;
                    }
                    Ok(())
                })? as f64
                    / 10.0;
                reps.push(ms);
            }
            reps.sort_by(f64::total_cmp);
            let ms = reps[reps.len() / 2];
            report.push(format!(
                "{} {ms:.3} ms",
                match choice {
                    Some(p) => format!("fused {}x{}", p.bm, p.nt),
                    None => "unfused".into(),
                }
            ));
            if ms < best.0 {
                best = (ms, *choice);
            }
        }
        if log {
            eprintln!("tuned {k}: {}", report.join(", "));
        }
        all.insert(ck, best.1);
        save_attn(&all);
        out.insert(k, best.1);
    }
    Ok(out)
}
