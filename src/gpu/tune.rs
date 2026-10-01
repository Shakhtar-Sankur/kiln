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

fn load() -> HashMap<String, MmParams> {
    let mut out = HashMap::new();
    let Ok(s) = std::fs::read_to_string(cache_path()) else {
        return out;
    };
    for line in s.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() == 7
            && let (Ok(bm), Ok(bn), Ok(bk), Ok(tm), Ok(tn), Ok(tc)) = (
                f[1].parse(),
                f[2].parse(),
                f[3].parse(),
                f[4].parse(),
                f[5].parse(),
                f[6].parse(),
            )
        {
            out.insert(
                f[0].to_string(),
                MmParams {
                    bm,
                    bn,
                    bk,
                    tm,
                    tn,
                    tc,
                },
            );
        }
    }
    out
}

fn save(all: &HashMap<String, MmParams>) {
    let mut lines: Vec<String> = all
        .iter()
        .map(|(k, p)| format!("{k} {} {} {} {} {} {}", p.bm, p.bn, p.bk, p.tm, p.tn, p.tc))
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

/// Times every schedule of each matmul not yet tuned on this GPU.
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
    let arch = (c.cc.0 * 10 + c.cc.1) as u32;
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
        let mut src = String::from(codegen::PRELUDE);
        let mut kernels = Vec::new();
        for (i, p) in space.iter().enumerate() {
            let gk = codegen::matmul_kernel(mk, *p);
            src.push('\n');
            src.push_str(&gk.src.replace("KNAME", &format!("t{i}")));
            kernels.push(gk);
        }
        let names: Vec<String> = (0..space.len()).map(|i| format!("t{i}")).collect();
        let (bin, _) = compile_cubin(&nv, &src, arch)?;
        let fs = c.load(&bin, &names)?;
        // Buffers for the arguments (the same for every schedule).
        let mut args = Vec::new();
        for a in &kernels[0].args {
            args.push(match a {
                Arg::Value(v) => match g.konst(*v) {
                    Some(t) => {
                        let p = c.alloc(t.len() * 4)?;
                        c.upload(p, t.as_f32())?;
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
                            f32::from_bits(
                                u32::from(c[0]) | (u32::from(*c.get(1).unwrap_or(&0)) << 16),
                            )
                        })
                        .collect();
                    let p = c.alloc(words.len() * 4)?;
                    c.upload(p, &words)?;
                    p
                }
            });
        }
        let flops = 2.0
            * (mk.m * mk.n * mk.k) as f64
            * mk.batch.iter().map(|b| b.1).product::<usize>() as f64;
        let mut best = (f64::INFINITY, space[0]);
        for (i, p) in space.iter().enumerate() {
            let gk = &kernels[i];
            if gk.smem > 48 * 1024 {
                c.allow_smem(fs[i], gk.smem)?;
            }
            let launch = || c.launch(fs[i], gk.grid, gk.block, gk.smem, &args);
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
            let ms = reps[reps.len() / 2];
            if ms < best.0 {
                best = (ms, *p);
            }
        }
        if log {
            eprintln!(
                "tuned {sig}: {:?} {:.3} ms ({:.0} GFLOPS) of {} schedules in {:.1} s",
                best.1,
                best.0,
                flops / best.0 * 1e-6,
                space.len(),
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
                names.push(n);
            }
        }
        let (bin, _) = compile_cubin(&nv, &src, arch)?;
        let fs = c.load(&bin, &names)?;
        // One device buffer per value any candidate touches.
        let mut bufs: HashMap<usize, u64> = HashMap::new();
        let mut launches: Vec<Vec<(usize, Vec<u64>)>> = Vec::new();
        let mut fi = 0;
        for (_, ks) in &cands {
            let mut l = Vec::new();
            for kern in ks {
                let mut args = Vec::new();
                for a in &kern.args {
                    let Arg::Value(v) = a else {
                        return Err("attention operands are activations".into());
                    };
                    let p = match bufs.get(v) {
                        Some(&p) => p,
                        None => {
                            let p = match g.konst(*v) {
                                Some(t) => {
                                    let p = c.alloc(t.len() * 4)?;
                                    c.upload(p, t.as_f32())?;
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
                if kern.smem > 48 * 1024 {
                    c.allow_smem(fs[fi], kern.smem)?;
                }
                l.push((fi, args));
                fi += 1;
            }
            launches.push(l);
        }
        let mut best = (f64::INFINITY, None);
        let mut report = Vec::new();
        for ((choice, ks), l) in cands.iter().zip(&launches) {
            let run = || -> Result<(), String> {
                for (kern, (f, args)) in ks.iter().zip(l) {
                    c.launch(fs[*f], kern.grid, kern.block, kern.smem, args)?;
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
