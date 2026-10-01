//! The GPU auto-tuner: for each distinct matmul (shape and operand kind),
//! every schedule of the search space is compiled into one module and
//! timed alone on the device with CUDA events; the fastest is cached per
//! GPU model in `gtune.txt`.

use super::codegen::{self, MmParams};
use super::driver::{Cuda, Nvrtc};
use super::{Device, compile_cubin};
use crate::codegen::Arg;
use crate::fuse::{Kernel, MatmulK, Plan, Step};
use crate::runtime::{matmul_signature, pack};
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
        if f.len() == 6
            && let (Ok(bm), Ok(bn), Ok(bk), Ok(tm), Ok(tn)) = (
                f[1].parse(),
                f[2].parse(),
                f[3].parse(),
                f[4].parse(),
                f[5].parse(),
            )
        {
            out.insert(f[0].to_string(), MmParams { bm, bn, bk, tm, tn });
        }
    }
    out
}

fn save(all: &HashMap<String, MmParams>) {
    let mut lines: Vec<String> = all
        .iter()
        .map(|(k, p)| format!("{k} {} {} {} {} {}", p.bm, p.bn, p.bk, p.tm, p.tn))
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
pub fn cached(plan: &Plan, device: Device) -> HashMap<String, MmParams> {
    if device != Device::Cuda {
        return HashMap::new();
    }
    let Ok(c) = Cuda::open() else {
        return HashMap::new();
    };
    let all = load();
    let mut out = HashMap::new();
    for mk in matmuls(plan) {
        let sig = matmul_signature(mk);
        if let Some(p) = all.get(&key(&c.name, &sig)) {
            out.insert(sig, *p);
        }
    }
    out
}

/// Times every schedule of each matmul not yet tuned on this GPU.
pub fn tune(plan: &Plan, device: Device, log: bool) -> Result<HashMap<String, MmParams>, String> {
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
        let sig = matmul_signature(mk);
        let k = key(&c.name, &sig);
        if let Some(p) = all.get(&k) {
            out.insert(sig, *p);
            continue;
        }
        let t0 = std::time::Instant::now();
        let space = codegen::mm_space();
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
