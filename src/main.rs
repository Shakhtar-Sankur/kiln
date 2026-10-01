use kiln::interp;
use kiln::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

const USAGE: &str = "kiln: an ML compiler from ONNX graphs to fused, auto-tuned CPU and GPU kernels

USAGE:
  kiln interp MODEL.onnx REF.ref     run the reference interpreter, compare with PyTorch
  kiln opt MODEL.onnx REF.ref        optimize the graph, then interpret it and compare
  kiln plan MODEL.onnx REF.ref [--dump] [--count]
                                     optimize and fuse, evaluate the kernel IR, compare
  kiln cuda-check MODEL.onnx REF.ref [--arch 75] [--save FILE.cu]
                                     compile the CUDA kernels with NVRTC (no GPU needed)
  kiln run MODEL.onnx REF.ref [--device cpu|cuda|emu] [--threads N] [--iters N]
                                     [--tune | --no-tune] [--profile] [--no-graphs]
                                     [--json FILE --label NAME]
                                     [--no-epilogue] [--no-rows] [--no-inline] [--no-fold] [--no-fusion]
                                     compile to native kernels, run, compare, time
                                     (--device cuda: CUDA kernels on the first GPU;
                                     emu: the same kernels on kiln's CPU emulator of CUDA)
                                     (--tune searches matmul schedules; results are cached;
                                     --no-* switch off one optimization, for ablations)
";

fn die(msg: &str) -> ! {
    eprintln!("kiln: {msg}");
    std::process::exit(1)
}

/// Largest absolute difference, and the largest |reference| value.
pub fn max_diff(got: &Tensor, want: &Tensor) -> (f32, f32) {
    assert_eq!(got.shape, want.shape, "output shape");
    let (g, w) = (got.as_f32(), want.as_f32());
    let d = g
        .iter()
        .zip(w)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let m = w.iter().map(|v| v.abs()).fold(0f32, f32::max);
    (d, m)
}

fn load_with_ref(
    model: &str,
    refp: &str,
) -> (kiln::graph::Graph, kiln::reffile::Reference, interp::Feeds) {
    let g = kiln::onnx::load(Path::new(model)).unwrap_or_else(|e| die(&e));
    let r = kiln::reffile::load(Path::new(refp)).unwrap_or_else(|e| die(&e));
    let mut feeds = HashMap::new();
    for (name, t) in &r.inputs {
        let v = g
            .inputs
            .iter()
            .find(|&&i| &g.values[i].name == name)
            .unwrap_or_else(|| die(&format!("no input {name}")));
        feeds.insert(*v, t.clone());
    }
    (g, r, feeds)
}

fn flag<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).and_then(|v| v.parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some(cmd @ ("interp" | "opt")) => {
            let (model, refp) = (&args[1], &args[2]);
            let t = Instant::now();
            let mut g = kiln::onnx::load(Path::new(model)).unwrap_or_else(|e| die(&e));
            if cmd == "opt" {
                let before = g.op_histogram();
                let t = Instant::now();
                let st = kiln::passes::optimize(&mut g).unwrap_or_else(|e| die(&e));
                eprintln!("optimized in {:.2} s: {:?}", t.elapsed().as_secs_f64(), st);
                eprintln!("before: {before:?}");
                eprintln!("after:  {:?}", g.op_histogram());
            }
            eprintln!(
                "loaded {} nodes, {} values in {:.2} s",
                g.nodes.len(),
                g.values.len(),
                t.elapsed().as_secs_f64()
            );
            let r = kiln::reffile::load(Path::new(refp)).unwrap_or_else(|e| die(&e));
            let mut feeds = HashMap::new();
            for (name, t) in &r.inputs {
                let v = g
                    .inputs
                    .iter()
                    .find(|&&i| &g.values[i].name == name)
                    .unwrap_or_else(|| die(&format!("no input {name}")));
                feeds.insert(*v, t.clone());
            }
            let t = Instant::now();
            let env = interp::run(&g, &feeds).unwrap_or_else(|e| die(&e));
            eprintln!("interpreted in {:.2} s", t.elapsed().as_secs_f64());
            for (name, want) in &r.outputs {
                let o = g
                    .outputs
                    .iter()
                    .find(|&&i| &g.values[i].name == name)
                    .unwrap();
                let (d, m) = max_diff(&env[o], want);
                println!("{name}: max |diff| {d:.3e} (max |value| {m:.3})");
            }
        }
        Some("plan") => {
            let (g, r, feeds) = load_with_ref(&args[1], &args[2]);
            let mut g = g;
            kiln::passes::optimize(&mut g).unwrap_or_else(|e| die(&e));
            let plan = kiln::fuse::plan(g).unwrap_or_else(|e| die(&e));
            eprintln!("plan: {:?}", plan.stats);
            if args.iter().any(|a| a == "--dump") {
                eprint!("{}", plan.dump());
            }
            if args.iter().any(|a| a == "--count") {
                let mm: Vec<_> = plan
                    .steps
                    .iter()
                    .filter_map(|s| {
                        if let kiln::fuse::Step::Kernel(kiln::fuse::Kernel::Matmul(m)) = s {
                            Some(m)
                        } else {
                            None
                        }
                    })
                    .collect();
                let fused = mm.iter().filter(|m| m.out != m.product).count();
                println!("matmuls {} with fused epilogue {}", mm.len(), fused);
                return;
            }
            let t = Instant::now();
            let outs = kiln::eval::run(&plan, &feeds).unwrap_or_else(|e| die(&e));
            eprintln!("evaluated kernel IR in {:.2} s", t.elapsed().as_secs_f64());
            for (name, want) in &r.outputs {
                let o = plan
                    .g
                    .outputs
                    .iter()
                    .find(|&&i| &plan.g.values[i].name == name)
                    .unwrap();
                let (d, m) = max_diff(&outs[o], want);
                println!("{name}: max |diff| {d:.3e} (max |value| {m:.3})");
            }
        }
        Some("run") => {
            let threads = flag(&args, "--threads").unwrap_or_else(kiln::pool::cores);
            let iters = flag(&args, "--iters").unwrap_or(10);
            let (g, r, feeds) = load_with_ref(&args[1], &args[2]);
            let t = Instant::now();
            let mut g = g;
            kiln::passes::optimize(&mut g).unwrap_or_else(|e| die(&e));
            let has = |f: &str| args.iter().any(|a| a == f);
            let none = has("--no-fusion");
            let fopts = kiln::fuse::FuseOpts {
                epilogue: !has("--no-epilogue") && !none,
                rows: !has("--no-rows") && !none,
                inline: !has("--no-inline") && !none,
                fold: !has("--no-fold"),
            };
            let plan = kiln::fuse::plan_with(g, fopts).unwrap_or_else(|e| die(&e));
            let pstats = plan.stats.clone();
            if let Some(dev) = flag::<String>(&args, "--device")
                && dev != "cpu"
            {
                run_gpu(&args, &dev, plan, &r, &feeds, iters, t);
                return;
            }
            let params = if args.iter().any(|a| a == "--no-tune") {
                Default::default()
            } else if args.iter().any(|a| a == "--tune") {
                kiln::tune::tune(&plan, threads, true).unwrap_or_else(|e| die(&e))
            } else {
                kiln::tune::cached(&plan, threads)
            };
            let opts = kiln::runtime::Options {
                threads,
                params,
                log: false,
            };
            let mut ex = kiln::runtime::Executable::build(plan, &opts).unwrap_or_else(|e| die(&e));
            let compile_s = t.elapsed().as_secs_f64();
            let s = &ex.stats;
            eprintln!(
                "compiled in {:.2} s (C compiler {:.2} s{}): {} kernels ({} distinct, {} lines of C), {} host steps; {pstats:?}",
                t.elapsed().as_secs_f64(),
                s.compile_seconds,
                if s.cached { ", cached" } else { "" },
                s.kernels,
                s.unique_kernels,
                s.c_lines,
                s.host_steps
            );
            eprintln!(
                "memory plan: {:.1} MB arena for {:.1} MB of intermediates",
                s.arena_floats as f64 * 4e-6,
                s.naive_floats as f64 * 4e-6
            );
            let outs = ex.run(&feeds).unwrap_or_else(|e| die(&e));
            for (name, want) in &r.outputs {
                let o = ex
                    .graph()
                    .outputs
                    .iter()
                    .find(|&&i| &ex.graph().values[i].name == name)
                    .unwrap();
                let (d, m) = max_diff(&outs[o], want);
                println!("{name}: max |diff| {d:.3e} (max |value| {m:.3})");
            }
            if args.iter().any(|a| a == "--profile") {
                ex.enable_profile();
            }
            let mut times = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                ex.run(&feeds).unwrap_or_else(|e| die(&e));
                times.push(t.elapsed().as_secs_f64());
            }
            times.sort_by(f64::total_cmp);
            let prof = ex.profile_report();
            if !prof.is_empty() {
                let total: f64 = prof.iter().map(|p| p.1).sum();
                for (d, t, n) in prof.iter().take(15) {
                    println!(
                        "  {:5.1}%  {:8.3} ms  x{n:<4} {d}",
                        100.0 * t / total,
                        t * 1e3 / iters as f64
                    );
                }
            }
            if !times.is_empty() {
                let (med, min) = (times[times.len() / 2] * 1e3, times[0] * 1e3);
                println!(
                    "latency: median {med:.3} ms, min {min:.3} ms over {iters} runs, {threads} threads"
                );
                if let Some(f) = flag::<String>(&args, "--json") {
                    let label = flag::<String>(&args, "--label").unwrap_or_else(|| "kiln".into());
                    let model = Path::new(&args[1])
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
                    let d = r
                        .outputs
                        .iter()
                        .map(|(name, want)| {
                            let o = ex
                                .graph()
                                .outputs
                                .iter()
                                .find(|&&i| &ex.graph().values[i].name == name)
                                .unwrap();
                            max_diff(&outs[o], want).0
                        })
                        .fold(0f32, f32::max);
                    let s = &ex.stats;
                    let line = format!(
                        r#"{{"model":"{model}","engine":"{label}","median_ms":{med:.3},"min_ms":{min:.3},"max_diff":{d:e},"threads":{threads},"kernels":{},"arena_mb":{:.2},"intermediate_mb":{:.2},"compile_s":{compile_s:.2},"times_ms":[{}]}}"#,
                        s.kernels,
                        s.arena_floats as f64 * 4e-6,
                        s.naive_floats as f64 * 4e-6,
                        times
                            .iter()
                            .map(|t| format!("{:.3}", t * 1e3))
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    let mut fh = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&f)
                        .unwrap_or_else(|e| die(&e.to_string()));
                    use std::io::Write;
                    writeln!(fh, "{line}").ok();
                }
            }
        }
        Some("cuda-check") => {
            // Compile the plan's CUDA kernels with NVRTC for a GPU
            // architecture, without a GPU: errors, registers and spills.
            let (g, _r, _feeds) = load_with_ref(&args[1], &args[2]);
            let arch: u32 = flag(&args, "--arch").unwrap_or(75);
            let mut g = g;
            kiln::passes::optimize(&mut g).unwrap_or_else(|e| die(&e));
            let plan = kiln::fuse::plan(g).unwrap_or_else(|e| die(&e));
            let opts = kiln::gpu::GpuOptions::new(kiln::gpu::Device::Cuda);
            let gn = kiln::gpu::generate(&plan, &opts, 40);
            if let Some(f) = flag::<String>(&args, "--save") {
                std::fs::write(&f, &gn.source).unwrap_or_else(|e| die(&e.to_string()));
            }
            let nv = kiln::gpu::driver::Nvrtc::open().unwrap_or_else(|e| die(&e));
            let t = Instant::now();
            let (bin, log) = nv
                .compile(&gn.source, arch, true)
                .unwrap_or_else(|e| die(&e));
            let spills = log
                .lines()
                .filter(|l| l.contains("spill") && !l.contains(" 0 bytes spill stores"))
                .count();
            for l in log.lines().filter(|l| {
                l.contains("registers") || (l.contains("spill") && !l.contains(" 0 bytes spill"))
            }) {
                println!("{}", l.trim());
            }
            let smem = gn.kernels.iter().map(|k| k.smem).max().unwrap_or(0);
            let blocks: usize = gn
                .kernels
                .iter()
                .map(|k| (k.grid[0] * k.grid[1] * k.grid[2]) as usize)
                .sum();
            println!(
                "largest shared memory {smem} bytes; {blocks} blocks over {} launches",
                gn.kernels.len()
            );
            println!(
                "sm_{arch}: {} kernels compiled in {:.2} s, {} bytes of CUBIN, NVRTC {:?}, {} with spills",
                gn.unique.len(),
                t.elapsed().as_secs_f64(),
                bin.len(),
                nv.version,
                spills
            );
        }
        _ => print!("{USAGE}"),
    }
}

fn output_diff(
    g: &kiln::graph::Graph,
    outs: &HashMap<usize, Tensor>,
    r: &kiln::reffile::Reference,
) -> f32 {
    let mut worst = 0f32;
    for (name, want) in &r.outputs {
        let o = g
            .outputs
            .iter()
            .find(|&&i| &g.values[i].name == name)
            .unwrap();
        let (d, m) = max_diff(&outs[o], want);
        println!("{name}: max |diff| {d:.3e} (max |value| {m:.3})");
        worst = worst.max(d);
    }
    worst
}

fn run_gpu(
    args: &[String],
    dev: &str,
    plan: kiln::fuse::Plan,
    r: &kiln::reffile::Reference,
    feeds: &interp::Feeds,
    iters: usize,
    t: Instant,
) {
    use kiln::gpu::{Device, GpuExecutable, GpuOptions};
    let device = match dev {
        "cuda" => Device::Cuda,
        "emu" => Device::Emu,
        _ => die(&format!("unknown device {dev}")),
    };
    let has = |f: &str| args.iter().any(|a| a == f);
    let mut opts = GpuOptions::new(device);
    opts.graphs = !has("--no-graphs");
    if !has("--no-tune") {
        opts.mm = kiln::gpu::tune::cached(&plan, device);
    }
    if has("--tune") {
        opts.mm = kiln::gpu::tune::tune(&plan, device, true).unwrap_or_else(|e| die(&e));
    }
    let mut ex = GpuExecutable::build(plan, &opts).unwrap_or_else(|e| die(&e));
    let compile_s = t.elapsed().as_secs_f64();
    let s = &ex.stats;
    eprintln!(
        "{}: compiled in {compile_s:.2} s (NVRTC/C++ {:.2} s{}): {} kernels ({} distinct, {} lines of CUDA), {} host steps",
        s.device,
        s.compile_seconds,
        if s.cached { ", cached" } else { "" },
        s.kernels,
        s.unique_kernels,
        s.cuda_lines,
        s.host_steps
    );
    eprintln!(
        "memory plan: {:.1} MB arena for {:.1} MB of intermediates",
        s.arena_floats as f64 * 4e-6,
        s.naive_floats as f64 * 4e-6
    );
    let outs = ex.run(feeds).unwrap_or_else(|e| die(&e));
    let d = output_diff(ex.graph(), &outs, r);
    // Warm-up (graph capture, clocks), then timed runs to device completion.
    for _ in 0..3.min(iters) {
        ex.run_on_device(feeds).unwrap_or_else(|e| die(&e));
    }
    let mut times = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        ex.run_on_device(feeds).unwrap_or_else(|e| die(&e));
        times.push(t.elapsed().as_secs_f64());
    }
    if has("--profile") {
        ex.enable_profile();
        for _ in 0..iters {
            ex.run_on_device(feeds).unwrap_or_else(|e| die(&e));
        }
        let prof = ex.profile_report();
        let total: f64 = prof.iter().map(|p| p.1).sum();
        for (desc, t, n) in prof.iter().take(20) {
            println!(
                "  {:5.1}%  {:8.3} ms  x{n:<4} {desc}",
                100.0 * t / total,
                t * 1e3 / iters.max(1) as f64
            );
        }
    }
    times.sort_by(f64::total_cmp);
    if times.is_empty() {
        return;
    }
    let (med, min) = (times[times.len() / 2] * 1e3, times[0] * 1e3);
    println!(
        "latency: median {med:.3} ms, min {min:.3} ms over {iters} runs, {}",
        ex.stats.device
    );
    if let Some(f) = flag::<String>(args, "--json") {
        let label = flag::<String>(args, "--label").unwrap_or_else(|| "kiln".into());
        let model = Path::new(&args[1])
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let s = &ex.stats;
        let line = format!(
            r#"{{"model":"{model}","engine":"{label}","device":"{}","median_ms":{med:.3},"min_ms":{min:.3},"max_diff":{d:e},"kernels":{},"arena_mb":{:.2},"intermediate_mb":{:.2},"compile_s":{compile_s:.2},"times_ms":[{}]}}"#,
            s.device,
            s.kernels,
            s.arena_floats as f64 * 4e-6,
            s.naive_floats as f64 * 4e-6,
            times
                .iter()
                .map(|t| format!("{:.3}", t * 1e3))
                .collect::<Vec<_>>()
                .join(",")
        );
        let mut fh = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&f)
            .unwrap_or_else(|e| die(&e.to_string()));
        use std::io::Write;
        writeln!(fh, "{line}").ok();
    }
}
