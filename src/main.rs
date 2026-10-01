use kiln::interp;
use kiln::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

const USAGE: &str = "kiln: an ML compiler from ONNX graphs to fused, auto-tuned CPU kernels

USAGE:
  kiln interp MODEL.onnx REF.ref     run the reference interpreter, compare with PyTorch
  kiln opt MODEL.onnx REF.ref        optimize the graph, then interpret it and compare
  kiln plan MODEL.onnx REF.ref       optimize and fuse, evaluate the kernel IR, compare
  kiln run MODEL.onnx REF.ref [--threads N] [--iters N] [--tune | --no-tune] [--profile]
                                     compile to native kernels, run, compare, time
                                     (--tune searches matmul schedules; results are cached)
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
            let plan = kiln::fuse::plan(g).unwrap_or_else(|e| die(&e));
            let pstats = plan.stats.clone();
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
                println!(
                    "latency: median {:.3} ms, min {:.3} ms over {iters} runs, {threads} threads",
                    times[times.len() / 2] * 1e3,
                    times[0] * 1e3
                );
            }
        }
        _ => print!("{USAGE}"),
    }
}
