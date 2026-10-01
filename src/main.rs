use kiln::interp;
use kiln::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

const USAGE: &str = "kiln: an ML compiler from ONNX graphs to fused, auto-tuned CPU kernels

USAGE:
  kiln interp MODEL.onnx REF.ref     run the reference interpreter, compare with PyTorch
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("interp") => {
            let (model, refp) = (&args[1], &args[2]);
            let t = Instant::now();
            let g = kiln::onnx::load(Path::new(model)).unwrap_or_else(|e| die(&e));
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
        _ => print!("{USAGE}"),
    }
}
