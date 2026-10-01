//! End-to-end checks on the committed tiny models (a 2-block MLP, a
//! 2-layer BERT with padding masks, a 2-layer Llama with grouped-query
//! attention and rotary embeddings; random weights) against PyTorch's
//! outputs, at every level: the interpreter, the optimized graph, the
//! fused plan (evaluated from the IR), and native code, with every
//! optimization switched off in turn and with arbitrary matmul schedules.

use kiln::codegen::Params;
use kiln::fuse::{self, FuseOpts, Kernel, Step};
use kiln::graph::Graph;
use kiln::interp::{self, Feeds};
use kiln::reffile::{self, Reference};
use kiln::runtime::{self, Executable, Options};
use kiln::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

const MODELS: [&str; 3] = ["mlp_tiny", "bert_tiny", "llama_tiny"];

fn load(name: &str) -> (Graph, Reference, Feeds) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let g = kiln::onnx::load(&dir.join(format!("{name}.onnx"))).unwrap();
    let r = reffile::load(&dir.join(format!("{name}.ref"))).unwrap();
    let mut feeds = HashMap::new();
    for (n, t) in &r.inputs {
        let v = *g.inputs.iter().find(|&&i| &g.values[i].name == n).unwrap();
        feeds.insert(v, t.clone());
    }
    (g, r, feeds)
}

/// Every output within 1e-5 of PyTorch's, relative to the largest value.
fn check(name: &str, what: &str, g: &Graph, outs: &HashMap<usize, Tensor>, r: &Reference) {
    for (n, want) in &r.outputs {
        let o = g.outputs.iter().find(|&&i| &g.values[i].name == n).unwrap();
        let got = &outs[o];
        assert_eq!(got.shape, want.shape, "{name} {what}");
        let scale = want.as_f32().iter().fold(1f32, |m, v| m.max(v.abs()));
        let d = got
            .as_f32()
            .iter()
            .zip(want.as_f32())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            d <= 1e-5 * scale,
            "{name} {what}: max |diff| {d} (scale {scale})"
        );
    }
}

fn optimized(name: &str) -> (Graph, Reference, Feeds) {
    let (mut g, r, f) = load(name);
    kiln::passes::optimize(&mut g).unwrap();
    (g, r, f)
}

#[test]
fn interpreter_matches_pytorch() {
    for m in MODELS {
        let (g, r, f) = load(m);
        check(m, "interpreter", &g, &interp::run(&g, &f).unwrap(), &r);
    }
}

#[test]
fn optimized_graph_matches_pytorch() {
    for m in MODELS {
        let (g, r, f) = optimized(m);
        check(m, "optimized graph", &g, &interp::run(&g, &f).unwrap(), &r);
    }
}

#[test]
fn fused_plan_matches_pytorch() {
    for m in MODELS {
        let (g, r, f) = optimized(m);
        let plan = fuse::plan(g).unwrap();
        let outs = kiln::eval::run(&plan, &f).unwrap();
        check(m, "fused plan", &plan.g, &outs, &r);
    }
}

fn native(m: &str, fo: FuseOpts, params: impl Fn(&fuse::Plan) -> HashMap<String, Params>) {
    let (g, r, f) = optimized(m);
    let plan = fuse::plan_with(g, fo).unwrap();
    let opts = Options {
        threads: 3,
        params: params(&plan),
        log: false,
    };
    let mut ex = Executable::build(plan, &opts).unwrap();
    // Twice: buffers are reused between runs.
    ex.run(&f).unwrap();
    let outs = ex.run(&f).unwrap();
    check(m, &format!("native {fo:?}"), ex.graph(), &outs, &r);
}

#[test]
fn native_code_matches_pytorch_with_each_optimization_off() {
    let all = FuseOpts::default();
    for m in MODELS {
        for fo in [
            all,
            FuseOpts {
                epilogue: false,
                ..all
            },
            FuseOpts { rows: false, ..all },
            FuseOpts {
                inline: false,
                ..all
            },
            FuseOpts { fold: false, ..all },
        ] {
            native(m, fo, |_| HashMap::new());
        }
    }
}

#[test]
fn native_code_matches_pytorch_under_arbitrary_schedules() {
    // Register tiles, K blocks, task splits and prefetch distances from the
    // tuner's space, drawn pseudo-randomly per matmul shape.
    let w = kiln::jit::vector_width();
    let mut seed = 7u64;
    let mut pick = |n: usize| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as usize % n
    };
    for round in 0..4 {
        for m in MODELS {
            let mut choices = Vec::new();
            for _ in 0..16 {
                choices.push((pick(4), pick(4), pick(3), pick(3), pick(3), pick(3)));
            }
            native(m, FuseOpts::default(), |plan| {
                let mut out = HashMap::new();
                let mut i = round;
                for s in &plan.steps {
                    if let Step::Kernel(Kernel::Matmul(mk)) = s {
                        let (a, b, c, d, e, f) = choices[i % choices.len()];
                        i += 1;
                        let mr = [1, 3, 4, 8][a];
                        let nr = w * (1 + b);
                        let kc = [mk.k, 7, 32][c].max(1);
                        let mc = mr * (1 + d * 2);
                        let ncp = 1 + e;
                        let pf = [0, 4, 16][f];
                        out.insert(
                            runtime::matmul_signature(mk),
                            Params {
                                mr,
                                nr,
                                mc,
                                ncp,
                                kc,
                                pf,
                            },
                        );
                    }
                }
                out
            });
        }
    }
}

#[test]
fn memory_plan_never_overlaps_live_buffers() {
    let mut seed = 1u64;
    let mut rnd = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    for _ in 0..200 {
        let bufs: Vec<(usize, usize, usize, usize)> = (0..40)
            .map(|v| {
                let s = rnd(30);
                (v, 1 + rnd(5000), s, s + rnd(10))
            })
            .collect();
        let (off, total) = runtime::plan_memory(&bufs);
        for a in &bufs {
            for b in &bufs {
                if a.0 < b.0 && a.2 <= b.3 && b.2 <= a.3 {
                    let (oa, ob) = (off[&a.0], off[&b.0]);
                    assert!(
                        oa + a.1 <= ob || ob + b.1 <= oa,
                        "live buffers {} and {} overlap",
                        a.0,
                        b.0
                    );
                }
            }
            assert!(off[&a.0] + a.1 <= total);
        }
    }
}
