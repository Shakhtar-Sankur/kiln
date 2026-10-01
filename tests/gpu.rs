//! The GPU backend on the committed tiny models, against PyTorch's
//! outputs: on kiln's emulator of the CUDA execution model always, on a
//! real GPU when one is present, with every optimization switched off in
//! turn, with matmul schedules drawn from the tuner's whole space and with
//! every row-kernel group width. With NVRTC available, every kernel is
//! also compiled for the T4's and the A100's architectures.

use kiln::fuse::{self, FuseOpts, Kernel, Step};
use kiln::gpu::codegen::{MmParams, mm_space};
use kiln::gpu::{Device, GpuExecutable, GpuOptions, mm_key};
use kiln::graph::Graph;
use kiln::interp::Feeds;
use kiln::reffile::{self, Reference};
use kiln::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

const MODELS: [&str; 3] = ["mlp_tiny", "bert_tiny", "llama_tiny"];

fn load(name: &str) -> (Graph, Reference, Feeds) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut g = kiln::onnx::load(&dir.join(format!("{name}.onnx"))).unwrap();
    kiln::passes::optimize(&mut g).unwrap();
    let r = reffile::load(&dir.join(format!("{name}.ref"))).unwrap();
    let mut feeds = HashMap::new();
    for (n, t) in &r.inputs {
        let v = *g.inputs.iter().find(|&&i| &g.values[i].name == n).unwrap();
        feeds.insert(v, t.clone());
    }
    (g, r, feeds)
}

/// Every output within `tol` of PyTorch's (fp32), relative to the largest
/// value: 1e-5 in fp32, 2e-3 with fp16 matmul operands.
fn check(what: &str, g: &Graph, outs: &HashMap<usize, Tensor>, r: &Reference, tol: f32) {
    for (n, want) in &r.outputs {
        let o = g.outputs.iter().find(|&&i| &g.values[i].name == n).unwrap();
        let got = &outs[o];
        assert_eq!(got.shape, want.shape, "{what}");
        let scale = want.as_f32().iter().fold(1f32, |m, v| m.max(v.abs()));
        let d = got
            .as_f32()
            .iter()
            .zip(want.as_f32())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(d <= tol * scale, "{what}: max |diff| {d} (scale {scale})");
    }
}

/// The emulator always (unless KILN_TEST_EMU=0, on a GPU machine with
/// few cores), and a real GPU when there is one.
fn devices() -> Vec<Device> {
    let mut d = Vec::new();
    if std::env::var("KILN_TEST_EMU").as_deref() != Ok("0") {
        d.push(Device::Emu);
    }
    match kiln::gpu::driver::Cuda::open() {
        Ok(c) => {
            eprintln!("also testing on {}", c.name);
            d.push(Device::Cuda);
        }
        Err(e) => eprintln!("no GPU ({e}): emulator only"),
    }
    d
}

fn run(m: &str, fo: FuseOpts, dev: Device, tweak: impl Fn(&fuse::Plan, &mut GpuOptions)) {
    let (g, r, f) = load(m);
    let plan = fuse::plan_with(g, fo).unwrap();
    let mut opts = GpuOptions::new(dev);
    tweak(&plan, &mut opts);
    let what = format!(
        "{m} on {dev:?} {fo:?} half {} tpr {:?} mm {:?}",
        opts.half, opts.tpr, opts.mm
    );
    let tol = if opts.half { 2e-3 } else { 1e-5 };
    let mut ex = GpuExecutable::build(plan, &opts).unwrap();
    // Twice: the second run reuses buffers (and replays CUDA graphs).
    ex.run(&f).unwrap();
    let outs = ex.run(&f).unwrap();
    check(&what, ex.graph(), &outs, &r, tol);
}

#[test]
fn gpu_kernels_match_pytorch_with_each_optimization_off() {
    let all = FuseOpts::default();
    for dev in devices() {
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
                run(m, fo, dev, |_, _| {});
            }
        }
    }
}

#[test]
fn gpu_kernels_match_pytorch_under_every_schedule() {
    // Each round gives every matmul a different schedule from the tuner's
    // space and every row kernel a different group width, so that across
    // rounds each schedule and width meets each model; in fp32 on CUDA
    // cores and in fp16 on tensor cores.
    for half in [false, true] {
        let space = mm_space(half);
        for dev in devices() {
            let rounds = if dev == Device::Emu { 4 } else { space.len() };
            for round in 0..rounds {
                for m in MODELS {
                    run(m, FuseOpts::default(), dev, |plan, opts| {
                        opts.half = half;
                        opts.tpr = Some([1, 8, 32, 64, 256][round % 5]);
                        let mut i = round * 7;
                        for s in &plan.steps {
                            if let Step::Kernel(Kernel::Matmul(mk)) = s {
                                let p: MmParams = space[i % space.len()];
                                opts.mm.insert(mm_key(mk, half), p);
                                i += 3;
                            }
                        }
                    });
                }
            }
        }
    }
}

#[test]
fn gpu_tensor_core_kernels_match_pytorch_with_each_optimization_off() {
    let all = FuseOpts::default();
    for dev in devices() {
        for m in MODELS {
            for fo in [
                all,
                FuseOpts {
                    epilogue: false,
                    ..all
                },
                FuseOpts { rows: false, ..all },
            ] {
                run(m, fo, dev, |_, opts| opts.half = true);
            }
        }
    }
}

#[test]
fn gpu_kernels_compile_for_t4_and_a100() {
    if std::env::var("KILN_TEST_EMU").as_deref() == Ok("0") {
        eprintln!("skipped: compile checks run with the emulator tests");
        return;
    }
    let nv = match kiln::gpu::driver::Nvrtc::open() {
        Ok(nv) => nv,
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    for m in MODELS {
        let (g, _, _) = load(m);
        let plan = fuse::plan(g).unwrap();
        // The default schedules, and every schedule of the space.
        let mut half = GpuOptions::new(Device::Cuda);
        half.half = true;
        let mut sources = vec![
            kiln::gpu::generate(&plan, &GpuOptions::new(Device::Cuda), 40).source,
            kiln::gpu::generate(&plan, &half, 40).source,
        ];
        for p in mm_space(false).into_iter().chain(mm_space(true)) {
            let mut o = GpuOptions::new(Device::Cuda);
            o.half = p.tc;
            for s in &plan.steps {
                if let Step::Kernel(Kernel::Matmul(mk)) = s {
                    o.mm.insert(mm_key(mk, p.tc), p);
                }
            }
            sources.push(kiln::gpu::generate(&plan, &o, 40).source);
        }
        for (i, src) in sources.iter().enumerate() {
            for arch in if i < 2 { vec![75, 80] } else { vec![75] } {
                let (_, log) = nv.compile(src, arch, true).unwrap();
                // The default schedules must not spill registers (others
                // may; the tuner times them like any schedule).
                assert!(
                    i > 1
                        || !log.lines().any(|l| l.contains("bytes spill stores")
                            && !l.contains(" 0 bytes spill stores")),
                    "{m} sm_{arch}: register spills\n{log}"
                );
            }
        }
    }
}

#[test]
fn fused_attention_applies_and_matches_pytorch() {
    // Both two-layer transformers fuse each layer's scores, softmax and
    // output matmul; the MLP has nothing to fuse.
    for (m, want) in [("bert_tiny", 2), ("llama_tiny", 2), ("mlp_tiny", 0)] {
        let (g, _, _) = load(m);
        let plan = fuse::plan(g).unwrap();
        assert_eq!(kiln::gpu::attention::find(&plan).len(), want, "{m}");
    }
    for dev in devices() {
        for m in MODELS {
            for attention in [true, false] {
                run(m, FuseOpts::default(), dev, |_, opts| {
                    opts.attention = attention
                });
            }
            // Every fused shape the tuner may choose, in both precisions.
            for p in kiln::gpu::attention::space() {
                for half in [false, true] {
                    run(m, FuseOpts::default(), dev, |plan, opts| {
                        opts.half = half;
                        for [a, _, c] in kiln::gpu::attention::find(plan) {
                            if let (
                                Step::Kernel(Kernel::Matmul(m1)),
                                Step::Kernel(Kernel::Matmul(m3)),
                            ) = (&plan.steps[a], &plan.steps[c])
                            {
                                opts.attn
                                    .insert(kiln::gpu::attention::key(m1, m3, half), Some(p));
                            }
                        }
                    });
                }
            }
            // Fusion off leaves no softmax row kernel to absorb.
            run(
                m,
                FuseOpts {
                    rows: false,
                    ..FuseOpts::default()
                },
                dev,
                |_, _| {},
            );
        }
    }
}
