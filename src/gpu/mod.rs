//! The GPU backend: CUDA kernels generated from the same fusion plan as
//! the CPU backend, compiled at run time with NVRTC and launched through
//! the driver API, with every intermediate in one device arena, weights
//! uploaded once, and the launches between host steps recorded once into
//! CUDA graphs and replayed. The emulator device compiles the identical
//! kernel source as C++ against a model of CUDA's execution model, so the
//! kernels are tested without a GPU.

pub mod attention;
pub mod codegen;
pub mod driver;
pub mod tune;

use crate::codegen::Arg;
use crate::fuse::{Kernel, Plan, Step};
use crate::interp;
use crate::jit::{self, Library};
use crate::runtime::{arena_layout, matmul_signature, pack, pack_half_t};
use crate::tensor::{DType, Tensor, numel};
use codegen::{GpuKernel, MmParams};
use driver::{Cuda, DevPtr, Function, GraphExec, Nvrtc};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cuda,
    Emu,
}

#[derive(Clone, Debug)]
pub struct GpuOptions {
    pub device: Device,
    /// Matmul schedules by kernel signature (else a default per shape).
    pub mm: HashMap<String, MmParams>,
    /// Threads per row for every row kernel (else by row width).
    pub tpr: Option<usize>,
    /// Record kernel launches into CUDA graphs.
    pub graphs: bool,
    /// Matmuls on fp16 tensor cores (fp32 accumulation; everything else
    /// stays fp32).
    pub half: bool,
    /// Fuse scores, softmax and output matmul into one kernel.
    pub attention: bool,
    /// The tuner's choice per attention pattern (by `attention::key`):
    /// fused with a shape, or not fused. Untuned patterns are fused with
    /// the default shape.
    pub attn: HashMap<String, Option<attention::AttnParams>>,
}

/// The schedule table key of a matmul: its signature, and the precision.
pub fn mm_key(mk: &crate::fuse::MatmulK, half: bool) -> String {
    let s = matmul_signature(mk);
    if half { format!("{s}_f16") } else { s }
}

impl GpuOptions {
    pub fn new(device: Device) -> GpuOptions {
        GpuOptions {
            device,
            mm: HashMap::new(),
            tpr: None,
            graphs: true,
            half: false,
            attention: true,
            attn: HashMap::new(),
        }
    }
}

pub struct GpuStats {
    pub device: String,
    pub kernels: usize,
    pub unique_kernels: usize,
    pub host_steps: usize,
    pub arena_floats: usize,
    pub naive_floats: usize,
    pub compile_seconds: f64,
    pub cached: bool,
    pub cuda_lines: usize,
}

type EmuFn = unsafe extern "C" fn(*mut *mut f32, u32, u32, u32, u32, u32);

enum Dev {
    Cuda(Box<Cuda>),
    /// Emulated device memory: host allocations, addressed by pointer.
    Emu(Vec<Vec<f32>>),
}

enum Funcs {
    Cuda(Vec<Function>),
    /// The functions, and the library that keeps them loaded.
    Emu(Vec<EmuFn>, #[allow(dead_code)] Library),
}

struct Launch {
    func: usize,
    grid: [u32; 3],
    block: u32,
    smem: u32,
    args: Vec<DevPtr>,
    name: String,
    /// A fused attention kernel.
    fused: bool,
}

enum GStep {
    Kernel(Launch),
    /// A kernel step absorbed into a fused kernel.
    Nop,
    Host(usize),
    /// A host Gather of rows of a constant f32 table, done on the device:
    /// only the indices are uploaded.
    Gather {
        node: usize,
        idx: DevPtr,
        launch: Launch,
        dim: i64,
    },
}

impl Dev {
    fn alloc(&mut self, floats: usize) -> Result<DevPtr, String> {
        match self {
            Dev::Cuda(c) => {
                let p = c.alloc(floats * 4)?;
                c.fill_nan(p, floats)?;
                Ok(p)
            }
            Dev::Emu(m) => {
                m.push(vec![f32::NAN; floats.max(1)]);
                Ok(m.last_mut().unwrap().as_mut_ptr() as DevPtr)
            }
        }
    }

    fn upload(&mut self, data: &[f32]) -> Result<DevPtr, String> {
        let p = self.alloc(data.len())?;
        match self {
            Dev::Cuda(c) => c.upload(p, data)?,
            // SAFETY: p was just allocated with data.len() floats.
            Dev::Emu(_) => unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut f32, data.len())
            },
        }
        Ok(p)
    }

    fn upload_half(&mut self, data: &[u16]) -> Result<DevPtr, String> {
        let mut words = vec![0f32; data.len().div_ceil(2)];
        // SAFETY: words has room for every half (two per f32).
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                words.as_mut_ptr() as *mut u8,
                data.len() * 2,
            )
        };
        self.upload(&words)
    }

    fn h2d(&self, dst: DevPtr, src: &[f32]) -> Result<(), String> {
        match self {
            Dev::Cuda(c) => c.h2d_async(dst, src.as_ptr(), src.len()),
            Dev::Emu(_) => {
                // SAFETY: dst points into an emulated allocation large enough.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut f32, src.len()) };
                Ok(())
            }
        }
    }

    fn d2h(&self, src: DevPtr, n: usize) -> Result<Vec<f32>, String> {
        let mut out = vec![0f32; n];
        match self {
            Dev::Cuda(c) => {
                c.d2h_async(out.as_mut_ptr(), src, n)?;
                c.sync()?;
            }
            // SAFETY: src points into an emulated allocation of n floats.
            Dev::Emu(_) => unsafe {
                std::ptr::copy_nonoverlapping(src as *const f32, out.as_mut_ptr(), n)
            },
        }
        Ok(out)
    }

    fn sync(&self) -> Result<(), String> {
        match self {
            Dev::Cuda(c) => c.sync(),
            Dev::Emu(_) => Ok(()),
        }
    }
}

/// The CUDA source of a plan: one kernel per kernel step, identical
/// kernels shared.
pub struct Generated {
    pub kernels: Vec<GpuKernel>,
    /// Function name of each kernel step.
    pub names: Vec<String>,
    pub unique: Vec<String>,
    pub source: String,
}

pub fn generate(plan: &Plan, opts: &GpuOptions, sms: usize) -> Generated {
    let mut kernels = Vec::new();
    // Fused attention: the first step of each pattern gets the fused
    // kernel, the other two none.
    let mut fused: HashMap<usize, GpuKernel> = HashMap::new();
    let mut skip = std::collections::HashSet::new();
    if opts.attention {
        for [a, b, c] in attention::find(plan) {
            if let (
                Step::Kernel(Kernel::Matmul(m1)),
                Step::Kernel(Kernel::Loop(lk)),
                Step::Kernel(Kernel::Matmul(m3)),
            ) = (&plan.steps[a], &plan.steps[b], &plan.steps[c])
                && let Some(p) = match opts.attn.get(&attention::key(m1, m3, opts.half)) {
                    Some(choice) => *choice,
                    None => attention::default_params(m1, lk, m3),
                }
                && let Some(k) = attention::attention_kernel(m1, lk, m3, p)
            {
                fused.insert(a, k);
                skip.insert(b);
                skip.insert(c);
            }
        }
    }
    for (si, step) in plan.steps.iter().enumerate() {
        if let Some(k) = fused.remove(&si) {
            kernels.push(k);
            continue;
        }
        if skip.contains(&si) {
            kernels.push(GpuKernel {
                src: String::new(),
                args: Vec::new(),
                outs: Vec::new(),
                grid: [0, 0, 0],
                block: 0,
                smem: 0,
            });
            continue;
        }
        if let Step::Kernel(k) = step {
            kernels.push(match k {
                Kernel::Loop(l) => codegen::loop_kernel(l, opts.tpr),
                Kernel::Matmul(mk) => {
                    let p = opts
                        .mm
                        .get(&mm_key(mk, opts.half))
                        .copied()
                        .filter(|p| p.tc == opts.half)
                        .unwrap_or_else(|| codegen::default_mm(mk, sms, opts.half));
                    codegen::matmul_kernel(mk, p)
                }
            });
        }
    }
    let mut by_src: HashMap<String, String> = HashMap::new();
    let mut unique = Vec::new();
    let mut source = String::from(codegen::PRELUDE);
    source.push_str(codegen::GATHER);
    let mut names = Vec::new();
    for k in &kernels {
        if k.src.is_empty() {
            names.push(String::new());
            continue;
        }
        let name = by_src
            .entry(k.src.clone())
            .or_insert_with(|| {
                let n = format!("k{}", unique.len());
                source.push('\n');
                source.push_str(&k.src.replace("KNAME", &n));
                unique.push(n.clone());
                n
            })
            .clone();
        names.push(name);
    }
    Generated {
        kernels,
        names,
        unique,
        source,
    }
}

/// Compiles CUDA source for sm_{arch} with NVRTC, cached by content.
pub fn compile_cubin(nv: &Nvrtc, src: &str, arch: u32) -> Result<(Vec<u8>, bool), String> {
    let key = format!(
        "{:016x}",
        jit::fnv(&format!("sm_{arch} nvrtc {:?}\n{src}", nv.version))
    );
    let dir = jit::cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("g{key}.cubin"));
    if let Ok(b) = std::fs::read(&path) {
        return Ok((b, true));
    }
    let (bin, _) = nv.compile(src, arch, false)?;
    let tmp = dir.join(format!("g{key}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, &bin).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    Ok((bin, false))
}

/// Flags for the emulator build of the kernels.
const EMU_FLAGS: &[&str] = &["-O1", "-std=c++17", "-fPIC", "-shared", "-pthread", "-w"];

pub struct GpuExecutable {
    plan: Plan,
    dev: Dev,
    funcs: Funcs,
    steps: Vec<GStep>,
    arena: DevPtr,
    offsets: HashMap<usize, usize>,
    /// Instantiated graphs of the kernel runs between host steps, by the
    /// index of the run's first step.
    graphs: HashMap<usize, GraphExec>,
    use_graphs: bool,
    pub stats: GpuStats,
    pub profile: Option<Vec<f64>>,
}

impl GpuExecutable {
    pub fn build(plan: Plan, opts: &GpuOptions) -> Result<GpuExecutable, String> {
        let t = Instant::now();
        let mut dev = match opts.device {
            Device::Cuda => Dev::Cuda(Box::new(Cuda::open()?)),
            Device::Emu => Dev::Emu(Vec::new()),
        };
        let (sms, devname) = match &dev {
            Dev::Cuda(c) => (c.sms, format!("{} (sm_{}{})", c.name, c.cc.0, c.cc.1)),
            Dev::Emu(_) => (40, "emulator".into()),
        };
        let devname = if opts.half {
            format!("{devname}, fp16 tensor cores")
        } else {
            devname
        };
        let gn = generate(&plan, opts, sms);
        let (funcs, cached) = match &dev {
            Dev::Cuda(c) => {
                let nv = Nvrtc::open()?;
                let arch = (c.cc.0 * 10 + c.cc.1) as u32;
                let (bin, cached) = compile_cubin(&nv, &gn.source, arch)?;
                let mut names = gn.unique.clone();
                names.push("kgather".into());
                let fs = c.load(&bin, &names)?;
                (Funcs::Cuda(fs), cached)
            }
            Dev::Emu(_) => {
                let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
                let lib = jit::compile_with(&cxx, EMU_FLAGS, &["-lm"], "cpp", &gn.source)?;
                let mut fs = Vec::new();
                for n in gn.unique.iter().map(String::as_str).chain(["kgather"]) {
                    let p = lib.symbol(&format!("{n}_emu"))?;
                    // SAFETY: every emulator entry point has this signature.
                    fs.push(unsafe { std::mem::transmute::<*mut std::ffi::c_void, EmuFn>(p) });
                }
                let cached = lib.cached;
                (Funcs::Emu(fs, lib), cached)
            }
        };
        let compile_seconds = t.elapsed().as_secs_f64();
        if let (Dev::Cuda(c), Funcs::Cuda(fs)) = (&dev, &funcs) {
            for (k, name) in gn.kernels.iter().zip(&gn.names) {
                if k.smem > 48 * 1024 {
                    let i = gn.unique.iter().position(|n| n == name).unwrap();
                    c.allow_smem(fs[i], k.smem)?;
                }
            }
        }
        let kargs: Vec<(&[Arg], Vec<usize>)> = gn
            .kernels
            .iter()
            .map(|k| (k.args.as_slice(), k.outs.clone()))
            .collect();
        let (offsets, total, naive) = arena_layout(&plan, &kargs);
        let arena = dev.alloc(total.max(16))?;
        let g = &plan.g;
        let mut consts: HashMap<usize, DevPtr> = HashMap::new();
        let mut packed: HashMap<(usize, crate::ir::Lin), DevPtr> = HashMap::new();
        let mut packed_half: HashMap<(usize, crate::ir::Lin), DevPtr> = HashMap::new();
        let mut steps = Vec::new();
        let mut kernel_steps = gn.kernels.iter().zip(&gn.names);
        let mut host = 0;
        for step in &plan.steps {
            match step {
                Step::Host(p) => {
                    host += 1;
                    let n = &g.nodes[*p];
                    // Gather along axis 0 of a constant f32 table, by
                    // indices only known at run time, into the arena.
                    let table = n.inputs.first().copied().flatten().and_then(|t| g.konst(t));
                    let ind = n.inputs.get(1).copied().flatten();
                    let out = n.outputs[0];
                    if n.op == "Gather"
                        && n.attr_int("axis", 0) == 0
                        && let (Some(t), Some(ind)) = (table, ind)
                        && t.dtype() == DType::F32
                        && g.konst(ind).is_none()
                        && plan.materialized[out]
                        && offsets.contains_key(&out)
                    {
                        let tp = match consts.get(&n.inputs[0].unwrap()) {
                            Some(&p) => p,
                            None => {
                                let p = dev.upload(t.as_f32())?;
                                consts.insert(n.inputs[0].unwrap(), p);
                                p
                            }
                        };
                        let count = numel(g.shape(ind));
                        let row = t.len() / t.shape[0];
                        let idx = dev.alloc(count)?;
                        steps.push(GStep::Gather {
                            node: *p,
                            idx,
                            launch: Launch {
                                func: gn.unique.len(),
                                grid: [(count * row).div_ceil(256) as u32, 1, 1],
                                block: 256,
                                smem: 0,
                                args: vec![
                                    arena + 4 * offsets[&out] as DevPtr,
                                    tp,
                                    idx,
                                    count as DevPtr,
                                    row as DevPtr,
                                ],
                                name: "kgather".into(),
                                fused: false,
                            },
                            dim: t.shape[0] as i64,
                        });
                    } else {
                        steps.push(GStep::Host(*p));
                    }
                }
                Step::Kernel(k) => {
                    let (gk, name) = kernel_steps.next().unwrap();
                    if gk.src.is_empty() {
                        steps.push(GStep::Nop);
                        continue;
                    }
                    let mut args = Vec::new();
                    for a in &gk.args {
                        args.push(match a {
                            Arg::Value(v) => {
                                if let Some(t) = g.konst(*v) {
                                    match consts.get(v) {
                                        Some(&p) => p,
                                        None => {
                                            let p = dev.upload(&t.to_f32())?;
                                            consts.insert(*v, p);
                                            p
                                        }
                                    }
                                } else {
                                    let off = *offsets.get(v).ok_or_else(|| {
                                        format!("value {} has no buffer", g.values[*v].name)
                                    })?;
                                    arena + 4 * off as DevPtr
                                }
                            }
                            Arg::PackedHalf {
                                value,
                                lin,
                                k: kk,
                                n,
                            } => {
                                let key = (*value, lin.clone());
                                match packed_half.get(&key) {
                                    Some(&p) => p,
                                    None => {
                                        let (vk, vj) = match k {
                                            Kernel::Matmul(mk) => (mk.vk, mk.vj),
                                            Kernel::Loop(_) => unreachable!(),
                                        };
                                        let data = pack_half_t(
                                            g.konst(*value).unwrap(),
                                            lin,
                                            *kk,
                                            *n,
                                            vk,
                                            vj,
                                        );
                                        let p = dev.upload_half(&data)?;
                                        packed_half.insert(key, p);
                                        p
                                    }
                                }
                            }
                            Arg::Packed {
                                value,
                                lin,
                                k: kk,
                                n,
                                nr,
                            } => {
                                let key = (*value, lin.clone());
                                match packed.get(&key) {
                                    Some(&p) => p,
                                    None => {
                                        let (vk, vj) = match k {
                                            Kernel::Matmul(mk) => (mk.vk, mk.vj),
                                            Kernel::Loop(_) => unreachable!(),
                                        };
                                        let data = pack(
                                            g.konst(*value).unwrap(),
                                            lin,
                                            *kk,
                                            *n,
                                            *nr,
                                            vk,
                                            vj,
                                        );
                                        let p = dev.upload(&data)?;
                                        packed.insert(key, p);
                                        p
                                    }
                                }
                            }
                        });
                    }
                    steps.push(GStep::Kernel(Launch {
                        func: gn.unique.iter().position(|n| n == name).unwrap(),
                        grid: gk.grid,
                        block: gk.block,
                        smem: gk.smem,
                        args,
                        name: name.clone(),
                        fused: matches!(k, Kernel::Matmul(_))
                            && gk.grid[2] == 1
                            && gk.src.contains("*Ss ="),
                    }));
                }
            }
        }
        dev.sync()?;
        let stats = GpuStats {
            device: devname,
            kernels: gn.kernels.iter().filter(|k| !k.src.is_empty()).count(),
            unique_kernels: gn.unique.len(),
            host_steps: host,
            arena_floats: total,
            naive_floats: naive,
            compile_seconds,
            cached,
            cuda_lines: gn.source.lines().count(),
        };
        Ok(GpuExecutable {
            plan,
            dev,
            funcs,
            steps,
            arena,
            offsets,
            graphs: HashMap::new(),
            use_graphs: opts.graphs && opts.device == Device::Cuda,
            stats,
            profile: None,
        })
    }

    fn launch(&self, l: &Launch) -> Result<(), String> {
        match (&self.dev, &self.funcs) {
            (Dev::Cuda(c), Funcs::Cuda(fs)) => {
                c.launch(fs[l.func], l.grid, l.block, l.smem, &l.args)
            }
            (Dev::Emu(_), Funcs::Emu(fs, _)) => {
                let mut ptrs: Vec<*mut f32> = l.args.iter().map(|&a| a as *mut f32).collect();
                // SAFETY: the arguments are emulated device buffers matching
                // the kernel's parameters.
                unsafe {
                    fs[l.func](
                        ptrs.as_mut_ptr(),
                        l.grid[0],
                        l.grid[1],
                        l.grid[2],
                        l.block,
                        l.smem,
                    )
                };
                Ok(())
            }
            _ => unreachable!(),
        }
    }

    fn value_ptr(&self, v: usize) -> DevPtr {
        self.arena + 4 * self.offsets[&v] as DevPtr
    }

    /// Runs the model; outputs stay on the device until `outputs`.
    fn execute(&mut self, feeds: &interp::Feeds) -> Result<HashMap<usize, Tensor>, String> {
        let mut env: HashMap<usize, Tensor> = HashMap::new();
        let mut staged: Vec<std::borrow::Cow<'_, [f32]>> = Vec::new();
        for (&v, t) in feeds {
            // Kernels read booleans and integers as carried f32; the host
            // keeps their real type.
            if self.plan.materialized[v] {
                let data = t.to_f32();
                self.dev.h2d(self.value_ptr(v), &data)?;
                staged.push(data);
            }
            if t.dtype() != DType::F32 || !self.plan.materialized[v] {
                env.insert(v, t.clone());
            }
        }
        let mut si = 0;
        while si < self.steps.len() {
            if let GStep::Gather {
                node,
                idx,
                launch,
                dim,
            } = &self.steps[si]
            {
                let g = &self.plan.g;
                let iv = g.nodes[*node].inputs[1].unwrap();
                let ind = env.get(&iv).ok_or_else(|| {
                    format!("gather indices {} not on the host", g.values[iv].name)
                })?;
                let mut words = Vec::with_capacity(ind.len());
                for mut i in ind.to_i64() {
                    if i < 0 {
                        i += dim;
                    }
                    if !(0..*dim).contains(&i) {
                        return Err(format!("gather index {i} out of range 0..{dim}"));
                    }
                    words.push(f32::from_bits(i as u32));
                }
                self.dev.h2d(*idx, &words)?;
                self.launch(launch)?;
                si += 1;
                continue;
            }
            let host = match &self.steps[si] {
                GStep::Host(p) => Some(*p),
                GStep::Kernel(_) | GStep::Gather { .. } | GStep::Nop => None,
            };
            match host {
                Some(p) => {
                    let g = &self.plan.g;
                    let n = &g.nodes[p];
                    let mut owned: Vec<Option<Tensor>> = Vec::new();
                    for i in &n.inputs {
                        owned.push(match i {
                            Some(i) if g.konst(*i).is_none() && !env.contains_key(i) => {
                                let data = self.dev.d2h(self.value_ptr(*i), numel(g.shape(*i)))?;
                                Some(Tensor::f32(g.shape(*i).to_vec(), data))
                            }
                            _ => None,
                        });
                    }
                    let refs: Vec<Option<&Tensor>> = n
                        .inputs
                        .iter()
                        .zip(&owned)
                        .map(|(i, o)| {
                            i.map(|i| g.konst(i).or_else(|| env.get(&i)).or(o.as_ref()).unwrap())
                        })
                        .collect();
                    let outs = interp::eval(n, &refs)
                        .map_err(|e| format!("step {si} host {}: {e}", n.op))?;
                    for (&o, t) in n.outputs.iter().zip(outs) {
                        if self.plan.materialized[o] {
                            self.dev.h2d(self.value_ptr(o), &t.to_f32())?;
                            self.dev.sync()?;
                        }
                        if t.dtype() != DType::F32 || !self.plan.materialized[o] {
                            env.insert(o, t);
                        }
                    }
                    si += 1;
                }
                None => {
                    let mut end = si;
                    while end < self.steps.len()
                        && matches!(self.steps[end], GStep::Kernel(_) | GStep::Nop)
                    {
                        end += 1;
                    }
                    self.run_kernels(si, end)?;
                    si = end;
                }
            }
        }
        Ok(env)
    }

    fn run_kernels(&mut self, a: usize, b: usize) -> Result<(), String> {
        if let Some(prof) = &self.profile {
            let mut prof = prof.clone();
            for (step, t) in self.steps[a..b].iter().zip(&mut prof[a..b]) {
                let GStep::Kernel(l) = step else {
                    continue;
                };
                let secs = match &self.dev {
                    Dev::Cuda(c) => c.time(&mut || self.launch(l))? as f64 * 1e-3,
                    Dev::Emu(_) => {
                        let t = Instant::now();
                        self.launch(l)?;
                        t.elapsed().as_secs_f64()
                    }
                };
                *t += secs;
            }
            self.profile = Some(prof);
            return Ok(());
        }
        if self.use_graphs
            && let Dev::Cuda(c) = &self.dev
        {
            let e = match self.graphs.get(&a) {
                Some(&e) => e,
                None => {
                    c.begin_capture()?;
                    for si in a..b {
                        if let GStep::Kernel(l) = &self.steps[si] {
                            self.launch(l)?;
                        }
                    }
                    c.end_capture()?
                }
            };
            c.graph_launch(e)?;
            self.graphs.insert(a, e);
            return Ok(());
        }
        for si in a..b {
            if let GStep::Kernel(l) = &self.steps[si] {
                self.launch(l)?;
            }
        }
        Ok(())
    }

    /// Runs the model and copies its outputs to the host.
    pub fn run(&mut self, feeds: &interp::Feeds) -> Result<HashMap<usize, Tensor>, String> {
        let env = self.execute(feeds)?;
        let g = &self.plan.g;
        let mut out = HashMap::new();
        for &o in &g.outputs {
            let t = match env.get(&o) {
                Some(t) => t.clone(),
                None => Tensor::f32(
                    g.shape(o).to_vec(),
                    self.dev.d2h(self.value_ptr(o), numel(g.shape(o)))?,
                ),
            };
            out.insert(o, t);
        }
        Ok(out)
    }

    /// Runs the model and waits for the device, leaving the outputs there
    /// (what the benchmark times, as for the GPU baselines).
    pub fn run_on_device(&mut self, feeds: &interp::Feeds) -> Result<(), String> {
        self.execute(feeds)?;
        self.dev.sync()
    }

    pub fn enable_profile(&mut self) {
        self.profile = Some(vec![0.0; self.steps.len()]);
    }

    /// GPU time per kind of step, largest first.
    pub fn profile_report(&self) -> Vec<(String, f64, usize)> {
        let Some(p) = &self.profile else {
            return Vec::new();
        };
        let mut by: HashMap<String, (f64, usize)> = HashMap::new();
        for ((step, t), gs) in self.plan.steps.iter().zip(p).zip(&self.steps) {
            let d = match (step, gs) {
                (Step::Host(n), _) => format!("host {}", self.plan.g.nodes[*n].op),
                (_, GStep::Nop) => continue,
                (Step::Kernel(Kernel::Matmul(_)), GStep::Kernel(l)) if l.fused => {
                    format!("attention (fused) {}", l.name)
                }
                (Step::Kernel(Kernel::Matmul(mk)), GStep::Kernel(l)) => {
                    format!("matmul {} {}", matmul_signature(mk), l.name)
                }
                (Step::Kernel(Kernel::Loop(lk)), GStep::Kernel(l)) => format!(
                    "rows {} x {} ({} stages) {}",
                    lk.outer.iter().map(|o| o.1).product::<usize>(),
                    lk.j.1,
                    lk.stages.len(),
                    l.name
                ),
                _ => unreachable!(),
            };
            let e = by.entry(d).or_default();
            e.0 += t;
            e.1 += 1;
        }
        let mut v: Vec<(String, f64, usize)> =
            by.into_iter().map(|(k, (t, n))| (k, t, n)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    pub fn graph(&self) -> &crate::graph::Graph {
        &self.plan.g
    }
}
