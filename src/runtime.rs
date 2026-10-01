//! The executable: generated kernels in one compiled library, a memory
//! plan that packs every intermediate into one arena by lifetime, weights
//! packed into matmul panels, and host steps for integer work.

use crate::codegen::{self, Arg, KernelSrc, Params};
use crate::fuse::{BOp, Kernel, Plan, Step};
use crate::interp;
use crate::jit::{self, KernelFn, Library};
use crate::pool::Pool;
use crate::tensor::{DType, Tensor, numel};
use std::collections::HashMap;

enum Exec {
    Kernel {
        f: KernelFn,
        args: Vec<*mut f32>,
        tasks: usize,
        name: String,
    },
    Host(usize),
}

pub struct Options {
    pub threads: usize,
    /// Matmul schedules chosen by the tuner, by kernel signature.
    pub params: HashMap<String, Params>,
    pub log: bool,
}

pub struct Stats {
    pub kernels: usize,
    pub unique_kernels: usize,
    pub host_steps: usize,
    pub arena_floats: usize,
    pub naive_floats: usize,
    pub compile_seconds: f64,
    pub cached: bool,
    pub c_lines: usize,
}

pub struct Executable {
    plan: Plan,
    steps: Vec<Exec>,
    arena: Vec<f32>,
    offsets: HashMap<usize, usize>,
    _packed: Vec<Vec<f32>>,
    _lib: Library,
    pool: Pool,
    pub stats: Stats,
    /// Accumulated seconds per step, when profiling.
    pub profile: Option<Vec<f64>>,
}

// Raw pointers into buffers the executable owns.
unsafe impl Send for Executable {}

/// A stable name for a matmul's schedule: its shape and operand kind.
pub fn matmul_signature(mk: &crate::fuse::MatmulK) -> String {
    let batch: usize = mk.batch.iter().map(|b| b.1).product();
    let kind = if matches!(mk.b, BOp::Packed { .. }) {
        "packed"
    } else {
        "expr"
    };
    format!("b{batch}_m{}_n{}_k{}_{kind}", mk.m, mk.n, mk.k)
}

/// Offsets for buffers with lifetimes [start, end] (in steps): greedy by
/// size, each placed at the lowest offset clear of every overlapping one.
pub fn plan_memory(bufs: &[(usize, usize, usize, usize)]) -> (HashMap<usize, usize>, usize) {
    // (value, size, start, end)
    let mut order: Vec<usize> = (0..bufs.len()).collect();
    order.sort_by(|&a, &b| bufs[b].1.cmp(&bufs[a].1).then(bufs[a].2.cmp(&bufs[b].2)));
    let mut placed: Vec<(usize, usize, usize, usize)> = Vec::new(); // offset, size, start, end
    let mut out = HashMap::new();
    let mut total = 0;
    for i in order {
        let (v, size, start, end) = bufs[i];
        let size = size.div_ceil(16) * 16;
        let mut busy: Vec<(usize, usize)> = placed
            .iter()
            .filter(|p| p.2 <= end && start <= p.3)
            .map(|p| (p.0, p.0 + p.1))
            .collect();
        busy.sort_unstable();
        let mut off = 0;
        for (a, b) in busy {
            if off + size <= a {
                break;
            }
            off = off.max(b);
        }
        placed.push((off, size, start, end));
        out.insert(v, off);
        total = total.max(off + size);
    }
    (out, total)
}

fn pack(
    t: &Tensor,
    lin: &crate::ir::Lin,
    k: usize,
    n: usize,
    nr: usize,
    vk: u32,
    vj: u32,
) -> Vec<f32> {
    let src = t.as_f32();
    let np = n.div_ceil(nr);
    let mut out = vec![0f32; np * k * nr];
    let mut env = HashMap::new();
    for jp in 0..np {
        for kk in 0..k {
            for jj in 0..nr {
                let j = jp * nr + jj;
                if j < n {
                    env.insert(vk, kk as i64);
                    env.insert(vj, j as i64);
                    out[(jp * k + kk) * nr + jj] = src[lin.eval(&env) as usize];
                }
            }
        }
    }
    out
}

impl Executable {
    pub fn build(plan: Plan, opts: &Options) -> Result<Executable, String> {
        let w = jit::vector_width();
        let g = &plan.g;
        // Generate every kernel; identical bodies share one function.
        let mut srcs: Vec<(KernelSrc, String)> = Vec::new();
        for step in &plan.steps {
            if let Step::Kernel(k) = step {
                let (src, sig) = match k {
                    Kernel::Loop(l) => (codegen::loop_kernel(g, l, w), String::new()),
                    Kernel::Matmul(mk) => {
                        let sig = matmul_signature(mk);
                        let p = opts
                            .params
                            .get(&sig)
                            .copied()
                            .unwrap_or_else(|| codegen::default_params(mk, w, opts.threads));
                        (codegen::matmul_kernel(g, mk, p, w), sig)
                    }
                };
                srcs.push((src, sig));
            }
        }
        let mut names: HashMap<String, String> = HashMap::new();
        let mut c = codegen::prelude(w);
        for (s, _) in &srcs {
            if !names.contains_key(&s.body) {
                let name = format!("k{}", names.len());
                c.push_str(&format!("\nvoid {name}{}", s.body));
                names.insert(s.body.clone(), name);
            }
        }
        let lib = jit::compile(&c)?;
        // Memory: lifetimes of every materialized f32 value.
        let nsteps = plan.steps.len();
        let mut start: HashMap<usize, usize> = HashMap::new();
        let mut end: HashMap<usize, usize> = HashMap::new();
        let touch = |v: usize,
                     s: usize,
                     write: bool,
                     start: &mut HashMap<usize, usize>,
                     end: &mut HashMap<usize, usize>| {
            if write {
                start.entry(v).or_insert(s);
            }
            let e = end.entry(v).or_insert(s);
            *e = (*e).max(s);
        };
        for &v in &g.inputs {
            if plan.materialized[v] {
                touch(v, 0, true, &mut start, &mut end);
            }
        }
        let mut src_iter = srcs.iter();
        for (si, step) in plan.steps.iter().enumerate() {
            match step {
                Step::Host(p) => {
                    let n = &g.nodes[*p];
                    for &i in n.inputs.iter().flatten() {
                        if plan.materialized[i] {
                            touch(i, si, false, &mut start, &mut end);
                        }
                    }
                    for &o in &n.outputs {
                        if plan.materialized[o] {
                            touch(o, si, true, &mut start, &mut end);
                        }
                    }
                }
                Step::Kernel(k) => {
                    let (src, _) = src_iter.next().unwrap();
                    let outs: Vec<usize> = match k {
                        Kernel::Loop(l) => l
                            .stages
                            .iter()
                            .filter_map(|s| {
                                if let crate::fuse::Stage::Store { out, .. } = s {
                                    Some(*out)
                                } else {
                                    None
                                }
                            })
                            .collect(),
                        Kernel::Matmul(mk) => vec![mk.out],
                    };
                    for a in &src.args {
                        if let Arg::Value(v) = a
                            && plan.materialized[*v]
                        {
                            touch(*v, si, outs.contains(v), &mut start, &mut end);
                        }
                    }
                }
            }
        }
        for &o in &g.outputs {
            if plan.materialized[o] {
                end.insert(o, nsteps);
            }
        }
        let bufs: Vec<(usize, usize, usize, usize)> = start
            .iter()
            .map(|(&v, &s)| (v, numel(g.shape(v)), s, end[&v]))
            .collect();
        let naive: usize = bufs.iter().map(|b| b.1.div_ceil(16) * 16).sum();
        let (offsets, total) = plan_memory(&bufs);
        let mut arena = vec![0f32; total.max(16)];
        let base = arena.as_mut_ptr();
        // Packed weights, shared between kernels using the same layout.
        let mut packed: Vec<Vec<f32>> = Vec::new();
        let mut packed_ids: HashMap<(usize, crate::ir::Lin, usize), usize> = HashMap::new();
        let mut steps = Vec::new();
        let mut src_iter = srcs.iter();
        let mut host = 0;
        for step in &plan.steps {
            match step {
                Step::Host(p) => {
                    steps.push(Exec::Host(*p));
                    host += 1;
                }
                Step::Kernel(k) => {
                    let (src, _) = src_iter.next().unwrap();
                    let mut args = Vec::new();
                    for a in &src.args {
                        let ptr = match a {
                            Arg::Value(v) => {
                                if let Some(t) = g.konst(*v) {
                                    t.as_f32().as_ptr() as *mut f32
                                } else {
                                    let off = *offsets.get(v).ok_or_else(|| {
                                        format!("value {} has no buffer", g.values[*v].name)
                                    })?;
                                    // SAFETY: offsets lie inside the arena.
                                    unsafe { base.add(off) }
                                }
                            }
                            Arg::Packed {
                                value,
                                lin,
                                k: kk,
                                n,
                                nr,
                            } => {
                                let key = (*value, lin.clone(), *nr);
                                let id = *packed_ids.entry(key).or_insert_with(|| {
                                    let (vk, vj) = match k {
                                        Kernel::Matmul(mk) => (mk.vk, mk.vj),
                                        Kernel::Loop(_) => unreachable!(),
                                    };
                                    packed.push(pack(
                                        g.konst(*value).unwrap(),
                                        lin,
                                        *kk,
                                        *n,
                                        *nr,
                                        vk,
                                        vj,
                                    ));
                                    packed.len() - 1
                                });
                                packed[id].as_ptr() as *mut f32
                            }
                        };
                        args.push(ptr);
                    }
                    let name = names[&src.body].clone();
                    steps.push(Exec::Kernel {
                        f: lib.kernel(&name)?,
                        args,
                        tasks: src.tasks,
                        name,
                    });
                }
            }
        }
        let stats = Stats {
            kernels: srcs.len(),
            unique_kernels: names.len(),
            host_steps: host,
            arena_floats: total,
            naive_floats: naive,
            compile_seconds: lib.compile_seconds,
            cached: lib.cached,
            c_lines: c.lines().count(),
        };
        Ok(Executable {
            plan,
            steps,
            arena,
            offsets,
            _packed: packed,
            _lib: lib,
            pool: Pool::new(opts.threads),
            stats,
            profile: None,
        })
    }

    fn arena_slice(&self, v: usize) -> &[f32] {
        let off = self.offsets[&v];
        &self.arena[off..off + numel(self.plan.g.shape(v))]
    }

    /// Runs the model on `feeds` (by graph value id); returns the outputs.
    pub fn run(&mut self, feeds: &interp::Feeds) -> Result<HashMap<usize, Tensor>, String> {
        let g = &self.plan.g;
        let mut env: HashMap<usize, Tensor> = HashMap::new();
        for (&v, t) in feeds {
            if t.dtype() == DType::F32 && self.plan.materialized[v] {
                let off = self.offsets[&v];
                self.arena[off..off + t.len()].copy_from_slice(t.as_f32());
            } else {
                env.insert(v, t.clone());
            }
        }
        for (si, step) in self.steps.iter().enumerate() {
            let t0 = self.profile.is_some().then(std::time::Instant::now);
            match step {
                Exec::Kernel { f, args, tasks, .. } => {
                    let (f, tasks) = (*f, *tasks);
                    let args = SendPtr(args.as_ptr());
                    let threads = self.pool.threads();
                    let chunks = tasks.min(threads * 4).max(1);
                    self.pool.run(chunks, &|c| {
                        let a = (c * tasks / chunks) as i64;
                        let b = ((c + 1) * tasks / chunks) as i64;
                        let args = &args;
                        // SAFETY: each chunk writes its own tasks' outputs;
                        // pointers outlive the call.
                        unsafe { f(args.0, a, b) };
                    });
                }
                Exec::Host(p) => {
                    let n = &g.nodes[*p];
                    // Borrow constants and host values; copy only arena-backed inputs.
                    let owned: Vec<Option<Tensor>> = n
                        .inputs
                        .iter()
                        .map(|i| {
                            i.and_then(|i| {
                                (g.konst(i).is_none() && !env.contains_key(&i)).then(|| {
                                    Tensor::f32(g.shape(i).to_vec(), self.arena_slice(i).to_vec())
                                })
                            })
                        })
                        .collect();
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
                        if t.dtype() == DType::F32 && self.plan.materialized[o] {
                            let off = self.offsets[&o];
                            self.arena[off..off + t.len()].copy_from_slice(t.as_f32());
                        } else {
                            env.insert(o, t);
                        }
                    }
                }
            }
            if let (Some(t0), Some(p)) = (t0, self.profile.as_mut()) {
                p[si] += t0.elapsed().as_secs_f64();
            }
        }
        let mut out = HashMap::new();
        for &o in &g.outputs {
            let t = if let Some(t) = env.get(&o) {
                t.clone()
            } else {
                Tensor::f32(g.shape(o).to_vec(), self.arena_slice(o).to_vec())
            };
            out.insert(o, t);
        }
        Ok(out)
    }

    /// Starts accumulating time per step.
    pub fn enable_profile(&mut self) {
        self.profile = Some(vec![0.0; self.steps.len()]);
    }

    /// Time per step description, largest first.
    pub fn profile_report(&self) -> Vec<(String, f64, usize)> {
        let Some(p) = &self.profile else {
            return Vec::new();
        };
        let mut by: HashMap<String, (f64, usize)> = HashMap::new();
        for (si, (step, t)) in self.plan.steps.iter().zip(p).enumerate() {
            let d = match step {
                Step::Host(n) => format!("host {}", self.plan.g.nodes[*n].op),
                Step::Kernel(Kernel::Matmul(mk)) => format!("matmul {}", matmul_signature(mk)),
                Step::Kernel(Kernel::Loop(l)) => format!(
                    "loop rows {} x {} ({} stages) {}",
                    l.outer.iter().map(|o| o.1).product::<usize>(),
                    l.j.1,
                    l.stages.len(),
                    match &self.steps[si] {
                        Exec::Kernel { name, .. } => name.clone(),
                        Exec::Host(_) => String::new(),
                    }
                ),
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

    /// Kernel names in execution order (for profiling).
    pub fn kernel_names(&self) -> Vec<String> {
        self.steps
            .iter()
            .map(|s| match s {
                Exec::Kernel { name, .. } => name.clone(),
                Exec::Host(p) => format!("host:{}", self.plan.g.nodes[*p].op),
            })
            .collect()
    }
}

struct SendPtr(*const *mut f32);
unsafe impl Sync for SendPtr {}
