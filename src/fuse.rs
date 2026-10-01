//! Fusion: decides which values are materialized and groups the rest into
//! kernels, in the manner of XLA's fusion passes and TVM's compute_inline.
//!
//! Every node output is either a *root* (written to memory) or *inline*
//! (recomputed inside whichever kernel reads it). Roots are matrix
//! multiplies, reductions, host operations, graph outputs, and elementwise
//! values that are read more than once or must exist in memory (matmul
//! operands, host inputs). Data-movement operations (reshape, transpose,
//! slice, concat, tile, expand) are never materialized: they become index
//! arithmetic in their readers.
//!
//! Kernels:
//! * a matmul with its *epilogue*: the elementwise root that alone consumes
//!   the product (bias, GELU, residual add, scaling, masks) is computed in
//!   registers before the single store;
//! * *row kernels*: consecutive elementwise and last-axis-reduction roots
//!   over the same rows become one kernel, one row per task, with row-local
//!   scalars and buffers for intermediates nothing else reads (so softmax
//!   and layer norm are one pass each);
//! * host steps for integer and boolean work (masks, token lookups).

use crate::graph::Graph;
use crate::ir::{Bin, Buf, Cond, E, Lin, Ranges, Un, Var, delinearize, linearize};
use crate::passes::slice_params;
use crate::tensor::{DType, Tensor};
use std::collections::{HashMap, HashSet};

/// The accumulator of a matmul kernel, inside its epilogue.
pub const ACC: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Host,
    MatMul,
    Reduce(Red),
    Elem,
    Move,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Red {
    Sum,
    Max,
    Mean,
}

#[derive(Clone, Debug)]
pub enum Stage {
    /// A per-row reduction over j into scalar `id`.
    Reduce { id: u32, op: Red, body: E },
    /// A per-row scalar.
    Scalar { id: u32, body: E },
    /// A row-local buffer over j.
    RowBuf { id: u32, body: E },
    /// A store to a graph value: over j, or once per row.
    Store {
        out: usize,
        idx: Lin,
        body: E,
        per_row: bool,
    },
}

#[derive(Clone, Debug)]
pub struct LoopK {
    /// Parallel dimensions (flattened into tasks).
    pub outer: Vec<(Var, usize)>,
    /// The inner, vectorized dimension.
    pub j: (Var, usize),
    pub stages: Vec<Stage>,
    pub ranges: Ranges,
}

#[derive(Clone, Debug)]
pub enum BOp {
    /// A constant operand, packed into panels at compile time; `lin`
    /// indexes the constant at (vk, vj).
    Packed { value: usize, lin: Lin },
    /// An operand read through `E` at (batch, vk, vj), packed per task.
    Expr(E),
}

#[derive(Clone, Debug)]
pub struct MatmulK {
    pub out: usize,
    pub batch: Vec<(Var, usize)>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub vi: Var,
    pub vj: Var,
    pub vk: Var,
    /// A at (batch, vi, vk).
    pub a: E,
    pub b: BOp,
    /// Temporaries computed in order before `epi` (E::Scalar(id) reads one).
    pub lets: Vec<(u32, E)>,
    /// The value stored at (batch, vi, vj); E::Scalar(ACC) is the product.
    pub epi: E,
    pub out_idx: Lin,
    pub ranges: Ranges,
    /// The MatMul node's own output (equals `out` without an epilogue).
    pub product: usize,
}

#[derive(Clone, Debug)]
pub enum Kernel {
    Loop(LoopK),
    Matmul(Box<MatmulK>),
}

#[derive(Clone, Debug)]
pub enum Step {
    Host(usize),
    Kernel(Kernel),
}

pub struct Plan {
    pub g: Graph,
    pub steps: Vec<Step>,
    /// Values with a buffer in the activation arena.
    pub materialized: Vec<bool>,
    pub stats: Vec<(String, usize)>,
}

pub fn classify(g: &Graph, n: &crate::graph::Node) -> Class {
    let f32_out = n
        .outputs
        .iter()
        .all(|&o| g.values[o].dtype == Some(DType::F32));
    let f32_in = n
        .inputs
        .iter()
        .flatten()
        .all(|&i| g.values[i].dtype == Some(DType::F32) || g.konst(i).is_some());
    if !f32_out {
        return Class::Host;
    }
    match n.op.as_str() {
        "MatMul" if f32_in => Class::MatMul,
        "ReduceSum" | "ReduceMax" | "ReduceMean" if f32_in => {
            let r = g.shape(n.input(0)).len();
            let axes = n.attr_ints("axes").unwrap_or_default();
            let keep = g.shape(n.outputs[0]).len() == r;
            if keep && axes == [r as i64 - 1] {
                Class::Reduce(match n.op.as_str() {
                    "ReduceSum" => Red::Sum,
                    "ReduceMax" => Red::Max,
                    _ => Red::Mean,
                })
            } else {
                Class::Host
            }
        }
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Neg" | "Sqrt" | "Erf" | "Exp" | "Log" | "Abs"
        | "Tanh" | "Relu" | "Sigmoid" | "Reciprocal"
            if f32_in =>
        {
            Class::Elem
        }
        "Identity" | "Reshape" | "Transpose" | "Slice" | "Concat" | "Expand" | "Tile" => {
            Class::Move
        }
        "Gather" => match g.konst(n.input(1)) {
            Some(t) if t.len() == 1 && t.shape.is_empty() => Class::Move,
            _ => Class::Host,
        },
        _ => Class::Host,
    }
}

/// Booleans and integers carried as f32 (booleans as 0 and 1, integers as
/// their nearest float, which is what ONNX's Cast to float computes): the
/// operations that stay exact in that representation, fused like f32 ones.
/// Integer arithmetic is not among them (it would round above 2^24).
fn lifted_class(g: &Graph, n: &crate::graph::Node) -> Option<Class> {
    match n.op.as_str() {
        // To bool: x != 0 (exact for any integer); to float: the carried
        // value itself.
        "Cast" => matches!(n.attr_int("to", 1), 1 | 9).then_some(Class::Elem),
        "And" | "Or" | "Not" | "Where" => Some(Class::Elem),
        "Identity" | "Reshape" | "Transpose" | "Slice" | "Concat" | "Expand" | "Tile" => {
            Some(Class::Move)
        }
        "Gather" => {
            let u = n.input(0);
            let r = g.shape(u).len() as i64;
            let a = n.attr_int("axis", 0);
            let a = if a < 0 { a + r } else { a } as usize;
            let dim = g.shape(u)[a] as i64;
            g.konst(n.input(1))
                .and_then(|t| affine_index(t, dim))
                .map(|_| Class::Move)
        }
        _ => None,
    }
}

/// A constant index tensor that is an affine function of its position,
/// `c + Σ s_d · i_d` (after wrapping negative indices by `dim`), every
/// value in range: (c, s).
pub fn affine_index(t: &Tensor, dim: i64) -> Option<(i64, Vec<i64>)> {
    let v: Vec<i64> = t
        .to_i64()
        .iter()
        .map(|&x| if x < 0 { x + dim } else { x })
        .collect();
    if v.is_empty() || v.iter().any(|&x| !(0..dim).contains(&x)) {
        return None;
    }
    let shape = &t.shape;
    let st = crate::tensor::strides(shape);
    let c = v[0];
    let s: Vec<i64> = (0..shape.len())
        .map(|d| if shape[d] > 1 { v[st[d]] - c } else { 0 })
        .collect();
    let mut idx = vec![0usize; shape.len()];
    for &x in &v {
        let want = c + idx.iter().zip(&s).map(|(&i, &k)| i as i64 * k).sum::<i64>();
        if x != want {
            return None;
        }
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Some((c, s))
}

struct Ctx<'a> {
    g: &'a Graph,
    prod: &'a [Option<usize>],
    root: &'a [bool],
    ranges: Ranges,
    /// Values read as scalars or row buffers inside the kernel being built.
    bind: HashMap<usize, Bind>,
    /// The root being defined (expanded rather than loaded).
    expand: Option<usize>,
    /// The matmul product bound to the accumulator, at this index only.
    acc: Vec<(usize, Vec<Lin>, u32)>,
}

#[derive(Clone, Copy)]
enum Bind {
    Scalar(u32),
    Row(u32),
}

fn bcast(idx: &[Lin], out: &[usize], inp: &[usize]) -> Vec<Lin> {
    let off = out.len() - inp.len();
    (0..inp.len())
        .map(|d| {
            if inp[d] == 1 {
                Lin::konst(0)
            } else {
                idx[d + off].clone()
            }
        })
        .collect()
}

impl Ctx<'_> {
    fn build(&self, v: usize, idx: &[Lin]) -> Result<E, String> {
        if let Some((_, at, id)) = self.acc.iter().rev().find(|a| a.0 == v) {
            let shape = self.g.shape(v);
            let same =
                at.len() == idx.len() && (0..idx.len()).all(|d| shape[d] == 1 || at[d] == idx[d]);
            return if same {
                Ok(E::Scalar(*id))
            } else {
                Err("product read at another index".into())
            };
        }
        if let Some(b) = self.bind.get(&v) {
            return Ok(match *b {
                Bind::Scalar(id) => E::Scalar(id),
                Bind::Row(id) => {
                    E::Load(Buf::Row(id), idx.last().cloned().unwrap_or(Lin::konst(0)))
                }
            });
        }
        let shape = self.g.shape(v);
        if let Some(t) = self.g.konst(v) {
            if t.len() == 1 {
                return Ok(E::Const(t.to_f32()[0]));
            }
            return Ok(E::Load(Buf::Value(v), linearize(idx, shape)));
        }
        let Some(p) = self.prod[v] else {
            return Ok(E::Load(Buf::Value(v), linearize(idx, shape)));
        };
        if self.root[v] && self.expand != Some(v) {
            return Ok(E::Load(Buf::Value(v), linearize(idx, shape)));
        }
        self.expand_node(p, idx)
    }

    fn expand_node(&self, p: usize, idx: &[Lin]) -> Result<E, String> {
        let n = &self.g.nodes[p];
        let out = self.g.shape(n.outputs[0]);
        let input = |i: usize| -> Result<E, String> {
            let u = n.input(i);
            self.build(u, &bcast(idx, out, self.g.shape(u)))
        };
        let un = |op: Un| -> Result<E, String> { Ok(E::Un(op, Box::new(input(0)?))) };
        let bin = |op: Bin| -> Result<E, String> {
            Ok(E::Bin(op, Box::new(input(0)?), Box::new(input(1)?)))
        };
        match n.op.as_str() {
            "Add" => bin(Bin::Add),
            "Sub" => bin(Bin::Sub),
            "Mul" => bin(Bin::Mul),
            "Div" => bin(Bin::Div),
            "Pow" => bin(Bin::Pow),
            "Neg" => un(Un::Neg),
            "Sqrt" => un(Un::Sqrt),
            "Erf" => un(Un::Erf),
            "Exp" => un(Un::Exp),
            "Log" => un(Un::Log),
            "Abs" => un(Un::Abs),
            "Tanh" => un(Un::Tanh),
            "Relu" => un(Un::Relu),
            "Sigmoid" => un(Un::Sigmoid),
            "Reciprocal" => un(Un::Recip),
            // Carried booleans and integers (see `lifted_class`).
            "Cast" => {
                if n.attr_int("to", 1) == 9 {
                    un(Un::Nz)
                } else {
                    input(0)
                }
            }
            "And" => bin(Bin::Mul),
            "Or" => bin(Bin::Max),
            "Not" => Ok(E::Bin(
                Bin::Sub,
                Box::new(E::Const(1.0)),
                Box::new(input(0)?),
            )),
            "Where" => Ok(E::If(
                Box::new(input(0)?),
                Box::new(input(1)?),
                Box::new(input(2)?),
            )),
            "Identity" => input(0),
            "Expand" => input(0),
            "Reshape" => {
                let u = n.input(0);
                let lin = linearize(idx, out);
                self.build(u, &delinearize(&lin, self.g.shape(u), &self.ranges))
            }
            "Transpose" => {
                let u = n.input(0);
                let r = idx.len();
                let perm: Vec<usize> = n
                    .attr_ints("perm")
                    .map(|p| p.iter().map(|&q| q as usize).collect())
                    .unwrap_or_else(|| (0..r).rev().collect());
                let mut ii = vec![Lin::konst(0); r];
                for d in 0..r {
                    ii[perm[d]] = idx[d].clone();
                }
                self.build(u, &ii)
            }
            "Slice" => {
                let u = n.input(0);
                let kv = |i: usize| self.g.konst(n.input(i)).map(Tensor::to_i64);
                let p = slice_params(
                    self.g.shape(u),
                    &kv(1).ok_or("Slice starts")?,
                    &kv(2).ok_or("Slice ends")?,
                    if n.has_input(3) { kv(3) } else { None },
                    if n.has_input(4) { kv(4) } else { None },
                );
                let ii: Vec<Lin> = idx
                    .iter()
                    .enumerate()
                    .map(|(d, l)| l.scale(p.step[d]).add_const(p.begin[d]))
                    .collect();
                self.build(u, &ii)
            }
            "Tile" => {
                let u = n.input(0);
                let is = self.g.shape(u);
                let ii: Vec<Lin> = idx
                    .iter()
                    .enumerate()
                    .map(|(d, l)| {
                        if is[d] == out[d] {
                            l.clone()
                        } else {
                            l.rem(is[d] as i64, &self.ranges)
                        }
                    })
                    .collect();
                self.build(u, &ii)
            }
            "Gather" => {
                let u = n.input(0);
                let r = self.g.shape(u).len();
                let a = {
                    let a = n.attr_int("axis", 0);
                    if a < 0 {
                        (a + r as i64) as usize
                    } else {
                        a as usize
                    }
                };
                let dim = self.g.shape(u)[a] as i64;
                let t = self.g.konst(n.input(1)).ok_or("Gather indices")?;
                let (c, st) = affine_index(t, dim).ok_or("Gather indices not affine")?;
                let ir = t.shape.len();
                // Output index: idx[..a], then the index tensor's position
                // (ir dimensions), then the data's trailing dimensions.
                let mut at = Lin::konst(c);
                for d in 0..ir {
                    at = at.add(&idx[a + d].scale(st[d]));
                }
                let mut ii: Vec<Lin> = idx[..a].to_vec();
                ii.push(at);
                ii.extend(idx[a + ir..].iter().cloned());
                self.build(u, &ii)
            }
            "Concat" => {
                let a = {
                    let a = n.attr_int("axis", 0);
                    if a < 0 {
                        (a + out.len() as i64) as usize
                    } else {
                        a as usize
                    }
                };
                let parts: Vec<usize> = n.inputs.iter().flatten().copied().collect();
                let mut start = 0i64;
                let mut pieces = Vec::new();
                for &u in &parts {
                    let len = self.g.shape(u)[a] as i64;
                    let mut ii = idx.to_vec();
                    ii[a] = idx[a].add_const(-start);
                    pieces.push((start + len, self.build(u, &ii)?));
                    start += len;
                }
                let mut e = pieces.pop().unwrap().1;
                while let Some((end, p)) = pieces.pop() {
                    e = E::Sel(
                        Cond {
                            lhs: idx[a].clone(),
                            bound: end,
                        },
                        Box::new(p),
                        Box::new(e),
                    );
                }
                Ok(e)
            }
            op => Err(format!("cannot inline {op}")),
        }
    }
}

/// The roots an expression of `v` reads (through inline nodes).
fn root_deps(
    g: &Graph,
    prod: &[Option<usize>],
    root: &[bool],
    v: usize,
    out: &mut HashSet<usize>,
    seen: &mut HashSet<usize>,
) {
    let Some(p) = prod[v] else { return };
    for &u in g.nodes[p].inputs.iter().flatten() {
        if g.konst(u).is_some() || !seen.insert(u) {
            continue;
        }
        if root[u] || prod[u].is_none() {
            out.insert(u);
        } else {
            root_deps(g, prod, root, u, out, seen);
        }
    }
}

fn vars(dims: &[usize], first: Var) -> (Vec<(Var, usize)>, Ranges) {
    let v: Vec<(Var, usize)> = dims
        .iter()
        .enumerate()
        .map(|(i, &d)| (first + i as Var, d))
        .collect();
    let r = v.iter().map(|&(x, d)| (x, d as i64)).collect();
    (v, r)
}

/// Largest row-local buffer, in floats.
const MAX_ROW: usize = 1 << 16;

/// Which optimizations the planner may use (all, by default; switched off
/// one at a time to measure what each contributes).
#[derive(Clone, Copy, Debug)]
pub struct FuseOpts {
    /// Compute elementwise chains after a matmul in its registers.
    pub epilogue: bool,
    /// Group consecutive row operations (reductions, elementwise) into one kernel.
    pub rows: bool,
    /// Recompute single-use elementwise values where they are read (off:
    /// every node writes its result to memory, one kernel per node).
    pub inline: bool,
    /// Fold the batch of shared-weight matmuls into their rows.
    pub fold: bool,
}

impl Default for FuseOpts {
    fn default() -> Self {
        FuseOpts {
            epilogue: true,
            rows: true,
            inline: true,
            fold: true,
        }
    }
}

pub fn plan(g: Graph) -> Result<Plan, String> {
    plan_with(g, FuseOpts::default())
}

pub fn plan_with(g: Graph, opts: FuseOpts) -> Result<Plan, String> {
    let nv = g.values.len();
    let prod = g.producers();
    let mut class: Vec<Class> = g.nodes.iter().map(|n| classify(&g, n)).collect();
    // Lift boolean and integer operations into kernels, then hand back to
    // the host (cascading backwards) any whose non-f32 result something on
    // the host, a matmul, a reduction or the graph's outputs would read:
    // those need the real type.
    let mut lifted = vec![false; g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        if class[i] == Class::Host
            && let Some(c) = lifted_class(&g, n)
        {
            class[i] = c;
            lifted[i] = true;
        }
    }
    {
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); nv];
        for (i, n) in g.nodes.iter().enumerate() {
            for &u in n.inputs.iter().flatten() {
                consumers[u].push(i);
            }
        }
        loop {
            let mut changed = false;
            for (i, n) in g.nodes.iter().enumerate() {
                if !lifted[i] || class[i] == Class::Host {
                    continue;
                }
                let needs_type = n.outputs.iter().any(|&o| {
                    g.values[o].dtype != Some(DType::F32)
                        && (g.outputs.contains(&o)
                            || consumers[o].iter().any(|&c| {
                                matches!(class[c], Class::Host | Class::MatMul | Class::Reduce(_))
                            }))
                });
                if needs_type {
                    class[i] = Class::Host;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
    let uses = g.use_counts();
    let is_output: Vec<bool> = {
        let mut o = vec![false; nv];
        for &v in &g.outputs {
            o[v] = true;
        }
        o
    };
    // Roots.
    let mut root = vec![false; nv];
    for (i, n) in g.nodes.iter().enumerate() {
        let o = n.outputs[0];
        root[o] = match class[i] {
            Class::Host | Class::MatMul | Class::Reduce(_) => true,
            Class::Elem => uses[o] > 1 || is_output[o] || !opts.inline,
            Class::Move => is_output[o],
        };
    }
    // Matmul operands and host inputs must exist in memory: walk back
    // through movement for matmuls (they read through index maps).
    for (i, n) in g.nodes.iter().enumerate() {
        match class[i] {
            Class::MatMul => {
                for &u in n.inputs.iter().flatten() {
                    let mut v = u;
                    while let Some(p) = prod[v] {
                        if class[p] == Class::Move && g.konst(v).is_none() {
                            v = g.nodes[p].input(0);
                        } else {
                            break;
                        }
                    }
                    if prod[v].is_some() && g.konst(v).is_none() {
                        root[v] = true;
                    }
                }
            }
            Class::Host => {
                for &u in n.inputs.iter().flatten() {
                    if prod[u].is_some() && g.konst(u).is_none() {
                        root[u] = true;
                    }
                }
            }
            _ => {}
        }
    }
    let deps: HashMap<usize, HashSet<usize>> = (0..nv)
        .filter(|&v| root[v])
        .map(|v| {
            let mut d = HashSet::new();
            root_deps(&g, &prod, &root, v, &mut d, &mut HashSet::new());
            (v, d)
        })
        .collect();
    let readers = |x: usize| -> Vec<usize> {
        deps.iter()
            .filter(|(_, d)| d.contains(&x))
            .map(|(&y, _)| y)
            .collect()
    };

    let order: Vec<usize> = g
        .nodes
        .iter()
        .map(|n| n.outputs[0])
        .filter(|&v| root[v])
        .collect();
    let mut emitted: HashSet<usize> = g.inputs.iter().copied().collect();
    let mut fused: HashSet<usize> = HashSet::new();
    let mut steps = Vec::new();
    let mut group: Vec<usize> = Vec::new();
    let mut st_kernels = 0;
    let mut st_epi = 0;

    let ctx = |ranges: Ranges| Ctx {
        g: &g,
        prod: &prod,
        root: &root,
        ranges,
        bind: HashMap::new(),
        expand: None,
        acc: Vec::new(),
    };

    // Row space of a root: (outer dims, inner width) of its reduction input
    // or of its own shape; per-row values have width 1.
    let row_space = |v: usize| -> (Vec<usize>, usize) {
        let p = prod[v].unwrap();
        let s = match class[p] {
            Class::Reduce(_) => g.shape(g.nodes[p].input(0)).to_vec(),
            _ => g.shape(v).to_vec(),
        };
        if s.is_empty() {
            return (vec![], 1);
        }
        (s[..s.len() - 1].to_vec(), s[s.len() - 1])
    };

    let close = |group: &mut Vec<usize>,
                 steps: &mut Vec<Step>,
                 emitted: &mut HashSet<usize>|
     -> Result<(), String> {
        if group.is_empty() {
            return Ok(());
        }
        let members: HashSet<usize> = group.iter().copied().collect();
        let mut outer = vec![];
        let mut width = 1;
        for &x in group.iter() {
            let (o, w) = row_space(x);
            outer = o;
            width = width.max(w);
        }
        let (mut ov, mut ranges) = vars(&outer, 0);
        let jv = outer.len() as Var;
        ranges.insert(jv, width as i64);
        let mut c = ctx(ranges.clone());
        let mut stages = Vec::new();
        let idx_for = |shape: &[usize], c: &Ctx| -> Vec<Lin> {
            // Values here have shape outer ++ [width] or outer ++ [1].
            let _ = c;
            let mut idx: Vec<Lin> = ov.iter().map(|&(v, _)| Lin::var(v)).collect();
            if shape.len() > outer.len() {
                idx.push(if shape[shape.len() - 1] == 1 {
                    Lin::konst(0)
                } else {
                    Lin::var(jv)
                });
            }
            // Ranks below outer.len()+1 (scalars): broadcast from the right.
            let off = idx.len().saturating_sub(shape.len());
            idx[off..]
                .iter()
                .zip(shape)
                .map(|(l, &d)| if d == 1 { Lin::konst(0) } else { l.clone() })
                .collect()
        };
        for (id, &x) in group.iter().enumerate() {
            let id = id as u32;
            let p = prod[x].unwrap();
            let shape = g.shape(x).to_vec();
            let per_row = shape.last().is_none_or(|&d| d == 1) || width == 1;
            let outside = is_output[x] || readers(x).iter().any(|y| !members.contains(y));
            c.expand = Some(x);
            match class[p] {
                Class::Reduce(op) => {
                    let u = g.nodes[p].input(0);
                    let us = g.shape(u).to_vec();
                    let body = c.build(u, &idx_for(&us, &c))?;
                    stages.push(Stage::Reduce { id, op, body });
                    c.bind.insert(x, Bind::Scalar(id));
                    if outside {
                        stages.push(Stage::Store {
                            out: x,
                            idx: linearize(&idx_for(&shape, &c), &shape),
                            body: E::Scalar(id),
                            per_row: true,
                        });
                    }
                }
                _ => {
                    let idx = idx_for(&shape, &c);
                    let body = c.expand_node(p, &idx)?;
                    if outside {
                        stages.push(Stage::Store {
                            out: x,
                            idx: linearize(&idx, &shape),
                            body,
                            per_row,
                        });
                    } else if per_row {
                        stages.push(Stage::Scalar { id, body });
                        c.bind.insert(x, Bind::Scalar(id));
                    } else if width <= MAX_ROW {
                        stages.push(Stage::RowBuf { id, body });
                        c.bind.insert(x, Bind::Row(id));
                    } else {
                        stages.push(Stage::Store {
                            out: x,
                            idx: linearize(&idx, &shape),
                            body,
                            per_row,
                        });
                    }
                }
            }
            c.expand = None;
        }
        for &x in group.iter() {
            emitted.insert(x);
        }
        if ov.is_empty() {
            // A single row: give it a trivial parallel dimension.
            let v = jv + 1;
            ov.push((v, 1));
            ranges.insert(v, 1);
        }
        steps.push(Step::Kernel(Kernel::Loop(LoopK {
            outer: ov,
            j: (jv, width),
            stages,
            ranges: c.ranges.clone(),
        })));
        group.clear();
        Ok(())
    };

    for &v in &order {
        if fused.contains(&v) {
            continue;
        }
        let p = prod[v].unwrap();
        match class[p] {
            Class::Host => {
                close(&mut group, &mut steps, &mut emitted)?;
                steps.push(Step::Host(p));
                emitted.insert(v);
            }
            Class::MatMul => {
                close(&mut group, &mut steps, &mut emitted)?;
                let n = &g.nodes[p];
                let (a, b) = (n.input(0), n.input(1));
                let (sa, sb, so) = (
                    g.shape(a).to_vec(),
                    g.shape(b).to_vec(),
                    g.shape(v).to_vec(),
                );
                let r = so.len();
                let (m, nn, k) = (so[r - 2], so[r - 1], sa[sa.len() - 1]);
                // Batch folding: when B is shared across the batch (a
                // weight), the batch dimensions fold into the rows, giving
                // one tall matmul instead of one per batch element, provided
                // A and the output stay affine in the folded row index.
                let b_batched = sb.len() > 2 && sb[..sb.len() - 2].iter().any(|&d| d > 1);
                let nbatch: usize = so[..r - 2].iter().product();
                let try_fold = opts.fold && !b_batched && nbatch > 1;
                let mut frame = None;
                for fold in [true, false] {
                    if fold && !try_fold {
                        continue;
                    }
                    let (batch, mut ranges) = if fold {
                        (Vec::new(), Ranges::new())
                    } else {
                        vars(&so[..r - 2], 0)
                    };
                    let (vi, vj, vk) = (
                        batch.len() as Var,
                        batch.len() as Var + 1,
                        batch.len() as Var + 2,
                    );
                    let m_eff = if fold { nbatch * m } else { m };
                    ranges.insert(vi, m_eff as i64);
                    ranges.insert(vj, nn as i64);
                    ranges.insert(vk, k as i64);
                    // The output index (batch..., i, j) in terms of the loop variables.
                    let mut full: Vec<Lin> = if fold {
                        delinearize(&Lin::var(vi), &so[..r - 1], &ranges)
                    } else {
                        (0..r - 1).map(|d| Lin::var(d as Var)).collect()
                    };
                    full.push(Lin::var(vj));
                    let bidx: Vec<Lin> = full[..r - 2].to_vec();
                    let op_idx = |s: &[usize], x: Lin, y: Lin| -> Vec<Lin> {
                        let nb = s.len() - 2;
                        let mut idx = if nb <= bidx.len() {
                            bcast(&bidx, &so[..r - 2], &s[..nb])
                        } else {
                            vec![Lin::konst(0); nb]
                        };
                        idx.push(x);
                        idx.push(y);
                        idx
                    };
                    let c = ctx(ranges.clone());
                    let ae = c.build(a, &op_idx(&sa, full[r - 2].clone(), Lin::var(vk)))?;
                    let be = c.build(b, &op_idx(&sb, Lin::var(vk), Lin::var(vj)))?;
                    let affine = |e: &E| match e {
                        E::Load(_, l) => !l.coeff(vi).1,
                        _ => false,
                    };
                    let out_ok = !linearize(&full, &so).coeff(vi).1;
                    if fold && !(affine(&ae) && out_ok) {
                        continue;
                    }
                    frame = Some((batch, vi, vj, vk, ranges, full, m_eff, ae, be, c));
                    break;
                }
                let (batch, vi, vj, vk, ranges, full, m_eff, ae, be, mut c) = frame.unwrap();
                let bvars: Vec<Var> = batch.iter().map(|b| b.0).collect();
                let bop = match &be {
                    E::Load(Buf::Value(cv), lin)
                        if g.konst(*cv).is_some() && bvars.iter().all(|x| !lin.mentions(*x)) =>
                    {
                        BOp::Packed {
                            value: *cv,
                            lin: lin.clone(),
                        }
                    }
                    _ => BOp::Expr(be),
                };
                let m = m_eff;
                // Epilogue fusion, repeated along a chain: while the current
                // value has a single reader, an elementwise root of the same
                // shape whose other inputs already exist and which reads it
                // only at the output index, that root is computed in
                // registers too. Intermediate links become temporaries.
                let mut out = v;
                let mut lets: Vec<(u32, E)> = Vec::new();
                let mut epi = E::Scalar(ACC);
                let mut chain: Vec<usize> = vec![v];
                c.acc = vec![(v, full.clone(), ACC)];
                loop {
                    let cur = out;
                    if !opts.epilogue {
                        break;
                    }
                    let rd = readers(cur);
                    if is_output[cur] || rd.len() != 1 {
                        break;
                    }
                    let rr = rd[0];
                    let rp = prod[rr].unwrap();
                    let ok_class = class[rp] == Class::Elem && g.shape(rr) == so.as_slice();
                    let deps_ok = deps[&rr]
                        .iter()
                        .all(|d| chain.contains(d) || emitted.contains(d));
                    let direct_other = g.nodes.iter().enumerate().any(|(ni, nd)| {
                        nd.inputs.iter().flatten().any(|&u| u == cur)
                            && !(class[ni] == Class::Elem || class[ni] == Class::Move)
                    });
                    if !(ok_class && deps_ok && !direct_other) {
                        break;
                    }
                    c.expand = Some(rr);
                    let res = c.expand_node(rp, &full);
                    c.expand = None;
                    let Ok(e) = res else { break };
                    if cur != v {
                        // The previous link becomes a temporary.
                        let id = lets.len() as u32;
                        lets.push((id, std::mem::replace(&mut epi, E::Scalar(ACC))));
                        c.acc.push((cur, full.clone(), id));
                        // Re-expand now that `cur` is bound to the temporary.
                        c.expand = Some(rr);
                        epi = c.expand_node(rp, &full)?;
                        c.expand = None;
                    } else {
                        epi = e;
                    }
                    fused.insert(rr);
                    chain.push(rr);
                    out = rr;
                    st_epi += 1;
                }
                c.acc.clear();
                let out_idx = linearize(&full, &so);
                steps.push(Step::Kernel(Kernel::Matmul(Box::new(MatmulK {
                    out,
                    batch,
                    m,
                    n: nn,
                    k,
                    vi,
                    vj,
                    vk,
                    a: ae,
                    b: bop,
                    lets,
                    epi,
                    out_idx,
                    ranges,
                    product: v,
                }))));
                emitted.insert(v);
                emitted.insert(out);
            }
            Class::Elem | Class::Reduce(_) => {
                let (o, w) = row_space(v);
                let compatible = opts.rows
                    && group.first().is_some_and(|&f| {
                        let (go, gw) = group
                            .iter()
                            .map(|&x| row_space(x))
                            .fold((row_space(f).0, 1), |acc, (oo, ww)| (oo, acc.1.max(ww)));
                        go == o && (w == 1 || gw == 1 || w == gw)
                    });
                let avail = deps[&v]
                    .iter()
                    .all(|d| emitted.contains(d) || group.contains(d));
                if !(compatible && avail) {
                    close(&mut group, &mut steps, &mut emitted)?;
                }
                group.push(v);
            }
            Class::Move => {
                // A graph output that is pure data movement: copy it.
                close(&mut group, &mut steps, &mut emitted)?;
                group.push(v);
                close(&mut group, &mut steps, &mut emitted)?;
            }
        }
    }
    close(&mut group, &mut steps, &mut emitted)?;
    for s in &steps {
        if matches!(s, Step::Kernel(_)) {
            st_kernels += 1;
        }
    }
    // A value has an arena buffer if it is an f32 graph input or some
    // step writes it; everything else lives in registers or row-local
    // scratch, or is recomputed where it is read.
    // Inputs and host results get a buffer when f32 or read by a kernel
    // (booleans and integers then as carried f32).
    let mut kernel_reads: HashSet<usize> = HashSet::new();
    for s in &steps {
        if let Step::Kernel(k) = s {
            let mut bufs = Vec::new();
            match k {
                Kernel::Matmul(mk) => {
                    mk.a.loads(&mut bufs);
                    mk.epi.loads(&mut bufs);
                    for l in &mk.lets {
                        l.1.loads(&mut bufs);
                    }
                    if let BOp::Expr(e) = &mk.b {
                        e.loads(&mut bufs);
                    }
                }
                Kernel::Loop(l) => {
                    for st in &l.stages {
                        match st {
                            Stage::Reduce { body, .. }
                            | Stage::Scalar { body, .. }
                            | Stage::RowBuf { body, .. }
                            | Stage::Store { body, .. } => body.loads(&mut bufs),
                        }
                    }
                }
            }
            for b in bufs {
                if let Buf::Value(v) = b {
                    kernel_reads.insert(v);
                }
            }
        }
    }
    let mut materialized = vec![false; nv];
    for &v in &g.inputs {
        materialized[v] = g.values[v].dtype == Some(DType::F32) || kernel_reads.contains(&v);
    }
    for s in &steps {
        match s {
            Step::Host(p) => {
                for &o in &g.nodes[*p].outputs {
                    materialized[o] =
                        g.values[o].dtype == Some(DType::F32) || kernel_reads.contains(&o);
                }
            }
            Step::Kernel(Kernel::Matmul(mk)) => materialized[mk.out] = true,
            Step::Kernel(Kernel::Loop(l)) => {
                for st in &l.stages {
                    if let Stage::Store { out, .. } = st {
                        materialized[*out] = true;
                    }
                }
            }
        }
    }
    let n_host = steps.iter().filter(|s| matches!(s, Step::Host(_))).count();
    let n_mm = steps
        .iter()
        .filter(|s| matches!(s, Step::Kernel(Kernel::Matmul(_))))
        .count();
    let stats = vec![
        ("graph_nodes".to_string(), g.nodes.len()),
        ("kernels".to_string(), st_kernels),
        ("matmul_kernels".to_string(), n_mm),
        ("fused_epilogues".to_string(), st_epi),
        ("row_kernels".to_string(), st_kernels - n_mm),
        ("host_steps".to_string(), n_host),
    ];
    Ok(Plan {
        g,
        steps,
        materialized,
        stats,
    })
}

impl Plan {
    /// One line per step: kind, written values, values read.
    pub fn dump(&self) -> String {
        let name = |v: usize| self.g.values[v].name.clone();
        let reads = |es: &[&E]| -> Vec<String> {
            let mut b = Vec::new();
            for e in es {
                e.loads(&mut b);
            }
            b.iter()
                .filter_map(|x| match x {
                    Buf::Value(v) if self.g.konst(*v).is_none() => Some(name(*v)),
                    _ => None,
                })
                .collect()
        };
        let mut s = String::new();
        for (i, st) in self.steps.iter().enumerate() {
            let line = match st {
                Step::Host(p) => {
                    let n = &self.g.nodes[*p];
                    format!("host {} -> {}", n.op, name(n.outputs[0]))
                }
                Step::Kernel(Kernel::Matmul(m)) => {
                    let mut es = vec![&m.a, &m.epi];
                    es.extend(m.lets.iter().map(|l| &l.1));
                    if let BOp::Expr(e) = &m.b {
                        es.push(e);
                    }
                    format!(
                        "matmul {}x{}x{} batch {:?} -> {} (product {}) reads {:?}{}",
                        m.m,
                        m.n,
                        m.k,
                        m.batch.iter().map(|b| b.1).collect::<Vec<_>>(),
                        name(m.out),
                        name(m.product),
                        reads(&es),
                        if matches!(m.b, BOp::Packed { .. }) {
                            " [B packed]"
                        } else {
                            ""
                        }
                    )
                }
                Step::Kernel(Kernel::Loop(l)) => {
                    let mut outs = Vec::new();
                    let mut es = Vec::new();
                    let mut kinds = Vec::new();
                    for st in &l.stages {
                        match st {
                            Stage::Reduce { body, op, .. } => {
                                kinds.push(format!("{op:?}"));
                                es.push(body);
                            }
                            Stage::Scalar { body, .. } => {
                                kinds.push("s".into());
                                es.push(body);
                            }
                            Stage::RowBuf { body, .. } => {
                                kinds.push("row".into());
                                es.push(body);
                            }
                            Stage::Store { out, body, .. } => {
                                kinds.push("store".into());
                                outs.push(name(*out));
                                es.push(body);
                            }
                        }
                    }
                    format!(
                        "loop rows {:?} x {} [{}] -> {:?} reads {:?}",
                        l.outer.iter().map(|o| o.1).collect::<Vec<_>>(),
                        l.j.1,
                        kinds.join(" "),
                        outs,
                        reads(&es)
                    )
                }
            };
            s.push_str(&format!("{i:3}: {line}\n"));
        }
        s
    }
}
