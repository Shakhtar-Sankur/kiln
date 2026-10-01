//! Graph passes: shape and type inference with constant folding,
//! canonicalization into kiln's core operator set, and dead-code
//! elimination.

use crate::graph::{Attr, Graph, Node};
use crate::interp::{self, broadcast_shape, reshape_target};
use crate::tensor::{DType, Tensor, numel};
use std::collections::HashMap;

fn axis(a: i64, rank: usize) -> usize {
    if a < 0 {
        (a + rank as i64) as usize
    } else {
        a as usize
    }
}

/// Output types and shapes of `n` from its inputs' (constants must be
/// known where the shape depends on a value).
pub fn infer(g: &Graph, n: &Node) -> Result<Vec<(DType, Vec<usize>)>, String> {
    let sh = |i: usize| -> Vec<usize> { g.shape(n.input(i)).to_vec() };
    let dt = |i: usize| -> DType { g.values[n.input(i)].dtype.unwrap_or(DType::F32) };
    let kv = |i: usize| -> Result<Vec<i64>, String> {
        g.konst(n.input(i))
            .map(Tensor::to_i64)
            .ok_or_else(|| format!("{} ({}): input {i} must be a constant", n.name, n.op))
    };
    let out = match n.op.as_str() {
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => (dt(0), broadcast_shape(&sh(0), &sh(1))?),
        "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" | "And" | "Or" => {
            (DType::Bool, broadcast_shape(&sh(0), &sh(1))?)
        }
        "Not" => (DType::Bool, sh(0)),
        "Where" => (
            dt(1),
            broadcast_shape(&broadcast_shape(&sh(0), &sh(1))?, &sh(2))?,
        ),
        "Cast" => (
            match n.attr_int("to", 1) {
                1 => DType::F32,
                9 => DType::Bool,
                _ => DType::I64,
            },
            sh(0),
        ),
        "Identity" | "Sqrt" | "Reciprocal" | "Neg" | "Exp" | "Log" | "Abs" | "Tanh" | "Relu"
        | "Sigmoid" | "Erf" | "Softmax" | "LayerNormalization" => (dt(0), sh(0)),
        "MatMul" => {
            let (a, b) = (sh(0), sh(1));
            let batch = broadcast_shape(&a[..a.len() - 2], &b[..b.len() - 2])?;
            let mut s = batch;
            s.push(a[a.len() - 2]);
            s.push(b[b.len() - 1]);
            (DType::F32, s)
        }
        "Gemm" => {
            let (a, b) = (sh(0), sh(1));
            let m = if n.attr_int("transA", 0) == 1 {
                a[1]
            } else {
                a[0]
            };
            let nn = if n.attr_int("transB", 0) == 1 {
                b[0]
            } else {
                b[1]
            };
            (DType::F32, vec![m, nn])
        }
        "Shape" => (DType::I64, vec![sh(0).len()]),
        "Reshape" => (dt(0), reshape_target(&sh(0), &kv(1)?)?),
        "Flatten" => {
            let s = sh(0);
            let a = axis(n.attr_int("axis", 1), s.len());
            let outer: usize = s[..a].iter().product();
            (dt(0), vec![outer, numel(&s) / outer.max(1)])
        }
        "Unsqueeze" | "Squeeze" => {
            let axes = if n.has_input(1) {
                kv(1)?
            } else {
                n.attr_ints("axes").unwrap_or_default()
            };
            let mut s = sh(0);
            if n.op == "Unsqueeze" {
                let r = s.len() + axes.len();
                let mut ax: Vec<usize> = axes.iter().map(|&a| axis(a, r)).collect();
                ax.sort_unstable();
                for a in ax {
                    s.insert(a, 1);
                }
            } else {
                let r = s.len();
                let ax: Vec<usize> = if axes.is_empty() {
                    (0..r).filter(|&d| s[d] == 1).collect()
                } else {
                    axes.iter().map(|&a| axis(a, r)).collect()
                };
                s = (0..r).filter(|d| !ax.contains(d)).map(|d| s[d]).collect();
            }
            (dt(0), s)
        }
        "Transpose" => {
            let s = sh(0);
            let perm: Vec<usize> = n
                .attr_ints("perm")
                .map(|p| p.iter().map(|&q| q as usize).collect())
                .unwrap_or_else(|| (0..s.len()).rev().collect());
            (dt(0), perm.iter().map(|&p| s[p]).collect())
        }
        "Concat" => {
            let mut s = sh(0);
            let a = axis(n.attr_int("axis", 0), s.len());
            s[a] = n.inputs.iter().flatten().map(|&i| g.shape(i)[a]).sum();
            (dt(0), s)
        }
        "Slice" => {
            let p = slice_params(
                &sh(0),
                &kv(1)?,
                &kv(2)?,
                if n.has_input(3) { Some(kv(3)?) } else { None },
                if n.has_input(4) { Some(kv(4)?) } else { None },
            );
            (dt(0), p.shape)
        }
        "Gather" => {
            let (s, ind) = (sh(0), sh(1));
            let a = axis(n.attr_int("axis", 0), s.len());
            let mut o: Vec<usize> = s[..a].to_vec();
            o.extend(&ind);
            o.extend(&s[a + 1..]);
            (dt(0), o)
        }
        "GatherElements" => (dt(0), sh(1)),
        "Expand" => {
            let spec: Vec<usize> = kv(1)?.iter().map(|&d| d as usize).collect();
            (dt(0), broadcast_shape(&sh(0), &spec)?)
        }
        "Tile" => {
            let r = kv(1)?;
            (
                dt(0),
                sh(0)
                    .iter()
                    .zip(&r)
                    .map(|(&d, &k)| d * k as usize)
                    .collect(),
            )
        }
        "ConstantOfShape" => (
            match n.attrs.get("value") {
                Some(Attr::Tensor(t)) => t.dtype(),
                _ => DType::F32,
            },
            kv(0)?.iter().map(|&d| d as usize).collect(),
        ),
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceL2" => {
            let s = sh(0);
            let r = s.len();
            let raw = if n.has_input(1) {
                Some(kv(1)?)
            } else {
                n.attr_ints("axes")
            };
            let axes: Vec<usize> = match raw {
                Some(v) if !v.is_empty() => v.iter().map(|&a| axis(a, r)).collect(),
                _ => (0..r).collect(),
            };
            let keep = n.attr_int("keepdims", 1) == 1;
            let o = if keep {
                s.iter()
                    .enumerate()
                    .map(|(d, &v)| if axes.contains(&d) { 1 } else { v })
                    .collect()
            } else {
                (0..r).filter(|d| !axes.contains(d)).map(|d| s[d]).collect()
            };
            (dt(0), o)
        }
        op => return Err(format!("{}: no shape rule for {op}", n.name)),
    };
    Ok(vec![out])
}

/// Normalized Slice parameters: begin and step per axis, output shape.
pub struct SliceParams {
    pub begin: Vec<i64>,
    pub step: Vec<i64>,
    pub shape: Vec<usize>,
}

pub fn slice_params(
    shape: &[usize],
    starts: &[i64],
    ends: &[i64],
    axes: Option<Vec<i64>>,
    steps: Option<Vec<i64>>,
) -> SliceParams {
    let r = shape.len();
    let axes: Vec<usize> = axes.map_or_else(
        || (0..starts.len()).collect(),
        |a| a.iter().map(|&q| axis(q, r)).collect(),
    );
    let steps = steps.unwrap_or_else(|| vec![1; starts.len()]);
    let mut p = SliceParams {
        begin: vec![0; r],
        step: vec![1; r],
        shape: shape.to_vec(),
    };
    for (k, &a) in axes.iter().enumerate() {
        let dim = shape[a] as i64;
        let st = steps[k];
        let norm = |v: i64| if v < 0 { v + dim } else { v };
        let (s, e) = if st > 0 {
            (norm(starts[k]).clamp(0, dim), norm(ends[k]).clamp(0, dim))
        } else {
            (
                norm(starts[k]).clamp(-1, dim - 1),
                norm(ends[k]).clamp(-1, dim - 1),
            )
        };
        p.begin[a] = s;
        p.step[a] = st;
        p.shape[a] = if st > 0 {
            ((e - s).max(0) + st - 1) as usize / st as usize
        } else {
            ((s - e).max(0) - st - 1) as usize / (-st) as usize
        };
    }
    p
}

/// Infers every value's type and shape in topological order, folding
/// nodes whose inputs are all constant (and Shape of a known shape) into
/// constants. Returns the number of nodes folded.
pub fn fold_and_infer(g: &mut Graph) -> Result<usize, String> {
    let mut kept = Vec::with_capacity(g.nodes.len());
    let mut folded = 0;
    let nodes = std::mem::take(&mut g.nodes);
    for n in nodes {
        let all_const = n
            .inputs
            .iter()
            .flatten()
            .all(|&i| g.values[i].konst.is_some());
        let shape_of_known = n.op == "Shape" && g.values[n.input(0)].shape.is_some();
        if all_const || shape_of_known {
            let outs = if shape_of_known && !all_const {
                let s = g.shape(n.input(0));
                vec![Tensor::i64(
                    vec![s.len()],
                    s.iter().map(|&d| d as i64).collect(),
                )]
            } else {
                let ins: Vec<Option<&Tensor>> = n
                    .inputs
                    .iter()
                    .map(|i| i.and_then(|i| g.values[i].konst.as_ref()))
                    .collect();
                interp::eval(&n, &ins).map_err(|e| format!("{} ({}): {e}", n.name, n.op))?
            };
            for (&o, t) in n.outputs.iter().zip(outs) {
                g.values[o].dtype = Some(t.dtype());
                g.values[o].shape = Some(t.shape.clone());
                g.values[o].konst = Some(t);
            }
            folded += 1;
            continue;
        }
        let outs = infer(g, &n).map_err(|e| format!("{} ({}): {e}", n.name, n.op))?;
        for (&o, (d, s)) in n.outputs.iter().zip(outs) {
            g.values[o].dtype = Some(d);
            g.values[o].shape = Some(s);
        }
        kept.push(n);
    }
    g.nodes = kept;
    Ok(folded)
}

/// Removes nodes whose outputs nothing uses.
pub fn dce(g: &mut Graph) -> usize {
    let before = g.nodes.len();
    loop {
        let uses = g.use_counts();
        let n0 = g.nodes.len();
        g.nodes.retain(|n| n.outputs.iter().any(|&o| uses[o] > 0));
        if g.nodes.len() == n0 {
            break;
        }
    }
    before - g.nodes.len()
}

struct Builder<'a> {
    g: &'a mut Graph,
    out: Vec<Node>,
    seq: usize,
}

impl Builder<'_> {
    fn value(&mut self, like: usize, dtype: DType, shape: Vec<usize>) -> usize {
        self.seq += 1;
        let name = format!("{}/k{}", self.g.values[like].name, self.seq);
        let v = self.g.add_value(name);
        self.g.values[v].dtype = Some(dtype);
        self.g.values[v].shape = Some(shape);
        v
    }

    fn konst(&mut self, t: Tensor) -> usize {
        self.seq += 1;
        let v = self.g.add_value(format!("const/k{}", self.seq));
        self.g.values[v].dtype = Some(t.dtype());
        self.g.values[v].shape = Some(t.shape.clone());
        self.g.values[v].konst = Some(t);
        v
    }

    /// Emits `op(inputs)`; the output takes `out` if given, else a new value
    /// shaped like `shape`.
    fn node(
        &mut self,
        op: &str,
        inputs: &[usize],
        attrs: Vec<(&str, Attr)>,
        out: Option<usize>,
        shape: Vec<usize>,
    ) -> usize {
        let o = out.unwrap_or_else(|| self.value(inputs[0], DType::F32, shape));
        self.seq += 1;
        self.out.push(Node {
            op: op.into(),
            name: format!("{op}/k{}", self.seq),
            inputs: inputs.iter().map(|&i| Some(i)).collect(),
            outputs: vec![o],
            attrs: attrs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        });
        o
    }
}

/// Rewrites the graph into kiln's core operators: Gemm becomes MatMul (+
/// Mul, Add); Softmax, LayerNormalization and ReduceL2 become reductions
/// over the last axis and elementwise operations; shape-only operators
/// become Reshape; Pow by 2 or 0.5 becomes Mul or Sqrt; reduction axes
/// become attributes. Expects fold_and_infer to have run.
pub fn canonicalize(g: &mut Graph) -> Result<(), String> {
    let nodes = std::mem::take(&mut g.nodes);
    let mut b = Builder {
        g,
        out: Vec::new(),
        seq: 0,
    };
    for n in nodes {
        let shape_of = |b: &Builder, v: usize| b.g.shape(v).to_vec();
        let o = n.outputs[0];
        let os = shape_of(&b, o);
        let is_f32 = b.g.values[o].dtype == Some(DType::F32);
        match n.op.as_str() {
            "Flatten" | "Squeeze" | "Unsqueeze" => {
                let spec = b.konst(Tensor::i64(
                    vec![os.len()],
                    os.iter().map(|&d| d as i64).collect(),
                ));
                b.node("Reshape", &[n.input(0), spec], vec![], Some(o), os);
            }
            "Gemm" => {
                let (mut a, mut w) = (n.input(0), n.input(1));
                if n.attr_int("transA", 0) == 1 {
                    let s = shape_of(&b, a);
                    a = b.node(
                        "Transpose",
                        &[a],
                        vec![("perm", Attr::Ints(vec![1, 0]))],
                        None,
                        vec![s[1], s[0]],
                    );
                }
                if n.attr_int("transB", 0) == 1 {
                    let s = shape_of(&b, w);
                    w = b.node(
                        "Transpose",
                        &[w],
                        vec![("perm", Attr::Ints(vec![1, 0]))],
                        None,
                        vec![s[1], s[0]],
                    );
                }
                let (alpha, beta) = (n.attr_float("alpha", 1.0), n.attr_float("beta", 1.0));
                let has_c = n.has_input(2);
                let last = !has_c && alpha == 1.0;
                let mut y = b.node(
                    "MatMul",
                    &[a, w],
                    vec![],
                    if last { Some(o) } else { None },
                    os.clone(),
                );
                if alpha != 1.0 {
                    let c = b.konst(Tensor::scalar_f32(alpha));
                    y = b.node(
                        "Mul",
                        &[y, c],
                        vec![],
                        if has_c { None } else { Some(o) },
                        os.clone(),
                    );
                }
                if has_c {
                    let mut c = n.input(2);
                    if beta != 1.0 {
                        let bt = b.konst(Tensor::scalar_f32(beta));
                        let cs = shape_of(&b, c);
                        c = b.node("Mul", &[c, bt], vec![], None, cs);
                    }
                    b.node("Add", &[y, c], vec![], Some(o), os.clone());
                }
            }
            "Softmax" if is_f32 => {
                let x = n.input(0);
                let mut rs = os.clone();
                *rs.last_mut().unwrap() = 1;
                let ax = vec![("axes", Attr::Ints(vec![os.len() as i64 - 1]))];
                let m = b.node("ReduceMax", &[x], ax.clone(), None, rs.clone());
                let d = b.node("Sub", &[x, m], vec![], None, os.clone());
                let e = b.node("Exp", &[d], vec![], None, os.clone());
                let s = b.node("ReduceSum", &[e], ax, None, rs);
                b.node("Div", &[e, s], vec![], Some(o), os.clone());
            }
            "LayerNormalization" if is_f32 => {
                let x = n.input(0);
                let mut rs = os.clone();
                *rs.last_mut().unwrap() = 1;
                let ax = vec![("axes", Attr::Ints(vec![os.len() as i64 - 1]))];
                let mean = b.node("ReduceMean", &[x], ax.clone(), None, rs.clone());
                let d = b.node("Sub", &[x, mean], vec![], None, os.clone());
                let sq = b.node("Mul", &[d, d], vec![], None, os.clone());
                let var = b.node("ReduceMean", &[sq], ax, None, rs.clone());
                let eps = b.konst(Tensor::scalar_f32(n.attr_float("epsilon", 1e-5)));
                let ve = b.node("Add", &[var, eps], vec![], None, rs.clone());
                let sd = b.node("Sqrt", &[ve], vec![], None, rs);
                let nm = b.node("Div", &[d, sd], vec![], None, os.clone());
                if n.has_input(2) {
                    let y = b.node("Mul", &[nm, n.input(1)], vec![], None, os.clone());
                    b.node("Add", &[y, n.input(2)], vec![], Some(o), os.clone());
                } else {
                    b.node("Mul", &[nm, n.input(1)], vec![], Some(o), os.clone());
                }
            }
            "ReduceL2" | "ReduceMean" | "ReduceSum" | "ReduceMax" if is_f32 => {
                let x = n.input(0);
                let r = shape_of(&b, x).len();
                let raw = if n.has_input(1) {
                    b.g.konst(n.input(1)).map(Tensor::to_i64)
                } else {
                    n.attr_ints("axes")
                };
                let mut axes: Vec<i64> = match raw {
                    Some(v) if !v.is_empty() => v.iter().map(|&a| axis(a, r) as i64).collect(),
                    _ => (0..r as i64).collect(),
                };
                axes.sort_unstable();
                // Keep reduced dimensions; reshape away afterwards if needed.
                let ks: Vec<usize> = shape_of(&b, x)
                    .iter()
                    .enumerate()
                    .map(|(d, &v)| if axes.contains(&(d as i64)) { 1 } else { v })
                    .collect();
                let keep = n.attr_int("keepdims", 1) == 1;
                let target = if keep { Some(o) } else { None };
                let attrs = vec![("axes", Attr::Ints(axes))];
                let red = if n.op == "ReduceL2" {
                    let sq = b.node("Mul", &[x, x], vec![], None, shape_of(&b, x));
                    let s = b.node("ReduceSum", &[sq], attrs, None, ks.clone());
                    b.node("Sqrt", &[s], vec![], target, ks)
                } else {
                    b.node(&n.op, &[x], attrs, target, ks)
                };
                if !keep {
                    let spec = b.konst(Tensor::i64(
                        vec![os.len()],
                        os.iter().map(|&d| d as i64).collect(),
                    ));
                    b.node("Reshape", &[red, spec], vec![], Some(o), os.clone());
                }
            }
            "Pow" if is_f32 && b.g.konst(n.input(1)).is_some_and(|t| t.len() == 1) => {
                let e = b.g.konst(n.input(1)).unwrap().as_f32()[0];
                let x = n.input(0);
                match e {
                    2.0 => {
                        b.node("Mul", &[x, x], vec![], Some(o), os);
                    }
                    0.5 => {
                        b.node("Sqrt", &[x], vec![], Some(o), os);
                    }
                    1.0 => {
                        b.node("Identity", &[x], vec![], Some(o), os);
                    }
                    _ => b.out.push(n),
                }
            }
            _ => b.out.push(n),
        }
    }
    g.nodes = b.out;
    Ok(())
}

/// The standard pipeline; returns a summary for logging.
pub fn optimize(g: &mut Graph) -> Result<HashMap<&'static str, usize>, String> {
    let mut stats = HashMap::new();
    stats.insert("nodes_in", g.nodes.len());
    stats.insert("folded", fold_and_infer(g)?);
    canonicalize(g)?;
    stats.insert("folded_after_canon", fold_and_infer(g)?);
    stats.insert("dead", dce(g));
    stats.insert("nodes_out", g.nodes.len());
    Ok(stats)
}
