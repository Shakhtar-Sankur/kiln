//! The reference interpreter: every supported ONNX operator, written for
//! clarity. It is kiln's correctness oracle, evaluates constant subgraphs
//! during compilation, and runs the few integer and boolean operations
//! (masks, token lookups) that compiled code leaves to the host.

use crate::graph::{Attr, Graph, Node};
use crate::tensor::{Data, Tensor, numel, strides};
use std::collections::HashMap;

pub type Feeds = HashMap<usize, Tensor>;

/// Runs the whole graph; returns every computed value.
pub fn run(g: &Graph, feeds: &Feeds) -> Result<HashMap<usize, Tensor>, String> {
    let mut env: HashMap<usize, Tensor> = feeds.clone();
    for (i, v) in g.values.iter().enumerate() {
        if let Some(t) = &v.konst {
            env.entry(i).or_insert_with(|| t.clone());
        }
    }
    for n in &g.nodes {
        let ins: Vec<Option<&Tensor>> = n
            .inputs
            .iter()
            .map(|i| i.and_then(|i| env.get(&i)))
            .collect();
        for (k, i) in n.inputs.iter().enumerate() {
            if let Some(i) = i
                && ins[k].is_none()
            {
                return Err(format!(
                    "{} ({}): input {} not computed",
                    n.name, n.op, g.values[*i].name
                ));
            }
        }
        let outs = eval(n, &ins).map_err(|e| format!("{} ({}): {e}", n.name, n.op))?;
        for (o, t) in n.outputs.iter().zip(outs) {
            env.insert(*o, t);
        }
    }
    Ok(env)
}

fn axis(a: i64, rank: usize) -> usize {
    if a < 0 {
        (a + rank as i64) as usize
    } else {
        a as usize
    }
}

/// The broadcast shape of two shapes (NumPy rules).
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let r = a.len().max(b.len());
    let mut out = vec![0; r];
    for i in 0..r {
        let da = if i + a.len() >= r {
            a[i + a.len() - r]
        } else {
            1
        };
        let db = if i + b.len() >= r {
            b[i + b.len() - r]
        } else {
            1
        };
        out[i] = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            return Err(format!("cannot broadcast {a:?} with {b:?}"));
        };
    }
    Ok(out)
}

/// Strides to read a tensor of `shape` at indices of `out` (0 where broadcast).
fn bstrides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let s = strides(shape);
    let off = out.len() - shape.len();
    (0..out.len())
        .map(|i| {
            if i < off || shape[i - off] == 1 {
                0
            } else {
                s[i - off]
            }
        })
        .collect()
}

/// Visits every position of `out` in row-major order, passing the offset
/// into each input (whose strides, 0 where broadcast, are `st`). Offsets
/// advance incrementally: the innermost dimension in a tight loop, carries
/// only on the outer ones.
fn walk(out: &[usize], st: &[Vec<usize>], mut f: impl FnMut(&[usize])) {
    let n = numel(out);
    if n == 0 {
        return;
    }
    let r = out.len();
    let k = st.len();
    let mut cur = vec![0usize; k];
    if r == 0 {
        f(&cur);
        return;
    }
    let inner = out[r - 1];
    let step: Vec<usize> = st.iter().map(|s| s[r - 1]).collect();
    let mut idx = vec![0usize; r];
    let mut base = vec![0usize; k];
    for _ in 0..n / inner {
        cur.copy_from_slice(&base);
        for _ in 0..inner {
            f(&cur);
            for q in 0..k {
                cur[q] += step[q];
            }
        }
        for d in (0..r - 1).rev() {
            idx[d] += 1;
            for q in 0..k {
                base[q] += st[q][d];
            }
            if idx[d] < out[d] {
                break;
            }
            for q in 0..k {
                base[q] -= st[q][d] * out[d];
            }
            idx[d] = 0;
        }
    }
}

/// Offsets into each input for every output position, in order.
fn offsets(out: &[usize], ins: &[&[usize]]) -> Vec<Vec<usize>> {
    let n = numel(out);
    let st: Vec<Vec<usize>> = ins.iter().map(|s| bstrides(s, out)).collect();
    let mut res = vec![Vec::with_capacity(n); ins.len()];
    walk(out, &st, |o| {
        for (r, &x) in res.iter_mut().zip(o) {
            r.push(x);
        }
    });
    res
}

fn binary_f32(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor, String> {
    let out = broadcast_shape(&a.shape, &b.shape)?;
    let (x, y) = (a.as_f32(), b.as_f32());
    let mut v = Vec::with_capacity(numel(&out));
    if a.shape == out && b.shape == out {
        v.extend(x.iter().zip(y).map(|(&p, &q)| f(p, q)));
    } else {
        let st = [bstrides(&a.shape, &out), bstrides(&b.shape, &out)];
        walk(&out, &st, |o| v.push(f(x[o[0]], y[o[1]])));
    }
    Ok(Tensor::f32(out, v))
}

fn binary_i64(a: &Tensor, b: &Tensor, f: impl Fn(i64, i64) -> i64) -> Result<Tensor, String> {
    let out = broadcast_shape(&a.shape, &b.shape)?;
    let o = offsets(&out, &[&a.shape, &b.shape]);
    let (x, y) = (a.to_i64(), b.to_i64());
    let v = o[0]
        .iter()
        .zip(&o[1])
        .map(|(&i, &j)| f(x[i], y[j]))
        .collect();
    Ok(Tensor::i64(out, v))
}

fn compare(a: &Tensor, b: &Tensor, f: impl Fn(f64, f64) -> bool) -> Result<Tensor, String> {
    let out = broadcast_shape(&a.shape, &b.shape)?;
    let o = offsets(&out, &[&a.shape, &b.shape]);
    let num = |t: &Tensor| -> Vec<f64> {
        match &t.data {
            Data::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Data::I64(v) => v.iter().map(|&x| x as f64).collect(),
            Data::Bool(v) => v.iter().map(|&x| f64::from(u8::from(x))).collect(),
        }
    };
    let (x, y) = (num(a), num(b));
    let v = o[0]
        .iter()
        .zip(&o[1])
        .map(|(&i, &j)| f(x[i], y[j]))
        .collect();
    Ok(Tensor::bool(out, v))
}

fn unary(a: &Tensor, f: impl Fn(f32) -> f32) -> Tensor {
    Tensor::f32(a.shape.clone(), a.as_f32().iter().map(|&x| f(x)).collect())
}

/// The error function to double precision (Taylor series near zero, a
/// continued fraction for the tail).
pub fn erf(x: f64) -> f64 {
    if x.abs() < 2.5 {
        let x2 = x * x;
        let (mut term, mut sum) = (x, x);
        for n in 1..200 {
            term *= -x2 / n as f64;
            let add = term / (2 * n + 1) as f64;
            sum += add;
            if add.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        return sum * 2.0 / std::f64::consts::PI.sqrt();
    }
    let z = x.abs();
    let mut f = z;
    let (mut c, mut d) = (z, 0.0);
    for n in 1..300 {
        let a = n as f64 / 2.0;
        d = 1.0 / (z + a * d);
        c = z + a / c;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    let erfc = (-z * z).exp() / std::f64::consts::PI.sqrt() / f;
    x.signum() * (1.0 - erfc)
}

/// Gathers `src` (of `shape`) at the positions of a permutation/selection:
/// out[idx] = src[map(idx)] for every index of `out_shape`.
fn remap<T: Copy>(src: &[T], out_shape: &[usize], map: impl Fn(&[usize]) -> usize) -> Vec<T> {
    let n = numel(out_shape);
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..n {
        out.push(src[map(&idx)]);
        for d in (0..out_shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

fn remap_any(t: &Tensor, out_shape: Vec<usize>, map: impl Fn(&[usize]) -> usize) -> Tensor {
    match &t.data {
        Data::F32(v) => Tensor::f32(out_shape.clone(), remap(v, &out_shape, map)),
        Data::I64(v) => Tensor::i64(out_shape.clone(), remap(v, &out_shape, map)),
        Data::Bool(v) => Tensor::bool(out_shape.clone(), remap(v, &out_shape, map)),
    }
}

/// C[b] = A[b] @ B[b] with broadcast batch dimensions.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    let (mut ash, mut bsh) = (a.shape.clone(), b.shape.clone());
    let (a1, b1) = (ash.len() == 1, bsh.len() == 1);
    if a1 {
        ash.insert(0, 1);
    }
    if b1 {
        bsh.push(1);
    }
    let (m, k) = (ash[ash.len() - 2], ash[ash.len() - 1]);
    let (k2, n) = (bsh[bsh.len() - 2], bsh[bsh.len() - 1]);
    if k != k2 {
        return Err(format!("matmul {:?} x {:?}", a.shape, b.shape));
    }
    let batch = broadcast_shape(&ash[..ash.len() - 2], &bsh[..bsh.len() - 2])?;
    let nb = numel(&batch);
    let oa = offsets(&batch, &[&ash[..ash.len() - 2], &bsh[..bsh.len() - 2]]);
    let (x, y) = (a.as_f32(), b.as_f32());
    let mut out = vec![0f32; nb * m * n];
    for bi in 0..nb {
        let (ao, bo) = (oa[0][bi] * m * k, oa[1][bi] * k * n);
        let c = &mut out[bi * m * n..(bi + 1) * m * n];
        for i in 0..m {
            let row = &mut c[i * n..(i + 1) * n];
            for kk in 0..k {
                let av = x[ao + i * k + kk];
                let br = &y[bo + kk * n..bo + (kk + 1) * n];
                for (r, &bv) in row.iter_mut().zip(br) {
                    *r += av * bv;
                }
            }
        }
    }
    let mut shape = batch;
    if !a1 {
        shape.push(m);
    }
    if !b1 {
        shape.push(n);
    }
    Ok(Tensor::f32(shape, out))
}

fn transpose(t: &Tensor, perm: &[usize]) -> Tensor {
    let out: Vec<usize> = perm.iter().map(|&p| t.shape[p]).collect();
    let s = strides(&t.shape);
    remap_any(t, out, |idx| {
        idx.iter().zip(perm).map(|(&i, &p)| i * s[p]).sum()
    })
}

fn reduce(
    t: &Tensor,
    axes: &[usize],
    keep: bool,
    init: f32,
    f: impl Fn(f32, f32) -> f32,
    fin: impl Fn(f32, usize) -> f32,
) -> Tensor {
    let r = t.shape.len();
    let mut out_keep = t.shape.clone();
    for &a in axes {
        out_keep[a] = 1;
    }
    let count: usize = axes.iter().map(|&a| t.shape[a]).product();
    let n_out = numel(&out_keep);
    let mut acc = vec![init; n_out];
    let os = strides(&out_keep);
    let x = t.as_f32();
    let mut idx = vec![0usize; r];
    for &v in x {
        let o: usize = (0..r)
            .map(|d| if out_keep[d] == 1 { 0 } else { idx[d] * os[d] })
            .sum();
        acc[o] = f(acc[o], v);
        for d in (0..r).rev() {
            idx[d] += 1;
            if idx[d] < t.shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    let v: Vec<f32> = acc.into_iter().map(|a| fin(a, count)).collect();
    let shape = if keep {
        out_keep
    } else {
        (0..r)
            .filter(|d| !axes.contains(d))
            .map(|d| t.shape[d])
            .collect()
    };
    Tensor::f32(shape, v)
}

fn reduce_axes(n: &Node, ins: &[Option<&Tensor>], rank: usize) -> Vec<usize> {
    let raw = match ins.get(1).copied().flatten() {
        Some(t) => Some(t.to_i64()),
        None => n.attr_ints("axes"),
    };
    let mut a: Vec<usize> = match raw {
        Some(v) if !v.is_empty() => v.iter().map(|&x| axis(x, rank)).collect(),
        _ => (0..rank).collect(),
    };
    a.sort_unstable();
    a
}

/// Shape for Reshape: 0 copies the input dimension, -1 is inferred.
pub fn reshape_target(input: &[usize], spec: &[i64]) -> Result<Vec<usize>, String> {
    let mut out: Vec<usize> = Vec::with_capacity(spec.len());
    let mut infer = None;
    for (i, &s) in spec.iter().enumerate() {
        match s {
            0 => out.push(*input.get(i).ok_or("reshape: 0 beyond input rank")?),
            -1 => {
                infer = Some(i);
                out.push(1);
            }
            s => out.push(s as usize),
        }
    }
    if let Some(i) = infer {
        let known: usize = out.iter().product();
        out[i] = numel(input) / known.max(1);
    }
    if numel(&out) != numel(input) {
        return Err(format!("reshape {input:?} to {spec:?}"));
    }
    Ok(out)
}

/// Evaluates one node.
pub fn eval(n: &Node, ins: &[Option<&Tensor>]) -> Result<Vec<Tensor>, String> {
    let x =
        |i: usize| -> &Tensor { ins[i].unwrap_or_else(|| panic!("{}: missing input {i}", n.op)) };
    let out = match n.op.as_str() {
        "Identity" => x(0).clone(),
        "Constant" => match n.attrs.get("value") {
            Some(Attr::Tensor(t)) => t.clone(),
            _ => match (
                n.attrs.get("value_float"),
                n.attrs.get("value_ints"),
                n.attrs.get("value_int"),
            ) {
                (Some(Attr::Float(f)), _, _) => Tensor::scalar_f32(*f),
                (_, Some(Attr::Ints(v)), _) => Tensor::i64(vec![v.len()], v.clone()),
                (_, _, Some(Attr::Int(v))) => Tensor::i64(vec![], vec![*v]),
                _ => return Err("unsupported Constant".into()),
            },
        },
        "ConstantOfShape" => {
            let shape: Vec<usize> = x(0).to_i64().iter().map(|&d| d as usize).collect();
            let n_el = numel(&shape);
            match n.attrs.get("value") {
                Some(Attr::Tensor(t)) => match &t.data {
                    Data::F32(v) => Tensor::f32(shape, vec![v[0]; n_el]),
                    Data::I64(v) => Tensor::i64(shape, vec![v[0]; n_el]),
                    Data::Bool(v) => Tensor::bool(shape, vec![v[0]; n_el]),
                },
                _ => Tensor::f32(shape, vec![0.0; n_el]),
            }
        }
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
            let (a, b) = (x(0), x(1));
            match (a.dtype(), n.op.as_str()) {
                (crate::tensor::DType::F32, op) => binary_f32(
                    a,
                    b,
                    match op {
                        "Add" => |p: f32, q: f32| p + q,
                        "Sub" => |p: f32, q: f32| p - q,
                        "Mul" => |p: f32, q: f32| p * q,
                        "Div" => |p: f32, q: f32| p / q,
                        _ => |p: f32, q: f32| if q == 2.0 { p * p } else { p.powf(q) },
                    },
                )?,
                (_, op) => binary_i64(
                    a,
                    b,
                    match op {
                        "Add" => |p: i64, q: i64| p + q,
                        "Sub" => |p: i64, q: i64| p - q,
                        "Mul" => |p: i64, q: i64| p * q,
                        "Div" => |p: i64, q: i64| p.div_euclid(q),
                        _ => |p: i64, q: i64| p.pow(q as u32),
                    },
                )?,
            }
        }
        "Equal" => compare(x(0), x(1), |a, b| a == b)?,
        "Greater" => compare(x(0), x(1), |a, b| a > b)?,
        "GreaterOrEqual" => compare(x(0), x(1), |a, b| a >= b)?,
        "Less" => compare(x(0), x(1), |a, b| a < b)?,
        "LessOrEqual" => compare(x(0), x(1), |a, b| a <= b)?,
        "And" | "Or" => {
            let out = broadcast_shape(&x(0).shape, &x(1).shape)?;
            let st = [bstrides(&x(0).shape, &out), bstrides(&x(1).shape, &out)];
            let (a, b) = (x(0).as_bool(), x(1).as_bool());
            let and = n.op == "And";
            let mut v = Vec::with_capacity(numel(&out));
            walk(&out, &st, |o| {
                v.push(if and {
                    a[o[0]] && b[o[1]]
                } else {
                    a[o[0]] || b[o[1]]
                })
            });
            Tensor::bool(out, v)
        }
        "Not" => Tensor::bool(
            x(0).shape.clone(),
            x(0).as_bool().iter().map(|&b| !b).collect(),
        ),
        "Where" => {
            let (c, a, b) = (x(0), x(1), x(2));
            let out = broadcast_shape(&broadcast_shape(&c.shape, &a.shape)?, &b.shape)?;
            let st = [
                bstrides(&c.shape, &out),
                bstrides(&a.shape, &out),
                bstrides(&b.shape, &out),
            ];
            let cv = c.as_bool();
            let count = numel(&out);
            match (&a.data, &b.data) {
                (Data::F32(p), Data::F32(q)) => {
                    let mut v = Vec::with_capacity(count);
                    walk(&out, &st, |o| {
                        v.push(if cv[o[0]] { p[o[1]] } else { q[o[2]] })
                    });
                    Tensor::f32(out, v)
                }
                _ => {
                    let (p, q) = (a.to_i64(), b.to_i64());
                    let mut v = Vec::with_capacity(count);
                    walk(&out, &st, |o| {
                        v.push(if cv[o[0]] { p[o[1]] } else { q[o[2]] })
                    });
                    Tensor::i64(out, v)
                }
            }
        }
        "Cast" => {
            let t = x(0);
            match n.attr_int("to", 1) {
                1 => Tensor::f32(
                    t.shape.clone(),
                    match &t.data {
                        Data::F32(v) => v.clone(),
                        Data::I64(v) => v.iter().map(|&q| q as f32).collect(),
                        Data::Bool(v) => v.iter().map(|&q| f32::from(u8::from(q))).collect(),
                    },
                ),
                6 | 7 => Tensor::i64(t.shape.clone(), t.to_i64()),
                9 => Tensor::bool(
                    t.shape.clone(),
                    t.to_i64().iter().map(|&q| q != 0).collect(),
                ),
                c => return Err(format!("Cast to {c}")),
            }
        }
        "Sqrt" => unary(x(0), f32::sqrt),
        "Reciprocal" => unary(x(0), |v| 1.0 / v),
        "Neg" => match &x(0).data {
            Data::I64(v) => Tensor::i64(x(0).shape.clone(), v.iter().map(|q| -q).collect()),
            _ => unary(x(0), |v| -v),
        },
        "Exp" => unary(x(0), f32::exp),
        "Log" => unary(x(0), f32::ln),
        "Abs" => unary(x(0), f32::abs),
        "Tanh" => unary(x(0), f32::tanh),
        "Relu" => unary(x(0), |v| v.max(0.0)),
        "Sigmoid" => unary(x(0), |v| 1.0 / (1.0 + (-v).exp())),
        "Erf" => unary(x(0), |v| erf(f64::from(v)) as f32),
        "MatMul" => matmul(x(0), x(1))?,
        "Gemm" => {
            let mut a = x(0).clone();
            let mut b = x(1).clone();
            if n.attr_int("transA", 0) == 1 {
                a = transpose(&a, &[1, 0]);
            }
            if n.attr_int("transB", 0) == 1 {
                b = transpose(&b, &[1, 0]);
            }
            let (alpha, beta) = (n.attr_float("alpha", 1.0), n.attr_float("beta", 1.0));
            let mut y = matmul(&a, &b)?;
            if alpha != 1.0 {
                y = unary(&y, |v| v * alpha);
            }
            if n.has_input(2) {
                y = binary_f32(&y, x(2), |p, q| p + beta * q)?;
            }
            y
        }
        "Shape" => Tensor::i64(
            vec![x(0).shape.len()],
            x(0).shape.iter().map(|&d| d as i64).collect(),
        ),
        "Reshape" => {
            let shape = reshape_target(&x(0).shape, &x(1).to_i64())?;
            x(0).reshaped(shape)
        }
        "Flatten" => {
            let t = x(0);
            let a = axis(n.attr_int("axis", 1), t.shape.len());
            let outer: usize = t.shape[..a].iter().product();
            t.reshaped(vec![outer, t.len() / outer.max(1)])
        }
        "Unsqueeze" | "Squeeze" => {
            let t = x(0);
            let axes = match ins.get(1).copied().flatten() {
                Some(a) => a.to_i64(),
                None => n.attr_ints("axes").unwrap_or_default(),
            };
            let mut shape = t.shape.clone();
            if n.op == "Unsqueeze" {
                let r = shape.len() + axes.len();
                let mut ax: Vec<usize> = axes.iter().map(|&a| axis(a, r)).collect();
                ax.sort_unstable();
                for a in ax {
                    shape.insert(a, 1);
                }
            } else {
                let r = shape.len();
                let ax: Vec<usize> = if axes.is_empty() {
                    (0..r).filter(|&d| shape[d] == 1).collect()
                } else {
                    axes.iter().map(|&a| axis(a, r)).collect()
                };
                shape = (0..r)
                    .filter(|d| !ax.contains(d))
                    .map(|d| shape[d])
                    .collect();
            }
            t.reshaped(shape)
        }
        "Transpose" => {
            let r = x(0).shape.len();
            let perm: Vec<usize> = n
                .attr_ints("perm")
                .map(|p| p.iter().map(|&q| q as usize).collect())
                .unwrap_or_else(|| (0..r).rev().collect());
            transpose(x(0), &perm)
        }
        "Concat" => {
            let ts: Vec<&Tensor> = ins.iter().flatten().copied().collect();
            let r = ts[0].shape.len();
            let a = axis(n.attr_int("axis", 0), r);
            let mut shape = ts[0].shape.clone();
            shape[a] = ts.iter().map(|t| t.shape[a]).sum();
            let mut starts = vec![0];
            for t in &ts {
                starts.push(starts.last().unwrap() + t.shape[a]);
            }
            let pick = |idx: &[usize]| -> (usize, usize) {
                let k = (0..ts.len()).rfind(|&k| starts[k] <= idx[a]).unwrap();
                let s = strides(&ts[k].shape);
                let off: usize = idx
                    .iter()
                    .enumerate()
                    .map(|(d, &i)| {
                        if d == a {
                            (i - starts[k]) * s[d]
                        } else {
                            i * s[d]
                        }
                    })
                    .sum();
                (k, off)
            };
            let total = numel(&shape);
            let mut picks = Vec::with_capacity(total);
            let mut idx = vec![0usize; r];
            for _ in 0..total {
                picks.push(pick(&idx));
                for d in (0..r).rev() {
                    idx[d] += 1;
                    if idx[d] < shape[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
            match ts[0].dtype() {
                crate::tensor::DType::F32 => {
                    let v: Vec<&[f32]> = ts.iter().map(|t| t.as_f32()).collect();
                    Tensor::f32(shape, picks.iter().map(|&(k, o)| v[k][o]).collect())
                }
                _ => {
                    let v: Vec<Vec<i64>> = ts.iter().map(|t| t.to_i64()).collect();
                    Tensor::i64(shape, picks.iter().map(|&(k, o)| v[k][o]).collect())
                }
            }
        }
        "Slice" => {
            let t = x(0);
            let r = t.shape.len();
            let starts = x(1).to_i64();
            let ends = x(2).to_i64();
            let axes: Vec<usize> = match ins.get(3).copied().flatten() {
                Some(a) => a.to_i64().iter().map(|&q| axis(q, r)).collect(),
                None => (0..starts.len()).collect(),
            };
            let steps = match ins.get(4).copied().flatten() {
                Some(s) => s.to_i64(),
                None => vec![1; starts.len()],
            };
            let mut begin = vec![0i64; r];
            let mut step = vec![1i64; r];
            let mut shape = t.shape.clone();
            for (k, &a) in axes.iter().enumerate() {
                let dim = t.shape[a] as i64;
                let clamp = |v: i64, lo: i64, hi: i64| v.max(lo).min(hi);
                let st = steps[k];
                let norm = |v: i64| if v < 0 { v + dim } else { v };
                let (s, e) = if st > 0 {
                    (clamp(norm(starts[k]), 0, dim), clamp(norm(ends[k]), 0, dim))
                } else {
                    (
                        clamp(norm(starts[k]), -1, dim - 1),
                        clamp(norm(ends[k]), -1, dim - 1),
                    )
                };
                begin[a] = s;
                step[a] = st;
                shape[a] = if st > 0 {
                    ((e - s).max(0) + st - 1) as usize / st as usize
                } else {
                    ((s - e).max(0) + (-st) - 1) as usize / (-st) as usize
                };
            }
            let s = strides(&t.shape);
            remap_any(t, shape, |idx| {
                idx.iter()
                    .enumerate()
                    .map(|(d, &i)| (begin[d] + i as i64 * step[d]) as usize * s[d])
                    .sum()
            })
        }
        "Gather" => {
            let (t, ind) = (x(0), x(1));
            let r = t.shape.len();
            let a = axis(n.attr_int("axis", 0), r);
            let iv = ind.to_i64();
            let mut shape: Vec<usize> = t.shape[..a].to_vec();
            shape.extend(&ind.shape);
            shape.extend(&t.shape[a + 1..]);
            // Whole runs of the trailing dimensions are contiguous in both
            // the input and the output: copy them as slices.
            let dim = t.shape[a] as i64;
            let outer: usize = t.shape[..a].iter().product();
            let inner: usize = t.shape[a + 1..].iter().product();
            let mut rows = Vec::with_capacity(iv.len());
            for &i in &iv {
                let g = if i < 0 { i + dim } else { i };
                if !(0..dim).contains(&g) {
                    return Err(format!("Gather index {i} out of range for axis of {dim}"));
                }
                rows.push(g as usize);
            }
            fn pick<T: Copy>(
                v: &[T],
                outer: usize,
                dim: usize,
                inner: usize,
                rows: &[usize],
            ) -> Vec<T> {
                let mut out = Vec::with_capacity(outer * rows.len() * inner);
                for o in 0..outer {
                    for &g in rows {
                        let at = (o * dim + g) * inner;
                        out.extend_from_slice(&v[at..at + inner]);
                    }
                }
                out
            }
            let d = dim as usize;
            let _ = r;
            match &t.data {
                Data::F32(v) => Tensor::f32(shape, pick(v, outer, d, inner, &rows)),
                Data::I64(v) => Tensor::i64(shape, pick(v, outer, d, inner, &rows)),
                Data::Bool(v) => Tensor::bool(shape, pick(v, outer, d, inner, &rows)),
            }
        }
        "GatherElements" => {
            let (t, ind) = (x(0), x(1));
            let a = axis(n.attr_int("axis", 0), t.shape.len());
            let iv = ind.to_i64();
            let s = strides(&t.shape);
            let is = strides(&ind.shape);
            let dim = t.shape[a] as i64;
            remap_any(t, ind.shape.clone(), |idx| {
                let io: usize = idx.iter().zip(&is).map(|(p, q)| p * q).sum();
                let g = if iv[io] < 0 { iv[io] + dim } else { iv[io] } as usize;
                idx.iter()
                    .enumerate()
                    .map(|(d, &i)| if d == a { g } else { i } * s[d])
                    .sum()
            })
        }
        "Expand" => {
            let t = x(0);
            let spec: Vec<usize> = x(1).to_i64().iter().map(|&d| d as usize).collect();
            let out = broadcast_shape(&t.shape, &spec)?;
            let st = [bstrides(&t.shape, &out)];
            fn expand<T: Copy>(v: &[T], out: &[usize], st: &[Vec<usize>]) -> Vec<T> {
                let mut r = Vec::with_capacity(numel(out));
                walk(out, st, |o| r.push(v[o[0]]));
                r
            }
            match &t.data {
                Data::F32(v) => Tensor::f32(out.clone(), expand(v, &out, &st)),
                Data::I64(v) => Tensor::i64(out.clone(), expand(v, &out, &st)),
                Data::Bool(v) => Tensor::bool(out.clone(), expand(v, &out, &st)),
            }
        }
        "Tile" => {
            let t = x(0);
            let reps = x(1).to_i64();
            let out: Vec<usize> = t
                .shape
                .iter()
                .zip(&reps)
                .map(|(&d, &r)| d * r as usize)
                .collect();
            let s = strides(&t.shape);
            remap_any(t, out, |idx| {
                idx.iter()
                    .enumerate()
                    .map(|(d, &i)| (i % t.shape[d]) * s[d])
                    .sum()
            })
        }
        "Softmax" => {
            let t = x(0);
            let r = t.shape.len();
            let a = axis(n.attr_int("axis", -1), r);
            if a != r - 1 {
                return Err("Softmax only over the last axis".into());
            }
            let d = t.shape[r - 1];
            let mut v = t.as_f32().to_vec();
            for row in v.chunks_exact_mut(d) {
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0f32;
                for q in row.iter_mut() {
                    *q = (*q - m).exp();
                    s += *q;
                }
                for q in row.iter_mut() {
                    *q /= s;
                }
            }
            Tensor::f32(t.shape.clone(), v)
        }
        "LayerNormalization" => {
            let t = x(0);
            let r = t.shape.len();
            if axis(n.attr_int("axis", -1), r) != r - 1 {
                return Err("LayerNormalization only over the last axis".into());
            }
            let eps = n.attr_float("epsilon", 1e-5);
            let d = t.shape[r - 1];
            let g = x(1).as_f32();
            let b = if n.has_input(2) {
                Some(x(2).as_f32())
            } else {
                None
            };
            let mut v = t.as_f32().to_vec();
            for row in v.chunks_exact_mut(d) {
                let mean = row.iter().sum::<f32>() / d as f32;
                let var = row.iter().map(|q| (q - mean) * (q - mean)).sum::<f32>() / d as f32;
                let inv = 1.0 / (var + eps).sqrt();
                for (i, q) in row.iter_mut().enumerate() {
                    *q = (*q - mean) * inv * g[i % g.len()] + b.map_or(0.0, |b| b[i % b.len()]);
                }
            }
            Tensor::f32(t.shape.clone(), v)
        }
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceL2" => {
            let t = x(0);
            let axes = reduce_axes(n, ins, t.shape.len());
            let keep = n.attr_int("keepdims", 1) == 1;
            match n.op.as_str() {
                "ReduceMean" => reduce(t, &axes, keep, 0.0, |a, v| a + v, |a, c| a / c as f32),
                "ReduceSum" => reduce(t, &axes, keep, 0.0, |a, v| a + v, |a, _| a),
                "ReduceMax" => reduce(t, &axes, keep, f32::NEG_INFINITY, f32::max, |a, _| a),
                _ => reduce(t, &axes, keep, 0.0, |a, v| a + v * v, |a, _| a.sqrt()),
            }
        }
        op => return Err(format!("operator {op} is not supported")),
    };
    Ok(vec![out])
}
