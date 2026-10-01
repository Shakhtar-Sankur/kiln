//! Executes a fusion plan by evaluating kernel IR expressions directly: a
//! slow, independent check of fusion and index arithmetic, separate from
//! code generation.

use crate::fuse::{ACC, BOp, Kernel, Plan, Red, Stage, Step};
use crate::interp::{self, Feeds};
use crate::ir::{Bin, Buf, E, Lin, Un, Var};
use crate::tensor::Tensor;
use std::collections::HashMap;

struct Env<'a> {
    vals: &'a HashMap<usize, Vec<f32>>,
    names: &'a [crate::graph::Value],
    rows: HashMap<u32, Vec<f32>>,
    scalars: HashMap<u32, f32>,
    vars: HashMap<Var, i64>,
}

fn un(op: Un, x: f32) -> f32 {
    match op {
        Un::Neg => -x,
        Un::Sqrt => x.sqrt(),
        Un::Erf => interp::erf(f64::from(x)) as f32,
        Un::Exp => x.exp(),
        Un::Log => x.ln(),
        Un::Abs => x.abs(),
        Un::Tanh => x.tanh(),
        Un::Relu => x.max(0.0),
        Un::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Un::Recip => 1.0 / x,
        Un::Nz => f32::from(u8::from(x != 0.0)),
    }
}

pub fn bin(op: Bin, a: f32, b: f32) -> f32 {
    match op {
        Bin::Add => a + b,
        Bin::Sub => a - b,
        Bin::Mul => a * b,
        Bin::Div => a / b,
        Bin::Pow => a.powf(b),
        Bin::Max => a.max(b),
    }
}

impl Env<'_> {
    fn at(&self, l: &Lin) -> usize {
        l.eval(&self.vars) as usize
    }

    fn e(&self, e: &E) -> f32 {
        match e {
            E::Load(Buf::Value(v), l) => match self.vals.get(v) {
                Some(b) => b[self.at(l)],
                None => panic!("value {} read before it was computed", self.names[*v].name),
            },
            E::Load(Buf::Row(id), l) => self.rows[id][self.at(l)],
            E::Const(c) => *c,
            E::Scalar(id) => self.scalars[id],
            E::Un(op, a) => un(*op, self.e(a)),
            E::Bin(op, a, b) => bin(*op, self.e(a), self.e(b)),
            E::Sel(c, a, b) => {
                if c.lhs.eval(&self.vars) < c.bound {
                    self.e(a)
                } else {
                    self.e(b)
                }
            }
            E::If(c, a, b) => {
                if self.e(c) != 0.0 {
                    self.e(a)
                } else {
                    self.e(b)
                }
            }
        }
    }
}

/// Visits every assignment of `dims` (row-major).
fn each(dims: &[(Var, usize)], f: &mut dyn FnMut(&HashMap<Var, i64>)) {
    let mut vars: HashMap<Var, i64> = dims.iter().map(|&(v, _)| (v, 0)).collect();
    let total: usize = dims.iter().map(|d| d.1).product();
    for _ in 0..total {
        f(&vars);
        for &(v, n) in dims.iter().rev() {
            let x = vars.get_mut(&v).unwrap();
            *x += 1;
            if (*x as usize) < n {
                break;
            }
            *x = 0;
        }
    }
}

/// Runs the plan; returns the f32 values of the graph outputs.
pub fn run(plan: &Plan, feeds: &Feeds) -> Result<HashMap<usize, Tensor>, String> {
    let g = &plan.g;
    let mut vals: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut tensors: HashMap<usize, Tensor> = feeds.clone();
    for (i, v) in g.values.iter().enumerate() {
        if let Some(t) = &v.konst {
            vals.insert(i, t.to_f32().into_owned());
            tensors.insert(i, t.clone());
        }
    }
    for (&v, t) in feeds {
        vals.insert(v, t.to_f32().into_owned());
    }
    for step in &plan.steps {
        match step {
            Step::Host(p) => {
                let n = &g.nodes[*p];
                let ins: Vec<Option<Tensor>> = n
                    .inputs
                    .iter()
                    .map(|i| {
                        i.map(|i| {
                            tensors.get(&i).cloned().unwrap_or_else(|| {
                                Tensor::f32(g.shape(i).to_vec(), vals[&i].clone())
                            })
                        })
                    })
                    .collect();
                let refs: Vec<Option<&Tensor>> = ins.iter().map(Option::as_ref).collect();
                let outs = interp::eval(n, &refs)?;
                for (&o, t) in n.outputs.iter().zip(outs) {
                    vals.insert(o, t.to_f32().into_owned());
                    tensors.insert(o, t);
                }
            }
            Step::Kernel(Kernel::Loop(k)) => {
                for st in &k.stages {
                    if let Stage::Store { out, .. } = st {
                        vals.insert(*out, vec![0.0; crate::tensor::numel(g.shape(*out))]);
                    }
                }
                let (jv, w) = k.j;
                let mut rows_done: Vec<HashMap<Var, i64>> = Vec::new();
                each(&k.outer, &mut |vars| rows_done.push(vars.clone()));
                for vars in rows_done {
                    let mut rows: HashMap<u32, Vec<f32>> = HashMap::new();
                    let mut scalars: HashMap<u32, f32> = HashMap::new();
                    for st in &k.stages {
                        // Each stage sees everything earlier stages stored.
                        let mut env = Env {
                            vals: &vals,
                            names: &g.values,
                            rows: std::mem::take(&mut rows),
                            scalars: std::mem::take(&mut scalars),
                            vars: vars.clone(),
                        };
                        let mut stores: Vec<(usize, usize, f32)> = Vec::new();
                        match st {
                            Stage::Reduce { id, op, body } => {
                                let mut acc = if *op == Red::Max {
                                    f32::NEG_INFINITY
                                } else {
                                    0.0
                                };
                                for j in 0..w {
                                    env.vars.insert(jv, j as i64);
                                    let x = env.e(body);
                                    acc = if *op == Red::Max { acc.max(x) } else { acc + x };
                                }
                                if *op == Red::Mean {
                                    acc /= w as f32;
                                }
                                env.scalars.insert(*id, acc);
                            }
                            Stage::Scalar { id, body } => {
                                env.vars.insert(jv, 0);
                                let x = env.e(body);
                                env.scalars.insert(*id, x);
                            }
                            Stage::RowBuf { id, body } => {
                                let mut row = vec![0.0; w];
                                for (j, r) in row.iter_mut().enumerate() {
                                    env.vars.insert(jv, j as i64);
                                    *r = env.e(body);
                                }
                                env.rows.insert(*id, row);
                            }
                            Stage::Store {
                                out,
                                idx,
                                body,
                                per_row,
                            } => {
                                let n = if *per_row { 1 } else { w };
                                for j in 0..n {
                                    env.vars.insert(jv, j as i64);
                                    stores.push((*out, env.at(idx), env.e(body)));
                                }
                            }
                        }
                        rows = env.rows;
                        scalars = env.scalars;
                        for (o, i, x) in stores {
                            vals.get_mut(&o).unwrap()[i] = x;
                        }
                    }
                }
            }
            Step::Kernel(Kernel::Matmul(mk)) => {
                let mut out = vec![0.0; crate::tensor::numel(g.shape(mk.out))];
                let mut dims = mk.batch.clone();
                dims.push((mk.vi, mk.m));
                dims.push((mk.vj, mk.n));
                each(&dims, &mut |vars| {
                    let mut env = Env {
                        vals: &vals,
                        names: &g.values,
                        rows: HashMap::new(),
                        scalars: HashMap::new(),
                        vars: vars.clone(),
                    };
                    let mut acc = 0f32;
                    for k in 0..mk.k {
                        env.vars.insert(mk.vk, k as i64);
                        let a = env.e(&mk.a);
                        let b = match &mk.b {
                            BOp::Packed { value, lin } => vals[value][env.at(lin)],
                            BOp::Expr(e) => env.e(e),
                        };
                        acc += a * b;
                    }
                    env.scalars.insert(ACC, acc);
                    for (id, e) in &mk.lets {
                        let x = env.e(e);
                        env.scalars.insert(*id, x);
                    }
                    out[env.at(&mk.out_idx)] = env.e(&mk.epi);
                });
                vals.insert(mk.out, out);
            }
        }
    }
    let mut res = HashMap::new();
    for &o in &g.outputs {
        let t = match vals.get(&o) {
            Some(v) => Tensor::f32(g.shape(o).to_vec(), v.clone()),
            None => tensors.get(&o).cloned().ok_or("output not computed")?,
        };
        res.insert(o, t);
    }
    Ok(res)
}
