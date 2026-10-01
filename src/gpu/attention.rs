//! Fused attention: a matmul whose output only a row kernel reads (the
//! scores, with their scale and mask in its epilogue), that row kernel
//! (softmax), and a matmul whose A operand is the row kernel's output (the
//! probabilities) become one kernel. A block takes BM rows of one batch
//! entry: it computes their scores into shared memory, runs the row
//! kernel's stages over them there, and multiplies the probabilities by
//! the second operand, so neither the scores nor the probabilities ever
//! reach device memory.
//!
//! The rewrite applies only when it is exact: the scores and
//! probabilities are read by nothing else, every access to them is the
//! canonical row-major index of their row and column (checked on the index
//! expressions, size-1 dimensions aside), and the three kernels agree on
//! the batch dimensions and the row count.

use super::codegen::{
    GpuKernel, Gx, collect_args, decode, emu_entry, epilogue, mm_names, signature, stage_code,
};
use crate::fuse::{BOp, Kernel, LoopK, MatmulK, Plan, Stage, Step};
use crate::ir::{Buf, E, Lin, Ranges, Var, linearize};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// Shared-memory budget of the fused kernel (the default per-block limit).
const SMEM_LIMIT: usize = 48 * 1024;
const THREADS: usize = 128;
/// Rows of K (scores phase) and of V (output phase) staged at a time.
const CHUNK: usize = 32;
/// Row-buffer ids standing for the scores and probabilities rows.
const SID: u32 = u32::MAX - 1;
const PID: u32 = u32::MAX - 2;

fn drop_unit(l: &Lin, dims: &[(Var, usize)], r: &Ranges) -> Lin {
    let mut l = l.clone();
    for &(v, n) in dims {
        if n == 1 {
            l = l.subst(v, &Lin::konst(0), r);
        }
    }
    l
}

/// Whether `l` is the row-major index of `dims` (size-1 dimensions aside).
fn canonical(l: &Lin, dims: &[(Var, usize)], r: &Ranges) -> bool {
    let want = linearize(
        &dims.iter().map(|d| Lin::var(d.0)).collect::<Vec<_>>(),
        &dims.iter().map(|d| d.1).collect::<Vec<_>>(),
    );
    drop_unit(l, dims, r) == drop_unit(&want, dims, r)
}

/// Every load of value `v` in `e` (their indices).
fn loads_of<'a>(e: &'a E, v: usize, out: &mut Vec<&'a Lin>) {
    match e {
        E::Load(Buf::Value(x), l) if *x == v => out.push(l),
        E::Un(_, a) => loads_of(a, v, out),
        E::Bin(_, a, b) | E::Sel(_, a, b) => {
            loads_of(a, v, out);
            loads_of(b, v, out);
        }
        _ => {}
    }
}

/// `e` with every load of `v` replaced by row buffer `id` at column `j`.
fn to_row(e: &E, v: usize, id: u32, j: Var) -> E {
    match e {
        E::Load(Buf::Value(x), _) if *x == v => E::Load(Buf::Row(id), Lin::var(j)),
        E::Un(op, a) => E::Un(*op, Box::new(to_row(a, v, id, j))),
        E::Bin(op, a, b) => E::Bin(
            *op,
            Box::new(to_row(a, v, id, j)),
            Box::new(to_row(b, v, id, j)),
        ),
        E::Sel(c, a, b) => E::Sel(
            c.clone(),
            Box::new(to_row(a, v, id, j)),
            Box::new(to_row(b, v, id, j)),
        ),
        _ => e.clone(),
    }
}

fn step_exprs(s: &Step) -> Vec<&E> {
    match s {
        Step::Kernel(Kernel::Matmul(mk)) => {
            let mut v = vec![&mk.a, &mk.epi];
            v.extend(mk.lets.iter().map(|l| &l.1));
            if let BOp::Expr(e) = &mk.b {
                v.push(e);
            }
            v
        }
        Step::Kernel(Kernel::Loop(l)) => l
            .stages
            .iter()
            .map(|s| match s {
                Stage::Reduce { body, .. }
                | Stage::Scalar { body, .. }
                | Stage::RowBuf { body, .. }
                | Stage::Store { body, .. } => body,
            })
            .collect(),
        Step::Host(_) => Vec::new(),
    }
}

/// Values read by each step.
fn readers(plan: &Plan) -> HashMap<usize, HashSet<usize>> {
    let mut out: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (si, s) in plan.steps.iter().enumerate() {
        let mut bufs = Vec::new();
        for e in step_exprs(s) {
            e.loads(&mut bufs);
        }
        for b in bufs {
            if let Buf::Value(v) = b {
                out.entry(v).or_default().insert(si);
            }
        }
        if let Step::Host(n) = s {
            for &i in plan.g.nodes[*n].inputs.iter().flatten() {
                out.entry(i).or_default().insert(si);
            }
        }
    }
    out
}

/// The attention patterns of a plan: steps (scores, softmax, output).
pub fn find(plan: &Plan) -> Vec<[usize; 3]> {
    let rd = readers(plan);
    let only = |v: usize, s: usize| {
        !plan.g.outputs.contains(&v) && rd.get(&v).is_some_and(|r| r.len() == 1 && r.contains(&s))
    };
    let mut out = Vec::new();
    for i in 0..plan.steps.len().saturating_sub(2) {
        let (
            Step::Kernel(Kernel::Matmul(m1)),
            Step::Kernel(Kernel::Loop(lk)),
            Step::Kernel(Kernel::Matmul(m3)),
        ) = (&plan.steps[i], &plan.steps[i + 1], &plan.steps[i + 2])
        else {
            continue;
        };
        if matches(m1, lk, m3) && only(m1.out, i + 1) {
            let p = p_value(lk).unwrap();
            if only(p, i + 2) && attention_kernel(m1, lk, m3).is_some() {
                out.push([i, i + 1, i + 2]);
            }
        }
    }
    out
}

/// The row kernel's single store (the probabilities), if it has one.
fn p_value(lk: &LoopK) -> Option<usize> {
    let stores: Vec<&Stage> = lk
        .stages
        .iter()
        .filter(|s| matches!(s, Stage::Store { .. }))
        .collect();
    match stores.as_slice() {
        [
            Stage::Store {
                out,
                per_row: false,
                ..
            },
        ] => Some(*out),
        _ => None,
    }
}

fn matches(m1: &MatmulK, lk: &LoopK, m3: &MatmulK) -> bool {
    let s = m1.out;
    let Some(p) = p_value(lk) else { return false };
    if !matches!(m1.b, BOp::Expr(_)) || !matches!(m3.b, BOp::Expr(_)) || s == p {
        return false;
    }
    let b1: Vec<usize> = m1.batch.iter().map(|b| b.1).collect();
    let b3: Vec<usize> = m3.batch.iter().map(|b| b.1).collect();
    let rows = b1.iter().product::<usize>() * m1.m;
    if b1 != b3 || m3.m != m1.m || m3.k != m1.n || lk.j.1 != m1.n {
        return false;
    }
    if lk.outer.iter().map(|o| o.1).product::<usize>() != rows {
        return false;
    }
    // The scores are stored at their canonical index.
    let mut d1 = m1.batch.clone();
    d1.extend([(m1.vi, m1.m), (m1.vj, m1.n)]);
    if !canonical(&m1.out_idx, &d1, &m1.ranges) {
        return false;
    }
    // The row kernel reads scores and writes probabilities canonically.
    let mut dl = lk.outer.clone();
    dl.push(lk.j);
    for st in &lk.stages {
        let body = match st {
            Stage::Reduce { body, .. }
            | Stage::Scalar { body, .. }
            | Stage::RowBuf { body, .. }
            | Stage::Store { body, .. } => body,
        };
        let mut ls = Vec::new();
        loads_of(body, s, &mut ls);
        loads_of(body, p, &mut ls);
        if ls.iter().any(|l| !canonical(l, &dl, &lk.ranges)) {
            return false;
        }
        if let Stage::Store { idx, .. } = st
            && !canonical(idx, &dl, &lk.ranges)
        {
            return false;
        }
    }
    // The output matmul reads the probabilities canonically as A, and
    // nothing else of S or P.
    let mut d3 = m3.batch.clone();
    d3.extend([(m3.vi, m3.m), (m3.vk, m3.k)]);
    match &m3.a {
        E::Load(Buf::Value(x), l) if *x == p && canonical(l, &d3, &m3.ranges) => {}
        _ => return false,
    }
    let mut others = vec![&m3.epi];
    others.extend(m3.lets.iter().map(|l| &l.1));
    if let BOp::Expr(e) = &m3.b {
        others.push(e);
    }
    for e in others {
        let mut ls = Vec::new();
        loads_of(e, p, &mut ls);
        loads_of(e, s, &mut ls);
        if !ls.is_empty() {
            return false;
        }
    }
    true
}

/// The fused kernel, if its shared memory fits.
pub fn attention_kernel(m1: &MatmulK, lk: &LoopK, m3: &MatmulK) -> Option<GpuKernel> {
    let s_val = m1.out;
    let p_val = p_value(lk)?;
    let (n1, k1, n2) = (m1.n, m1.k, m3.n);
    let rowbufs: Vec<u32> = lk
        .stages
        .iter()
        .filter_map(|s| match s {
            Stage::RowBuf { id, .. } => Some(*id),
            _ => None,
        })
        .collect();
    // Rows per block: the most that fit in shared memory (scores,
    // probabilities and row buffers for each row, plus the Q rows and one
    // chunk each of K and V), in whole passes of the row groups.
    let tpr = n1.next_power_of_two().clamp(1, 32);
    let rp = THREADS / tpr;
    let floats = |bm: usize| {
        (2 + rowbufs.len()) * bm * n1 + bm * (k1 + 1) + CHUNK * (k1 + 1) + CHUNK * (n2 + 1)
    };
    let bm = [32usize, 16, 8, 4]
        .into_iter()
        .map(|b| b.div_ceil(rp) * rp)
        .find(|&b| floats(b) * 4 <= SMEM_LIMIT)?;
    // The row kernel, with scores and probabilities as row buffers.
    let jv = lk.j.0;
    let mut rk = lk.clone();
    rk.stages = lk
        .stages
        .iter()
        .map(|st| match st {
            Stage::Reduce { id, op, body } => Stage::Reduce {
                id: *id,
                op: *op,
                body: to_row(&to_row(body, s_val, SID, jv), p_val, PID, jv),
            },
            Stage::Scalar { id, body } => Stage::Scalar {
                id: *id,
                body: to_row(&to_row(body, s_val, SID, jv), p_val, PID, jv),
            },
            Stage::RowBuf { id, body } => Stage::RowBuf {
                id: *id,
                body: to_row(&to_row(body, s_val, SID, jv), p_val, PID, jv),
            },
            Stage::Store { body, .. } => Stage::RowBuf {
                id: PID,
                body: to_row(&to_row(body, s_val, SID, jv), p_val, PID, jv),
            },
        })
        .collect();
    // Arguments: the output, then everything any phase reads.
    let mut exprs: Vec<&E> = vec![&m1.a, &m1.epi, &m3.epi];
    exprs.extend(m1.lets.iter().map(|l| &l.1));
    exprs.extend(m3.lets.iter().map(|l| &l.1));
    let (BOp::Expr(b1), BOp::Expr(b3)) = (&m1.b, &m3.b) else {
        return None;
    };
    exprs.push(b1);
    exprs.push(b3);
    let row_exprs: Vec<&E> = rk
        .stages
        .iter()
        .map(|s| match s {
            Stage::Reduce { body, .. }
            | Stage::Scalar { body, .. }
            | Stage::RowBuf { body, .. }
            | Stage::Store { body, .. } => body,
        })
        .collect();
    exprs.extend(row_exprs);
    let (args, map) = collect_args(&exprs, &[m3.out]);
    let floats = floats(bm);
    let batch: usize = m1.batch.iter().map(|b| b.1).product();
    let m = m1.m;
    let (k1p, n2p) = (k1 + 1, n2 + 1);
    let mut s = String::new();
    let _ = writeln!(s, "{} {{", signature(THREADS, &args, 1));
    s.push_str("  SMEM;\n");
    let _ = writeln!(
        s,
        "  float *Ss = (float *)kiln_smem, *Ps = Ss + {};",
        bm * n1
    );
    let mut off = 2 * bm * n1;
    let mut rb_off = Vec::new();
    for _ in &rowbufs {
        rb_off.push(off);
        off += bm * n1;
    }
    let _ = writeln!(
        s,
        "  float *Qs = (float *)kiln_smem + {off}, *Ks = Qs + {}, *Vs = Ks + {};",
        bm * k1p,
        CHUNK * k1p
    );
    let _ = writeln!(
        s,
        "  const int tid = threadIdx.x, m0 = blockIdx.x * {bm}, tb = blockIdx.y;"
    );
    // Phase 1: scores of rows m0..m0+BM, chunk by chunk of K's rows.
    let g1 = Gx::new(&map, mm_names(m1), Some("acc".into()), true);
    s.push_str("  {\n");
    s.push_str(&decode(&m1.batch, "tb", "    "));
    let _ = writeln!(
        s,
        "    for (int e = tid; e < {}; e += {THREADS}) {{ const int r = e / {k1}, kk = e % {k1}, _i = m0 + r, _k = kk; Qs[r * {k1p} + kk] = (_i < {m}) ? {} : 0.0f; }}",
        bm * k1,
        g1.e(&m1.a)
    );
    let _ = writeln!(s, "    for (int j0 = 0; j0 < {n1}; j0 += {CHUNK}) {{");
    s.push_str("      KSYNC();\n");
    let _ = writeln!(
        s,
        "      for (int e = tid; e < {}; e += {THREADS}) {{ const int jj = e / {k1}, kk = e % {k1}, _j = j0 + jj, _k = kk; Ks[jj * {k1p} + kk] = (_j < {n1}) ? {} : 0.0f; }}",
        CHUNK * k1,
        g1.e(b1)
    );
    s.push_str("      KSYNC();\n");
    let _ = writeln!(
        s,
        "      for (int o = tid; o < {}; o += {THREADS}) {{\n        const int r = o / {CHUNK}, jj = o % {CHUNK}, _i = m0 + r, _j = j0 + jj;\n        if (_j < {n1}) {{\n          float acc = 0.0f;\n          #pragma unroll 8\n          for (int kk = 0; kk < {k1}; kk++) acc += Qs[r * {k1p} + kk] * Ks[jj * {k1p} + kk];",
        bm * CHUNK
    );
    let _ = writeln!(s, "          if (_i < {m}) {{");
    let mut ep = String::new();
    for (id, e) in &m1.lets {
        let _ = writeln!(ep, "            const float e{id} = {};", g1.e(e));
    }
    let _ = writeln!(ep, "            Ss[r * {n1} + _j] = {};", g1.e(&m1.epi));
    s.push_str(&ep);
    let _ = writeln!(s, "          }} else Ss[r * {n1} + _j] = 0.0f;");
    s.push_str("        }\n      }\n    }\n  }\n  KSYNC();\n");
    // Phase 2: the row kernel over the BM rows, a group of TPR threads per row.
    let g2 = Gx::new(&map, HashMap::new(), None, false);
    let _ = writeln!(s, "  for (int r0 = 0; r0 < {bm}; r0 += {rp}) {{");
    let _ = writeln!(
        s,
        "    const int lane = tid % {tpr}, r = r0 + tid / {tpr}, row = tb * {m} + m0 + r;"
    );
    let _ = writeln!(s, "    const bool live = m0 + r < {m};");
    let _ = writeln!(s, "    const int t = live ? row : 0;");
    s.push_str(&decode(&lk.outer, "t", "    "));
    let _ = writeln!(
        s,
        "    float *rb{SID} = Ss + r * {n1}, *rb{PID} = Ps + r * {n1};"
    );
    for (id, o) in rowbufs.iter().zip(&rb_off) {
        let _ = writeln!(
            s,
            "    float *rb{id} = (float *)kiln_smem + {o} + r * {n1};"
        );
    }
    s.push_str("    (void)live;\n");
    s.push_str(&indent(&stage_code(&rk, &g2, &map, tpr), "  "));
    s.push_str("  }\n  KSYNC();\n");
    // Phase 3: the output rows, chunk by chunk of V's rows.
    let g3 = Gx::new(&map, mm_names(m3), Some("acc".into()), true);
    let q3 = (bm * n2).div_ceil(THREADS);
    s.push_str("  {\n");
    s.push_str(&decode(&m3.batch, "tb", "    "));
    let _ = writeln!(s, "    float c[{q3}];");
    let _ = writeln!(
        s,
        "    #pragma unroll\n    for (int q = 0; q < {q3}; q++) c[q] = 0.0f;"
    );
    let _ = writeln!(s, "    for (int k0 = 0; k0 < {n1}; k0 += {CHUNK}) {{");
    let _ = writeln!(
        s,
        "      for (int e = tid; e < {}; e += {THREADS}) {{ const int kk = e / {n2}, jj = e % {n2}, _k = k0 + kk, _j = jj; Vs[kk * {n2p} + jj] = (_k < {n1}) ? {} : 0.0f; }}",
        CHUNK * n2,
        g3.e(b3)
    );
    s.push_str("      KSYNC();\n");
    let _ = writeln!(
        s,
        "      #pragma unroll\n      for (int q = 0; q < {q3}; q++) {{\n        const int o = tid + q * {THREADS}, r = o / {n2}, jj = o % {n2};\n        if (o < {}) {{\n          const int kn = {n1} - k0 < {CHUNK} ? {n1} - k0 : {CHUNK};\n          for (int kk = 0; kk < kn; kk++) c[q] += Ps[r * {n1} + k0 + kk] * Vs[kk * {n2p} + jj];\n        }}\n      }}",
        bm * n2
    );
    s.push_str("      KSYNC();\n    }\n");
    let _ = writeln!(
        s,
        "    #pragma unroll\n    for (int q = 0; q < {q3}; q++) {{\n      const int o = tid + q * {THREADS}, r = o / {n2}, _j = o % {n2}, _i = m0 + r;\n      if (o < {} && _i < {m}) {{\n        const float acc = c[q];",
        bm * n2
    );
    s.push_str(&epilogue(m3, &g3, &map, "        "));
    s.push_str("      }\n    }\n  }\n}\n");
    s.push_str(&emu_entry(&args));
    Some(GpuKernel {
        src: s,
        args,
        outs: vec![m3.out],
        grid: [m.div_ceil(bm) as u32, batch as u32, 1],
        block: THREADS as u32,
        smem: (floats * 4) as u32,
    })
}

fn indent(code: &str, pre: &str) -> String {
    code.lines().map(|l| format!("{pre}{l}\n")).collect()
}
