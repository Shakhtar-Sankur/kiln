//! CUDA code generation from the same kernel IR as the CPU backend.
//!
//! * Row kernels: a group of `tpr` threads per row (a power of two; groups
//!   of up to 32 share warps, wider groups span warps of one block), the
//!   group striding over the row for each stage, with a group barrier
//!   between stages so that a stage may read anything an earlier stage of
//!   the same row wrote. Reductions are warp shuffles, then (for groups
//!   wider than a warp) one shared-memory exchange; row buffers live in
//!   shared memory.
//! * Elementwise kernels (no reductions, buffers or cross-element reads):
//!   one thread per element over the whole tensor.
//! * Matmuls: a block computes a BM x BN tile, each thread a TM x TN
//!   register tile interleaved across the block (so that shared-memory
//!   reads are broadcasts or conflict-free and stores coalesce), operands
//!   staged through shared memory BK deep; the epilogue runs per element
//!   on the accumulator before the one store.
//!
//! Every index is an `int`: no tensor kiln compiles has 2^31 elements.

use crate::codegen::Arg;
use crate::fuse::{ACC, BOp, LoopK, MatmulK, Red, Stage};
use crate::ir::{Bin, Buf, E, Lin, Un, Var};
use std::collections::HashMap;
use std::fmt::Write;

pub const PRELUDE: &str = include_str!("prelude.cuh");

/// A generated kernel. `src` names the function `KNAME`; kernels with
/// identical source share one function.
#[derive(Clone, Debug)]
pub struct GpuKernel {
    pub src: String,
    pub args: Vec<Arg>,
    pub grid: [u32; 3],
    pub block: u32,
    pub smem: u32,
}

/// fp32 matmul schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MmParams {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub tm: usize,
    pub tn: usize,
}

impl MmParams {
    pub fn threads(&self) -> usize {
        (self.bm / self.tm) * (self.bn / self.tn)
    }
}

/// Schedules the tuner and the tests choose from.
pub fn mm_space() -> Vec<MmParams> {
    let mut v = Vec::new();
    for (bm, bn, tm, tn) in [
        (128, 128, 8, 8),
        (128, 64, 8, 4),
        (64, 128, 4, 8),
        (64, 64, 4, 4),
        (64, 64, 8, 8),
        (64, 32, 4, 2),
        (32, 64, 2, 4),
        (32, 32, 2, 2),
        (32, 32, 4, 4),
        (16, 64, 1, 4),
    ] {
        for bk in [8, 16, 32] {
            let p = MmParams { bm, bn, bk, tm, tn };
            if (32..=256).contains(&p.threads()) {
                v.push(p);
            }
        }
    }
    v
}

/// Largest tile that still gives every streaming multiprocessor work.
pub fn default_mm(mk: &MatmulK, sms: usize) -> MmParams {
    let batch: usize = mk.batch.iter().map(|b| b.1).product();
    let tiles = |p: &MmParams| batch * mk.m.div_ceil(p.bm) * mk.n.div_ceil(p.bn);
    let order = [
        (128, 128, 8, 8),
        (128, 64, 8, 4),
        (64, 64, 4, 4),
        (64, 32, 4, 2),
        (32, 32, 2, 2),
    ];
    for (bm, bn, tm, tn) in order {
        let p = MmParams {
            bm,
            bn,
            bk: 16,
            tm,
            tn,
        };
        if tiles(&p) >= 2 * sms {
            return p;
        }
    }
    MmParams {
        bm: 32,
        bn: 32,
        bk: 16,
        tm: 2,
        tn: 2,
    }
}

fn var(v: Var) -> String {
    format!("v{v}")
}

fn flt(c: f32) -> String {
    if c.is_infinite() {
        return if c > 0.0 {
            "KINF".into()
        } else {
            "(-KINF)".into()
        };
    }
    if c.is_nan() {
        return "KNAN".into();
    }
    format!("{c:e}f")
}

/// Scalar CUDA expressions.
struct Gx<'a> {
    args: &'a HashMap<usize, usize>,
    /// Variable renames (matmul loop variables).
    names: HashMap<Var, String>,
    acc: Option<String>,
    /// Scalar ids other than ACC name matmul temporaries `e{id}`.
    temps: bool,
}

impl Gx<'_> {
    fn name(&self, v: Var) -> String {
        self.names.get(&v).cloned().unwrap_or_else(|| var(v))
    }

    fn lin(&self, l: &Lin) -> String {
        l.c(&|x| self.name(x))
    }

    fn buf(&self, b: &Buf) -> String {
        match b {
            Buf::Value(v) => format!("b{}", self.args[v]),
            Buf::Row(id) => format!("rb{id}"),
        }
    }

    fn e(&self, e: &E) -> String {
        match e {
            E::Load(b, l) => format!("{}[{}]", self.buf(b), self.lin(l)),
            E::Const(c) => flt(*c),
            E::Scalar(id) if *id == ACC => self.acc.clone().expect("accumulator outside a matmul"),
            E::Scalar(id) => {
                if self.temps {
                    format!("e{id}")
                } else {
                    format!("s{id}")
                }
            }
            E::Un(op, a) => {
                let a = self.e(a);
                match op {
                    Un::Neg => format!("(-{a})"),
                    Un::Sqrt => format!("sqrtf({a})"),
                    Un::Erf => format!("erff({a})"),
                    Un::Exp => format!("expf({a})"),
                    Un::Log => format!("logf({a})"),
                    Un::Abs => format!("fabsf({a})"),
                    Un::Tanh => format!("tanhf({a})"),
                    Un::Relu => format!("krelu({a})"),
                    Un::Sigmoid => format!("ksig({a})"),
                    Un::Recip => format!("(1.0f / {a})"),
                }
            }
            E::Bin(op, a, b) => {
                let (a, b) = (self.e(a), self.e(b));
                match op {
                    Bin::Add => format!("({a} + {b})"),
                    Bin::Sub => format!("({a} - {b})"),
                    Bin::Mul => format!("({a} * {b})"),
                    Bin::Div => format!("({a} / {b})"),
                    Bin::Pow => format!("powf({a}, {b})"),
                    Bin::Max => format!("kmax({a}, {b})"),
                }
            }
            E::Sel(c, a, b) => format!(
                "(({}) < {} ? {} : {})",
                self.lin(&c.lhs),
                c.bound,
                self.e(a),
                self.e(b)
            ),
        }
    }
}

fn collect_args(exprs: &[&E], outs: &[usize]) -> (Vec<Arg>, HashMap<usize, usize>) {
    let mut bufs = Vec::new();
    for e in exprs {
        e.loads(&mut bufs);
    }
    let mut args = Vec::new();
    let mut map = HashMap::new();
    for &o in outs {
        if let std::collections::hash_map::Entry::Vacant(e) = map.entry(o) {
            e.insert(args.len());
            args.push(Arg::Value(o));
        }
    }
    for b in bufs {
        if let Buf::Value(v) = b
            && !map.contains_key(&v)
        {
            map.insert(v, args.len());
            args.push(Arg::Value(v));
        }
    }
    (args, map)
}

/// The kernel signature (outputs writable, inputs read-only) and, for the
/// emulator, an entry point `KNAME_emu` taking the arguments as an array.
fn signature(threads: usize, args: &[Arg], nout: usize) -> String {
    let params: Vec<String> = (0..args.len())
        .map(|i| {
            if i < nout {
                format!("float *__restrict__ b{i}")
            } else {
                format!("const float *__restrict__ b{i}")
            }
        })
        .collect();
    format!("KGLOBAL({threads}) KNAME({})", params.join(", "))
}

fn emu_entry(args: usize) -> String {
    let call: Vec<String> = (0..args).map(|i| format!("A[{i}]")).collect();
    format!(
        "#ifndef KILN_CUDA\nstatic void KNAME_body(float **A) {{ KNAME({}); }}\nextern \"C\" void KNAME_emu(float **A, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned smem) {{ kemu_launch(KNAME_body, A, gx, gy, gz, bx, 1, smem); }}\n#endif\n",
        call.join(", ")
    )
}

/// Decodes `t` into the given variables (row-major).
fn decode(dims: &[(Var, usize)], t: &str, indent: &str) -> String {
    let mut s = String::new();
    let mut stride = 1usize;
    for &(v, d) in dims.iter().rev() {
        if d == 1 {
            let _ = writeln!(s, "{indent}const int {} = 0;", var(v));
        } else {
            let _ = writeln!(s, "{indent}const int {} = ({t} / {stride}) % {d};", var(v));
        }
        stride *= d;
    }
    s
}

fn stage_exprs(k: &LoopK) -> (Vec<&E>, Vec<usize>) {
    let mut exprs = Vec::new();
    let mut outs = Vec::new();
    for st in &k.stages {
        match st {
            Stage::Reduce { body, .. }
            | Stage::Scalar { body, .. }
            | Stage::RowBuf { body, .. } => exprs.push(body),
            Stage::Store { out, body, .. } => {
                outs.push(*out);
                exprs.push(body);
            }
        }
    }
    (exprs, outs)
}

/// Whether some stage reads a value an earlier stage of the kernel stored.
fn reads_own_store(k: &LoopK) -> Vec<bool> {
    // For each stage: does a later stage load what it stores?
    let mut out = vec![false; k.stages.len()];
    for (i, st) in k.stages.iter().enumerate() {
        if let Stage::Store { out: o, .. } = st {
            for later in &k.stages[i + 1..] {
                let body = match later {
                    Stage::Reduce { body, .. }
                    | Stage::Scalar { body, .. }
                    | Stage::RowBuf { body, .. }
                    | Stage::Store { body, .. } => body,
                };
                let mut b = Vec::new();
                body.loads(&mut b);
                if b.contains(&Buf::Value(*o)) {
                    out[i] = true;
                }
            }
        }
    }
    out
}

/// Threads per row: the row width rounded up to a power of two, at most 256.
pub fn default_tpr(width: usize) -> usize {
    width.next_power_of_two().clamp(1, 256)
}

pub fn loop_kernel(k: &LoopK, tpr: Option<usize>) -> GpuKernel {
    let (exprs, outs) = stage_exprs(k);
    let (args, map) = collect_args(&exprs, &outs);
    let nout = outs.iter().collect::<std::collections::HashSet<_>>().len();
    let rows: usize = k.outer.iter().map(|d| d.1).product();
    let (jv, width) = k.j;
    let j = var(jv);
    let gx = Gx {
        args: &map,
        names: HashMap::new(),
        acc: None,
        temps: false,
    };
    let flat = k.stages.iter().all(|s| match s {
        Stage::Scalar { .. } => true,
        Stage::Store { per_row, .. } => !per_row,
        _ => false,
    }) && !reads_own_store(k).iter().any(|&b| b);
    let mut s = String::new();
    if flat {
        let total = rows * width;
        let threads = 256usize;
        let _ = writeln!(s, "{} {{", signature(threads, &args, nout));
        let _ = writeln!(s, "  const int e = blockIdx.x * {threads} + threadIdx.x;");
        let _ = writeln!(s, "  if (e >= {total}) return;");
        let _ = writeln!(s, "  const int t = e / {width};");
        s.push_str(&decode(&k.outer, "t", "  "));
        for st in &k.stages {
            match st {
                Stage::Scalar { id, body } => {
                    let _ = writeln!(
                        s,
                        "  float s{id};\n  {{ const int {j} = 0; s{id} = {}; }}",
                        gx.e(body)
                    );
                }
                Stage::Store { out, idx, body, .. } => {
                    let _ = writeln!(
                        s,
                        "  {{ const int {j} = e % {width}; b{}[{}] = {}; }}",
                        map[out],
                        gx.lin(idx),
                        gx.e(body)
                    );
                }
                _ => unreachable!(),
            }
        }
        s.push_str("}\n");
        s.push_str(&emu_entry(args.len()));
        return GpuKernel {
            src: s,
            args,
            grid: [total.div_ceil(threads) as u32, 1, 1],
            block: threads as u32,
            smem: 0,
        };
    }
    let tpr = tpr.unwrap_or_else(|| default_tpr(width));
    let threads = tpr.max(128);
    let rpb = threads / tpr;
    let rowbufs: Vec<u32> = k
        .stages
        .iter()
        .filter_map(|s| match s {
            Stage::RowBuf { id, .. } => Some(*id),
            _ => None,
        })
        .collect();
    let red_floats = if tpr > 32 { rpb * (tpr / 32) } else { 0 };
    let smem = (rowbufs.len() * rpb * width + red_floats) * 4;
    let gsync = if tpr > 32 { "KSYNC();" } else { "KWSYNC();" };
    let _ = writeln!(s, "{} {{", signature(threads, &args, nout));
    s.push_str("  SMEM;\n");
    let _ = writeln!(
        s,
        "  const int lane = threadIdx.x % {tpr}, grp = threadIdx.x / {tpr};"
    );
    let _ = writeln!(s, "  const int row = blockIdx.x * {rpb} + grp;");
    let _ = writeln!(s, "  const bool live = row < {rows};");
    let _ = writeln!(s, "  const int t = live ? row : 0;");
    s.push_str(&decode(&k.outer, "t", "  "));
    for (i, id) in rowbufs.iter().enumerate() {
        let _ = writeln!(
            s,
            "  float *rb{id} = (float *)kiln_smem + {} + grp * {width};",
            i * rpb * width
        );
    }
    if red_floats > 0 {
        let _ = writeln!(
            s,
            "  float *kred = (float *)kiln_smem + {} + grp * {};",
            rowbufs.len() * rpb * width,
            tpr / 32
        );
    }
    let _ = writeln!(s, "  int {j};");
    let rows_store = reads_own_store(k);
    for (si, st) in k.stages.iter().enumerate() {
        match st {
            Stage::Reduce { id, op, body } => {
                let (init, comb, wred) = match op {
                    Red::Max => ("(-KINF)", "kmax(_a, {})", "kwarp_max"),
                    _ => ("0.0f", "(_a + {})", "kwarp_sum"),
                };
                let _ = writeln!(s, "  float s{id};\n  {{\n    float _a = {init};");
                let _ = writeln!(
                    s,
                    "    for ({j} = lane; {j} < {width}; {j} += {tpr}) _a = {};",
                    comb.replace("{}", &gx.e(body))
                );
                if tpr <= 32 {
                    let _ = writeln!(s, "    _a = {wred}(_a, {tpr});");
                } else {
                    let _ = writeln!(s, "    _a = {wred}(_a, 32);");
                    let _ = writeln!(s, "    if (lane % 32 == 0) kred[lane / 32] = _a;");
                    s.push_str("    KSYNC();\n");
                    let _ = writeln!(s, "    _a = kred[0];");
                    let _ = writeln!(
                        s,
                        "    for (int _w = 1; _w < {}; _w++) _a = {};",
                        tpr / 32,
                        comb.replace("{}", "kred[_w]")
                    );
                    s.push_str("    KSYNC();\n");
                }
                let fin = if *op == Red::Mean {
                    format!("_a / {}", flt(width as f32))
                } else {
                    "_a".into()
                };
                let _ = writeln!(s, "    s{id} = {fin};\n  }}");
            }
            Stage::Scalar { id, body } => {
                let _ = writeln!(s, "  {j} = 0;\n  const float s{id} = {};", gx.e(body));
            }
            Stage::RowBuf { id, body } => {
                let _ = writeln!(
                    s,
                    "  for ({j} = lane; {j} < {width}; {j} += {tpr}) rb{id}[{j}] = {};",
                    gx.e(body)
                );
                let _ = writeln!(s, "  {gsync}");
            }
            Stage::Store {
                out,
                idx,
                body,
                per_row,
            } => {
                let o = format!("b{}", map[out]);
                if *per_row {
                    let _ = writeln!(
                        s,
                        "  if (live && lane == 0) {{ {j} = 0; {o}[{}] = {}; }}",
                        gx.lin(idx),
                        gx.e(body)
                    );
                } else {
                    let _ = writeln!(
                        s,
                        "  if (live) for ({j} = lane; {j} < {width}; {j} += {tpr}) {o}[{}] = {};",
                        gx.lin(idx),
                        gx.e(body)
                    );
                }
                if rows_store[si] {
                    let _ = writeln!(s, "  {gsync}");
                }
            }
        }
    }
    s.push_str("}\n");
    s.push_str(&emu_entry(args.len()));
    GpuKernel {
        src: s,
        args,
        grid: [rows.div_ceil(rpb) as u32, 1, 1],
        block: threads as u32,
        smem: smem as u32,
    }
}

pub fn matmul_kernel(mk: &MatmulK, p: MmParams) -> GpuKernel {
    assert!(p.bm.is_multiple_of(p.tm) && p.bn.is_multiple_of(p.tn));
    let mut exprs: Vec<&E> = vec![&mk.a, &mk.epi];
    exprs.extend(mk.lets.iter().map(|l| &l.1));
    if let BOp::Expr(e) = &mk.b {
        exprs.push(e);
    }
    let (mut args, map) = collect_args(&exprs, &[mk.out]);
    let packed = match &mk.b {
        BOp::Packed { value, lin } => {
            args.push(Arg::Packed {
                value: *value,
                lin: lin.clone(),
                k: mk.k,
                n: mk.n,
                nr: mk.n,
            });
            Some(args.len() - 1)
        }
        BOp::Expr(_) => None,
    };
    let (m, n, kk) = (mk.m, mk.n, mk.k);
    let MmParams { bm, bn, bk, tm, tn } = p;
    let (sx, sy) = (bn / tn, bm / tm);
    let nt = sx * sy;
    let batch: usize = mk.batch.iter().map(|b| b.1).product();
    let names: HashMap<Var, String> = [
        (mk.vi, "_i".to_string()),
        (mk.vj, "_j".to_string()),
        (mk.vk, "_k".to_string()),
    ]
    .into();
    let gx = Gx {
        args: &map,
        names,
        acc: Some("acc".into()),
        temps: true,
    };
    let lda = bm + 1;
    let mut s = String::new();
    let _ = writeln!(s, "{} {{", signature(nt, &args, 1));
    s.push_str("  SMEM;\n");
    let _ = writeln!(
        s,
        "  float *As = (float *)kiln_smem, *Bs = As + {};",
        bk * lda
    );
    let _ = writeln!(
        s,
        "  const int tid = threadIdx.x, tx = tid % {sx}, ty = tid / {sx};"
    );
    let _ = writeln!(
        s,
        "  const int m0 = blockIdx.y * {bm}, n0 = blockIdx.x * {bn}, tb = blockIdx.z;"
    );
    s.push_str(&decode(&mk.batch, "tb", "  "));
    let _ = writeln!(s, "  float c[{tm}][{tn}];");
    let _ = writeln!(
        s,
        "  #pragma unroll\n  for (int r = 0; r < {tm}; r++)\n    #pragma unroll\n    for (int q = 0; q < {tn}; q++) c[r][q] = 0.0f;"
    );
    let _ = writeln!(s, "  for (int k0 = 0; k0 < {kk}; k0 += {bk}) {{");
    // Stage A (BM x BK, k fastest for coalescing) and B (BK x BN).
    let _ = writeln!(
        s,
        "    for (int e = tid; e < {}; e += {nt}) {{ const int kk = e % {bk}, ii = e / {bk}, _i = m0 + ii, _k = k0 + kk; As[kk * {lda} + ii] = (_i < {m} && _k < {kk}) ? {} : 0.0f; }}",
        bm * bk,
        gx.e(&mk.a)
    );
    let bload = match (&mk.b, packed) {
        (BOp::Packed { .. }, Some(a)) => format!("b{a}[_k * {n} + _j]"),
        (BOp::Expr(e), _) => gx.e(e),
        _ => unreachable!(),
    };
    let _ = writeln!(
        s,
        "    for (int e = tid; e < {}; e += {nt}) {{ const int jj = e % {bn}, kk = e / {bn}, _j = n0 + jj, _k = k0 + kk; Bs[kk * {bn} + jj] = (_j < {n} && _k < {kk}) ? {bload} : 0.0f; }}",
        bn * bk
    );
    s.push_str("    KSYNC();\n");
    let _ = writeln!(
        s,
        "    #pragma unroll\n    for (int kk = 0; kk < {bk}; kk++) {{"
    );
    let _ = writeln!(s, "      float a[{tm}], w[{tn}];");
    let _ = writeln!(
        s,
        "      #pragma unroll\n      for (int r = 0; r < {tm}; r++) a[r] = As[kk * {lda} + ty + r * {sy}];"
    );
    let _ = writeln!(
        s,
        "      #pragma unroll\n      for (int q = 0; q < {tn}; q++) w[q] = Bs[kk * {bn} + tx + q * {sx}];"
    );
    let _ = writeln!(
        s,
        "      #pragma unroll\n      for (int r = 0; r < {tm}; r++)\n        #pragma unroll\n        for (int q = 0; q < {tn}; q++) c[r][q] += a[r] * w[q];"
    );
    s.push_str("    }\n    KSYNC();\n  }\n");
    let _ = writeln!(
        s,
        "  #pragma unroll\n  for (int r = 0; r < {tm}; r++)\n    #pragma unroll\n    for (int q = 0; q < {tn}; q++) {{"
    );
    let _ = writeln!(
        s,
        "      const int _i = m0 + ty + r * {sy}, _j = n0 + tx + q * {sx};"
    );
    let _ = writeln!(s, "      if (_i < {m} && _j < {n}) {{");
    s.push_str("        const float acc = c[r][q];\n");
    for (id, e) in &mk.lets {
        let _ = writeln!(s, "        const float e{id} = {};", gx.e(e));
    }
    let _ = writeln!(
        s,
        "        b{}[{}] = {};",
        map[&mk.out],
        gx.lin(&mk.out_idx),
        gx.e(&mk.epi)
    );
    s.push_str("      }\n    }\n}\n");
    s.push_str(&emu_entry(args.len()));
    GpuKernel {
        src: s,
        args,
        grid: [n.div_ceil(bn) as u32, m.div_ceil(bm) as u32, batch as u32],
        block: nt as u32,
        smem: ((bk * lda + bk * bn) * 4) as u32,
    }
}
