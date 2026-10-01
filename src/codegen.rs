//! C code generation. Each kernel becomes a function
//! `void k(float *const *A, long t0, long t1)` that runs tasks t0..t1 of
//! its parallel decomposition over the buffers in `A`. Vectors use GCC's
//! vector extensions (16 floats with AVX-512, 8 with AVX2); the compiler
//! allocates registers and emits FMAs, kiln decides tiling, unrolling,
//! vectorization, data layout and fusion.

use crate::fuse::{ACC, BOp, LoopK, MatmulK, Red, Stage};
use crate::graph::Graph;
use crate::ir::{Bin, Buf, Cond, E, Lin, Un, Var};
use std::collections::HashMap;
use std::fmt::Write;

/// A kernel argument: a graph value's buffer, or a constant packed into
/// matmul panels of width `nr`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Arg {
    Value(usize),
    Packed {
        value: usize,
        lin: Lin,
        k: usize,
        n: usize,
        nr: usize,
    },
    /// A constant matmul operand as fp16, transposed: element (k, j) at
    /// j·K + k (the GPU's tensor-core kernels).
    PackedHalf {
        value: usize,
        lin: Lin,
        k: usize,
        n: usize,
    },
}

pub struct KernelSrc {
    /// The function body after `void NAME`, identical for kernels that
    /// differ only in which buffers they use.
    pub body: String,
    pub args: Vec<Arg>,
    pub tasks: usize,
}

/// Matmul schedule: register tile MR x NR, MC rows and NCP panels per
/// task, and the reduction split into blocks of KC (partial sums are kept
/// in the output between blocks; the epilogue runs on the last).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Params {
    pub mr: usize,
    pub nr: usize,
    pub mc: usize,
    pub ncp: usize,
    pub kc: usize,
    /// Prefetch B this many k-rows ahead (0: none).
    pub pf: usize,
}

pub fn prelude(w: usize) -> String {
    let bcast = vec!["x"; w].join(", ");
    format!(
        r#"#include <math.h>
#include <string.h>
#include <stdint.h>
#define W {w}
typedef float vf __attribute__((vector_size(W * 4)));
typedef int32_t vi __attribute__((vector_size(W * 4)));
static inline vf vld(const float *p) {{ vf v; memcpy(&v, p, sizeof v); return v; }}
static inline void vst(float *p, vf v) {{ memcpy(p, &v, sizeof v); }}
/* Broadcast, as an initializer so it compiles to one vbroadcastss
   (`x + (vf){{0}}` would keep an add, since x + 0 is not x for x = -0). */
static inline vf vb(float x) {{ return (vf){{{bcast}}}; }}
static inline vf vsel(vi m, vf a, vf b) {{ return (vf)(((vi)a & m) | ((vi)b & ~m)); }}
static inline vf vmax(vf a, vf b) {{ return vsel(a > b, a, b); }}
static inline float hsum(vf v) {{ float s = 0; for (int l = 0; l < W; l++) s += v[l]; return s; }}
static inline float hmax(vf v) {{ float s = v[0]; for (int l = 1; l < W; l++) s = s > v[l] ? s : v[l]; return s; }}
/* exp: 2^n * e^r, |r| <= ln2/2, Cephes' degree-6 polynomial (about 2 ulp).
   Below -87.3 the result is 0 (as for -inf), never a subnormal: subnormal
   products slow x86 down a hundredfold. */
static inline vf vexp(vf x) {{
  vi under = x < vb(-87.3f);
  x = vmax(x, vb(-87.0f));
  x = vsel(x > vb(88.0f), vb(88.0f), x);
  vf u = x * vb(1.44269504088896341f) + vb(0.5f);
  vf tr = __builtin_convertvector(__builtin_convertvector(u, vi), vf);
  vf n = tr - __builtin_convertvector(-(tr > u), vf);
  vf r = x - n * vb(0.693359375f) - n * vb(-2.12194440e-4f);
  vf p = vb(1.9875691500e-4f);
  p = p * r + vb(1.3981999507e-3f);
  p = p * r + vb(8.3334519073e-3f);
  p = p * r + vb(4.1665795894e-2f);
  p = p * r + vb(1.6666665459e-1f);
  p = p * r + vb(5.0000001201e-1f);
  vf y = p * r * r + r + vb(1.0f);
  vi e = (__builtin_convertvector(n, vi) + 127) << 23;
  return vsel(under, vb(0.0f), y * (vf)e);
}}
/* erf: Abramowitz and Stegun 7.1.26 (absolute error below 1.5e-7). */
static inline vf verf(vf x) {{
  vi neg = x < vb(0.0f);
  vf a = vsel(neg, -x, x);
  vf t = vb(1.0f) / (vb(1.0f) + vb(0.3275911f) * a);
  vf p = ((((vb(1.061405429f) * t + vb(-1.453152027f)) * t + vb(1.421413741f)) * t + vb(-0.284496736f)) * t + vb(0.254829592f)) * t;
  vf y = vb(1.0f) - p * vexp(-a * a);
  return vsel(neg, -y, y);
}}
static inline vf vsqrt(vf x) {{ vf r; for (int l = 0; l < W; l++) r[l] = __builtin_sqrtf(x[l]); return r; }}
static inline vf vlog(vf x) {{ vf r; for (int l = 0; l < W; l++) r[l] = logf(x[l]); return r; }}
static inline vf vpow(vf x, vf y) {{ vf r; for (int l = 0; l < W; l++) r[l] = powf(x[l], y[l]); return r; }}
static inline vf vabs(vf x) {{ return (vf)((vi)x & (vi){{0}} + 0x7fffffff); }}
static inline vf vsig(vf x) {{ return vb(1.0f) / (vb(1.0f) + vexp(-x)); }}
static inline vf vtanh(vf x) {{ return vb(2.0f) * vsig(vb(2.0f) * x) - vb(1.0f); }}
static inline vf vrelu(vf x) {{ return vmax(x, vb(0.0f)); }}
/* Scalar forms compute exactly what one vector lane would. */
static inline float sexp(float x) {{ return vexp(vb(x))[0]; }}
static inline float serf(float x) {{ return verf(vb(x))[0]; }}
static inline float ssig(float x) {{ return vsig(vb(x))[0]; }}
static inline float stanh(float x) {{ return vtanh(vb(x))[0]; }}
static inline float smax(float a, float b) {{ return a > b ? a : b; }}
#define MIN(a, b) ((a) < (b) ? (a) : (b))
"#
    )
}

/// Emits expressions; knows buffer names, variable names and which
/// variable (if any) is vectorized.
struct Cx<'a> {
    args: &'a HashMap<usize, usize>,
    w: usize,
    vec: Option<Var>,
    /// The accumulator's name (matmul epilogues).
    acc: Option<String>,
    /// In a matmul epilogue, other scalar ids name temporaries `e{id}`
    /// (vectors in vector code, floats in per-lane code).
    temps: bool,
}

fn var(v: Var) -> String {
    format!("v{v}")
}

impl Cx<'_> {
    fn buf(&self, b: &Buf) -> String {
        match b {
            Buf::Value(v) => format!("b{}", self.args[v]),
            Buf::Row(id) => format!("rb{id}"),
        }
    }

    fn lin(&self, l: &Lin, lane: bool) -> String {
        let vec = self.vec;
        l.c(&|x| {
            if lane && Some(x) == vec {
                format!("({} + _l)", var(x))
            } else {
                var(x)
            }
        })
    }

    fn scalar_name(&self, id: u32, lane: bool) -> String {
        if id == ACC {
            let a = self.acc.clone().expect("accumulator outside a matmul");
            if lane { format!("{a}[_l]") } else { a }
        } else if self.temps {
            format!("e{id}")
        } else {
            format!("s{id}")
        }
    }

    /// Scalar C expression; with `lane`, the vector variable is `v + _l`.
    fn scalar(&self, e: &E, lane: bool) -> String {
        match e {
            E::Load(b, l) => format!("{}[{}]", self.buf(b), self.lin(l, lane)),
            E::Const(c) => flt(*c),
            E::Scalar(id) => self.scalar_name(*id, lane),
            E::Un(op, a) => {
                let a = self.scalar(a, lane);
                match op {
                    Un::Neg => format!("(-{a})"),
                    Un::Sqrt => format!("__builtin_sqrtf({a})"),
                    Un::Erf => format!("serf({a})"),
                    Un::Exp => format!("sexp({a})"),
                    Un::Log => format!("logf({a})"),
                    Un::Abs => format!("fabsf({a})"),
                    Un::Tanh => format!("stanh({a})"),
                    Un::Relu => format!("smax({a}, 0.0f)"),
                    Un::Sigmoid => format!("ssig({a})"),
                    Un::Recip => format!("(1.0f / {a})"),
                }
            }
            E::Bin(op, a, b) => {
                let (a, b) = (self.scalar(a, lane), self.scalar(b, lane));
                match op {
                    Bin::Add => format!("({a} + {b})"),
                    Bin::Sub => format!("({a} - {b})"),
                    Bin::Mul => format!("({a} * {b})"),
                    Bin::Div => format!("({a} / {b})"),
                    Bin::Pow => format!("powf({a}, {b})"),
                    Bin::Max => format!("smax({a}, {b})"),
                }
            }
            E::Sel(c, a, b) => format!(
                "(({}) < {} ? {} : {})",
                self.lin(&c.lhs, lane),
                c.bound,
                self.scalar(a, lane),
                self.scalar(b, lane)
            ),
        }
    }

    /// A vector C expression (type vf).
    fn vector(&self, e: &E) -> String {
        let (s, v) = self.vec_inner(e);
        if v { s } else { format!("vb({s})") }
    }

    /// Returns (code, is_vector).
    fn vec_inner(&self, e: &E) -> (String, bool) {
        let vv = self.vec.unwrap();
        match e {
            E::Load(b, l) => {
                let (c, nested) = l.coeff(vv);
                if !nested && c == 0 {
                    (format!("{}[{}]", self.buf(b), self.lin(l, false)), false)
                } else if !nested && c == 1 {
                    (
                        format!("vld({} + ({}))", self.buf(b), self.lin(l, false)),
                        true,
                    )
                } else {
                    (
                        format!(
                            "({{ vf _g; for (int _l = 0; _l < W; _l++) _g[_l] = {}[{}]; _g; }})",
                            self.buf(b),
                            self.lin(l, true)
                        ),
                        true,
                    )
                }
            }
            E::Const(c) => (flt(*c), false),
            E::Scalar(id) => (self.scalar_name(*id, false), *id == ACC || self.temps),
            E::Un(op, a) => {
                let (s, v) = self.vec_inner(a);
                if !v {
                    return (self.scalar(e, false), false);
                }
                let f = match op {
                    Un::Neg => return (format!("(-{s})"), true),
                    Un::Recip => return (format!("(vb(1.0f) / {s})"), true),
                    Un::Sqrt => "vsqrt",
                    Un::Erf => "verf",
                    Un::Exp => "vexp",
                    Un::Log => "vlog",
                    Un::Abs => "vabs",
                    Un::Tanh => "vtanh",
                    Un::Relu => "vrelu",
                    Un::Sigmoid => "vsig",
                };
                (format!("{f}({s})"), true)
            }
            E::Bin(op, a, b) => {
                let (sa, va) = self.vec_inner(a);
                let (sb, vb) = self.vec_inner(b);
                if !va && !vb {
                    return (self.scalar(e, false), false);
                }
                let pa = if va { sa } else { format!("vb({sa})") };
                let pb = if vb { sb } else { format!("vb({sb})") };
                (
                    match op {
                        Bin::Add => format!("({pa} + {pb})"),
                        Bin::Sub => format!("({pa} - {pb})"),
                        Bin::Mul => format!("({pa} * {pb})"),
                        Bin::Div => format!("({pa} / {pb})"),
                        Bin::Pow => format!("vpow({pa}, {pb})"),
                        Bin::Max => format!("vmax({pa}, {pb})"),
                    },
                    true,
                )
            }
            E::Sel(c, a, b) => {
                if self.uniform(c) {
                    let (sa, va) = self.vec_inner(a);
                    let (sb, vb) = self.vec_inner(b);
                    if !va && !vb {
                        return (self.scalar(e, false), false);
                    }
                    let pa = if va { sa } else { format!("vb({sa})") };
                    let pb = if vb { sb } else { format!("vb({sb})") };
                    (
                        format!(
                            "(({}) < {} ? {pa} : {pb})",
                            self.lin(&c.lhs, false),
                            c.bound
                        ),
                        true,
                    )
                } else {
                    (
                        format!(
                            "({{ vf _s; for (int _l = 0; _l < W; _l++) _s[_l] = {}; _s; }})",
                            self.scalar(e, true)
                        ),
                        true,
                    )
                }
            }
        }
    }

    /// Whether a condition has the same truth value in every lane of a
    /// vector chunk starting at a multiple of W.
    fn uniform(&self, c: &Cond) -> bool {
        let vv = self.vec.unwrap();
        let (k, nested) = c.lhs.coeff(vv);
        if nested {
            return false;
        }
        if k == 0 {
            return true;
        }
        let w = self.w as i64;
        k == 1
            && c.lhs
                .terms
                .iter()
                .all(|(a, q)| matches!(a, crate::ir::Atom::Var(x) if *x == vv) || q % w == 0)
            && (c.bound - c.lhs.c).rem_euclid(w) == 0
    }
}

fn flt(c: f32) -> String {
    if c.is_infinite() {
        return if c > 0.0 {
            "INFINITY".into()
        } else {
            "(-INFINITY)".into()
        };
    }
    if c.is_nan() {
        return "NAN".into();
    }
    let s = format!("{c:e}");
    format!("{s}f")
}

fn collect_args(exprs: &[&E], outs: &[usize], g: &Graph) -> (Vec<Arg>, HashMap<usize, usize>) {
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
    let _ = g;
    (args, map)
}

fn header(args: &[Arg], outs: usize) -> String {
    let mut s = String::from("(float *const *A, long t0, long t1) {\n");
    for i in 0..args.len() {
        if i < outs {
            let _ = writeln!(s, "  float *restrict b{i} = A[{i}];");
        } else {
            let _ = writeln!(s, "  const float *restrict b{i} = A[{i}];");
        }
    }
    s
}

/// Decodes task t into the given variables (row-major).
fn decode(dims: &[(Var, usize)], indent: &str) -> String {
    let mut s = String::new();
    let mut stride = 1usize;
    for &(v, d) in dims.iter().rev() {
        if d == 1 {
            let _ = writeln!(s, "{indent}const long {} = 0;", var(v));
        } else {
            let _ = writeln!(s, "{indent}const long {} = (t / {stride}) % {d};", var(v));
        }
        stride *= d;
    }
    s
}

pub fn loop_kernel(g: &Graph, k: &LoopK, w: usize) -> KernelSrc {
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
    let (args, map) = collect_args(&exprs, &outs, g);
    let nout = outs.iter().collect::<std::collections::HashSet<_>>().len();
    let (jv, width) = k.j;
    let cx = Cx {
        args: &map,
        w,
        vec: Some(jv),
        acc: None,
        temps: false,
    };
    let j = var(jv);
    let mut s = header(&args, nout);
    s.push_str("  for (long t = t0; t < t1; t++) {\n");
    s.push_str(&decode(&k.outer, "    "));
    let _ = writeln!(s, "    long {j} = 0;");
    for st in &k.stages {
        if let Stage::RowBuf { id, .. } = st {
            let _ = writeln!(s, "    float rb{id}[{width}] __attribute__((aligned(64)));");
        }
    }
    for st in &k.stages {
        match st {
            Stage::Reduce { id, op, body } => {
                let (init, vop, hred, sop) = match op {
                    Red::Max => ("-INFINITY", "vmax(_a, {})", "hmax", "smax(_r, {})"),
                    _ => ("0.0f", "(_a + {})", "hsum", "(_r + {})"),
                };
                let _ = writeln!(s, "    float s{id};\n    {{\n      vf _a = vb({init});");
                let _ = writeln!(
                    s,
                    "      for ({j} = 0; {j} + W <= {width}; {j} += W) _a = {};",
                    vop.replace("{}", &cx.vector(body))
                );
                let _ = writeln!(s, "      float _r = {hred}(_a);");
                let _ = writeln!(
                    s,
                    "      for (; {j} < {width}; {j}++) _r = {};",
                    sop.replace("{}", &cx.scalar(body, false))
                );
                let fin = if *op == Red::Mean {
                    format!("_r / {}", flt(width as f32))
                } else {
                    "_r".into()
                };
                let _ = writeln!(s, "      s{id} = {fin};\n    }}");
            }
            Stage::Scalar { id, body } => {
                let _ = writeln!(
                    s,
                    "    {j} = 0;\n    const float s{id} = {};",
                    cx.scalar(body, false)
                );
            }
            Stage::RowBuf { id, body } => {
                let _ = writeln!(
                    s,
                    "    for ({j} = 0; {j} + W <= {width}; {j} += W) vst(rb{id} + {j}, {});",
                    cx.vector(body)
                );
                let _ = writeln!(
                    s,
                    "    for (; {j} < {width}; {j}++) rb{id}[{j}] = {};",
                    cx.scalar(body, false)
                );
            }
            Stage::Store {
                out,
                idx,
                body,
                per_row,
            } => {
                let o = format!("b{}", map[out]);
                let (c, nested) = idx.coeff(jv);
                if *per_row {
                    let _ = writeln!(
                        s,
                        "    {j} = 0;\n    {o}[{}] = {};",
                        cx.lin(idx, false),
                        cx.scalar(body, false)
                    );
                } else if c == 1 && !nested {
                    let _ = writeln!(
                        s,
                        "    for ({j} = 0; {j} + W <= {width}; {j} += W) vst({o} + ({}), {});",
                        cx.lin(idx, false),
                        cx.vector(body)
                    );
                    let _ = writeln!(
                        s,
                        "    for (; {j} < {width}; {j}++) {o}[{}] = {};",
                        cx.lin(idx, false),
                        cx.scalar(body, false)
                    );
                } else {
                    let _ = writeln!(
                        s,
                        "    for ({j} = 0; {j} < {width}; {j}++) {o}[{}] = {};",
                        cx.lin(idx, false),
                        cx.scalar(body, false)
                    );
                }
            }
        }
    }
    s.push_str("  }\n}\n");
    KernelSrc {
        body: s,
        args,
        tasks: k.outer.iter().map(|d| d.1).product(),
    }
}

/// A reasonable schedule before tuning.
pub fn default_params(mk: &MatmulK, w: usize, threads: usize) -> Params {
    let nr = if mk.n >= 2 * w { 2 * w } else { w };
    let mr = 6;
    let np = mk.n.div_ceil(nr);
    let batch: usize = mk.batch.iter().map(|b| b.1).product();
    // Enough tasks for every thread, with whole register tiles.
    let mut mc = mk.m.div_ceil(mr) * mr;
    let mut ncp = np;
    while batch * mk.m.div_ceil(mc) * np.div_ceil(ncp) < 4 * threads && ncp > 1 {
        ncp = ncp.div_ceil(2);
    }
    while batch * mk.m.div_ceil(mc) * np.div_ceil(ncp) < 4 * threads && mc > mr {
        mc = (mc / 2).div_ceil(mr) * mr;
    }
    Params {
        mr,
        nr,
        mc,
        ncp,
        kc: mk.k,
        pf: 0,
    }
}

pub fn matmul_kernel(g: &Graph, mk: &MatmulK, p: Params, w: usize) -> KernelSrc {
    assert_eq!(p.nr % w, 0);
    assert_eq!(p.mc % p.mr, 0);
    let mut exprs: Vec<&E> = vec![&mk.a, &mk.epi];
    exprs.extend(mk.lets.iter().map(|l| &l.1));
    if let BOp::Expr(e) = &mk.b {
        exprs.push(e);
    }
    let (mut args, map) = collect_args(&exprs, &[mk.out], g);
    let packed_arg = match &mk.b {
        BOp::Packed { value, lin } => {
            args.push(Arg::Packed {
                value: *value,
                lin: lin.clone(),
                k: mk.k,
                n: mk.n,
                nr: p.nr,
            });
            Some(args.len() - 1)
        }
        BOp::Expr(_) => None,
    };
    let (m, n, kk) = (mk.m, mk.n, mk.k);
    let (mr, nr) = (p.mr, p.nr);
    let nv = nr / w;
    let np = n.div_ceil(nr);
    let nmb = m.div_ceil(p.mc);
    let nnb = np.div_ceil(p.ncp);
    let batch: usize = mk.batch.iter().map(|b| b.1).product();
    let mut s = header(&args, 1);
    let (vi, vj, vk) = (var(mk.vi), var(mk.vj), var(mk.vk));
    s.push_str("  for (long t = t0; t < t1; t++) {\n");
    let _ = writeln!(
        s,
        "    const long nb = t % {nnb}, mb = (t / {nnb}) % {nmb};"
    );
    let _ = writeln!(s, "    const long tb = t / {};", nmb * nnb);
    {
        let mut stride = 1usize;
        for &(v, d) in mk.batch.iter().rev() {
            let _ = writeln!(s, "    const long {} = (tb / {stride}) % {d};", var(v));
            stride *= d;
        }
    }
    let _ = writeln!(
        s,
        "    for (long jp = nb * {}; jp < MIN((nb + 1) * {}, {np}); jp++) {{",
        p.ncp, p.ncp
    );
    match (&mk.b, packed_arg) {
        (BOp::Packed { .. }, Some(a)) => {
            let _ = writeln!(
                s,
                "      const float *restrict bp = (const float *)A[{a}] + jp * {};",
                kk * nr
            );
        }
        (BOp::Expr(be), _) => {
            // Pack this panel: bp[k][jj] = B(batch, k, jp*NR + jj).
            let cx = Cx {
                args: &map,
                w,
                vec: Some(mk.vj),
                acc: None,
                temps: false,
            };
            let _ = writeln!(
                s,
                "      float bp[{}] __attribute__((aligned(64)));",
                kk * nr
            );
            let _ = writeln!(s, "      for (long {vk} = 0; {vk} < {kk}; {vk}++) {{");
            let (c, nested) = match be {
                E::Load(_, l) => l.coeff(mk.vj),
                _ => (2, true),
            };
            let _ = writeln!(s, "        for (long jj = 0; jj < {nr}; jj += W) {{");
            let _ = writeln!(s, "          const long {vj} = jp * {nr} + jj;");
            if c == 1 && !nested {
                let _ = writeln!(
                    s,
                    "          if ({vj} + W <= {n}) vst(bp + {vk} * {nr} + jj, {});",
                    cx.vector(be)
                );
                let _ = writeln!(
                    s,
                    "          else for (int _l = 0; _l < W; _l++) bp[{vk} * {nr} + jj + _l] = ({vj} + _l < {n}) ? {} : 0.0f;",
                    cx.scalar(be, true)
                );
            } else {
                let _ = writeln!(
                    s,
                    "          for (int _l = 0; _l < W; _l++) bp[{vk} * {nr} + jj + _l] = ({vj} + _l < {n}) ? {} : 0.0f;",
                    cx.scalar(be, true)
                );
            }
            s.push_str("        }\n      }\n");
        }
        _ => unreachable!(),
    }
    let kc = p.kc.min(kk).max(1);
    let blocked = kc < kk;
    if blocked {
        let _ = writeln!(s, "      for (long kc0 = 0; kc0 < {kk}; kc0 += {kc}) {{");
        let _ = writeln!(s, "      const long kc1 = MIN(kc0 + {kc}, {kk});");
    } else {
        let _ = writeln!(s, "      {{\n      const long kc0 = 0, kc1 = {kk};");
    }
    let _ = writeln!(
        s,
        "      for (long i0 = mb * {}; i0 < MIN((mb + 1) * {}, {m}); i0 += {mr}) {{",
        p.mc, p.mc
    );
    let o = format!("b{}", map[&mk.out]);
    let rows_list: Vec<usize> = if m % mr == 0 {
        vec![mr]
    } else {
        vec![mr, m % mr]
    };
    for (ri, &rows) in rows_list.iter().enumerate() {
        if rows_list.len() > 1 {
            let _ = writeln!(
                s,
                "        {}if (i0 + {mr} {} {m}) {{",
                if ri == 0 { "" } else { "else " },
                if ri == 0 { "<=" } else { ">" }
            );
        } else {
            s.push_str("        {\n");
        }
        // Accumulators: zero, or the partial sums of earlier K blocks.
        for r in 0..rows {
            for q in 0..nv {
                if blocked {
                    let cx = Cx {
                        args: &map,
                        w,
                        vec: Some(mk.vj),
                        acc: None,
                        temps: false,
                    };
                    let idx = cx.lin(&mk.out_idx, false);
                    let idx_l = cx.lin(&mk.out_idx, true);
                    let _ = writeln!(s, "          vf c{r}_{q} = vb(0.0f);");
                    let _ = writeln!(
                        s,
                        "          if (kc0 > 0) {{ const long {vi} = i0 + {r}; const long {vj} = jp * {nr} + {}; if ({vj} + W <= {n}) c{r}_{q} = vld({o} + ({idx})); else for (int _l = 0; _l < {n} - {vj}; _l++) c{r}_{q}[_l] = {o}[{idx_l}]; }}",
                        q * w
                    );
                } else {
                    let _ = writeln!(s, "          vf c{r}_{q} = vb(0.0f);");
                }
            }
        }
        let a_cx = |r: usize| -> String {
            let cx = Cx {
                args: &map,
                w,
                vec: None,
                acc: None,
                temps: false,
            };
            match &mk.a {
                E::Load(b, l) => {
                    let txt = l.c(&|x| {
                        if x == mk.vi {
                            format!("(i0 + {r})")
                        } else if x == mk.vk {
                            "k".into()
                        } else {
                            var(x)
                        }
                    });
                    format!("{}[{txt}]", cx.buf(b))
                }
                _ => panic!("matmul A operand must be a load"),
            }
        };
        let _ = writeln!(s, "          for (long k = kc0; k < kc1; k++) {{");
        if p.pf > 0 {
            for line in 0..(nr * 4).div_ceil(64) {
                let _ = writeln!(
                    s,
                    "            __builtin_prefetch(bp + (k + {}) * {nr} + {}, 0, 3);",
                    p.pf,
                    line * 16
                );
            }
        }
        for q in 0..nv {
            let _ = writeln!(
                s,
                "            const vf w{q} = vld(bp + k * {nr} + {});",
                q * w
            );
        }
        for r in 0..rows {
            let _ = writeln!(s, "            {{ const vf a = vb({}); ", a_cx(r));
            for q in 0..nv {
                let _ = writeln!(s, "              c{r}_{q} += a * w{q};");
            }
            s.push_str("            }\n");
        }
        s.push_str("          }\n");
        // Epilogue on the last block (the fused elementwise root, then one
        // store); partial sums otherwise.
        if blocked {
            let _ = writeln!(s, "          if (kc1 < {kk}) {{");
            for r in 0..rows {
                for q in 0..nv {
                    let cx = Cx {
                        args: &map,
                        w,
                        vec: Some(mk.vj),
                        acc: None,
                        temps: false,
                    };
                    let idx = cx.lin(&mk.out_idx, false);
                    let idx_l = cx.lin(&mk.out_idx, true);
                    let _ = writeln!(
                        s,
                        "            {{ const long {vi} = i0 + {r}; const long {vj} = jp * {nr} + {}; if ({vj} + W <= {n}) vst({o} + ({idx}), c{r}_{q}); else for (int _l = 0; _l < {n} - {vj}; _l++) {o}[{idx_l}] = c{r}_{q}[_l]; }}",
                        q * w
                    );
                }
            }
            s.push_str("          } else {\n");
        }
        for r in 0..rows {
            for q in 0..nv {
                let acc = format!("c{r}_{q}");
                let cx = Cx {
                    args: &map,
                    w,
                    vec: Some(mk.vj),
                    acc: Some(acc.clone()),
                    temps: true,
                };
                let idx = cx.lin(&mk.out_idx, false);
                let idx_l = cx.lin(&mk.out_idx, true);
                let _ = writeln!(
                    s,
                    "          {{ const long {vi} = i0 + {r}; const long {vj} = jp * {nr} + {};",
                    q * w
                );
                let _ = write!(s, "            if ({vj} + W <= {n}) {{ ");
                for (id, e) in &mk.lets {
                    let _ = write!(s, "const vf e{id} = {}; ", cx.vector(e));
                }
                let _ = writeln!(s, "vst({o} + ({idx}), {}); }}", cx.vector(&mk.epi));
                let _ = write!(
                    s,
                    "            else for (int _l = 0; _l < {n} - {vj}; _l++) {{ "
                );
                for (id, e) in &mk.lets {
                    let _ = write!(s, "const float e{id} = {}; ", cx.scalar(e, true));
                }
                let _ = writeln!(s, "{o}[{idx_l}] = {}; }}", cx.scalar(&mk.epi, true));
                s.push_str("          }\n");
            }
        }
        if blocked {
            s.push_str("          }\n");
        }
        s.push_str("        }\n");
    }
    s.push_str("      }\n      }\n");
    s.push_str("    }\n  }\n}\n");
    KernelSrc {
        body: s,
        args,
        tasks: batch * nmb * nnb,
    }
}
