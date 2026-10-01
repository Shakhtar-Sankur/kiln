//! The kernel IR. Index expressions are affine forms over loop variables
//! with floor division and remainder atoms, kept in a canonical form whose
//! simplifier knows each variable's range: reshapes and transposes then
//! compile to plain strided indexing, and the vectorizer can read off
//! whether an access is contiguous, broadcast or gathered. Value
//! expressions compute f32 results from loads, constants and operators.

use std::collections::HashMap;
use std::fmt::Write;

pub type Var = u32;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Atom {
    Var(Var),
    Div(Box<Lin>, i64),
    Mod(Box<Lin>, i64),
}

/// `c + Σ coeff·atom`, terms sorted and merged.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Lin {
    pub c: i64,
    pub terms: Vec<(Atom, i64)>,
}

/// Exclusive upper bounds of the loop variables (all start at 0).
pub type Ranges = HashMap<Var, i64>;

impl Lin {
    pub fn konst(c: i64) -> Lin {
        Lin {
            c,
            terms: Vec::new(),
        }
    }

    pub fn var(v: Var) -> Lin {
        Lin {
            c: 0,
            terms: vec![(Atom::Var(v), 1)],
        }
    }

    fn atom(a: Atom) -> Lin {
        Lin {
            c: 0,
            terms: vec![(a, 1)],
        }
    }

    fn normalize(mut self) -> Lin {
        self.terms.sort_by(|a, b| a.0.cmp(&b.0));
        let mut out: Vec<(Atom, i64)> = Vec::with_capacity(self.terms.len());
        for (a, k) in self.terms {
            match out.last_mut() {
                Some((b, kk)) if *b == a => *kk += k,
                _ => out.push((a, k)),
            }
        }
        out.retain(|t| t.1 != 0);
        self.terms = out;
        self
    }

    pub fn add(&self, o: &Lin) -> Lin {
        let mut t = self.terms.clone();
        t.extend(o.terms.iter().cloned());
        Lin {
            c: self.c + o.c,
            terms: t,
        }
        .normalize()
    }

    pub fn add_const(&self, c: i64) -> Lin {
        let mut l = self.clone();
        l.c += c;
        l
    }

    pub fn scale(&self, k: i64) -> Lin {
        if k == 0 {
            return Lin::konst(0);
        }
        Lin {
            c: self.c * k,
            terms: self.terms.iter().map(|(a, q)| (a.clone(), q * k)).collect(),
        }
    }

    pub fn is_const(&self) -> Option<i64> {
        self.terms.is_empty().then_some(self.c)
    }

    /// Smallest and largest values over the ranges.
    pub fn bounds(&self, r: &Ranges) -> (i64, i64) {
        let (mut lo, mut hi) = (self.c, self.c);
        for (a, k) in &self.terms {
            let (alo, ahi) = a.bounds(r);
            if *k >= 0 {
                lo += k * alo;
                hi += k * ahi;
            } else {
                lo += k * ahi;
                hi += k * alo;
            }
        }
        (lo, hi)
    }

    /// Splits into (multiple-of-k part divided by k, remainder part), with
    /// the constant split so the remainder's constant lies in [0, k).
    fn split(&self, k: i64) -> (Lin, Lin) {
        let (mut q, mut r) = (
            Lin::konst(self.c.div_euclid(k)),
            Lin::konst(self.c.rem_euclid(k)),
        );
        for (a, c) in &self.terms {
            if c % k == 0 {
                q.terms.push((a.clone(), c / k));
            } else {
                r.terms.push((a.clone(), *c));
            }
        }
        (q.normalize(), r.normalize())
    }

    /// floor(self / k), simplified when the remainder part provably lies in [0, k).
    pub fn div(&self, k: i64, r: &Ranges) -> Lin {
        if k == 1 {
            return self.clone();
        }
        let (q, rem) = self.split(k);
        let (lo, hi) = rem.bounds(r);
        if lo >= 0 && hi < k {
            return q;
        }
        // Nested division: (x / a) / b = x / (a·b) for non-negative x.
        if self.terms.len() == 1
            && self.c == 0
            && self.terms[0].1 == 1
            && let Atom::Div(inner, a) = &self.terms[0].0
        {
            return inner.div(a * k, r);
        }
        if lo >= 0 {
            // q + floor(rem / k)
            return q.add(&Lin::atom(Atom::Div(Box::new(rem), k)));
        }
        Lin::atom(Atom::Div(Box::new(self.clone()), k))
    }

    /// self mod k (non-negative self), simplified likewise.
    pub fn rem(&self, k: i64, r: &Ranges) -> Lin {
        if k == 1 {
            return Lin::konst(0);
        }
        let (_, rem) = self.split(k);
        let (lo, hi) = rem.bounds(r);
        if lo >= 0 && hi < k {
            return rem;
        }
        Lin::atom(Atom::Mod(Box::new(rem), k))
    }

    /// The coefficient of `v` as a direct term, and whether `v` appears
    /// inside a division or remainder atom.
    pub fn coeff(&self, v: Var) -> (i64, bool) {
        let mut c = 0;
        let mut nested = false;
        for (a, k) in &self.terms {
            match a {
                Atom::Var(x) if *x == v => c += k,
                Atom::Var(_) => {}
                Atom::Div(l, _) | Atom::Mod(l, _) => nested |= l.mentions(v),
            }
        }
        (c, nested)
    }

    pub fn mentions(&self, v: Var) -> bool {
        self.terms.iter().any(|(a, _)| match a {
            Atom::Var(x) => *x == v,
            Atom::Div(l, _) | Atom::Mod(l, _) => l.mentions(v),
        })
    }

    /// Replaces variable `v` by `by`.
    pub fn subst(&self, v: Var, by: &Lin, r: &Ranges) -> Lin {
        let mut out = Lin::konst(self.c);
        for (a, k) in &self.terms {
            let t = match a {
                Atom::Var(x) if *x == v => by.clone(),
                Atom::Var(_) => Lin::atom(a.clone()),
                Atom::Div(l, d) => l.subst(v, by, r).div(*d, r),
                Atom::Mod(l, d) => l.subst(v, by, r).rem(*d, r),
            };
            out = out.add(&t.scale(*k));
        }
        out
    }

    pub fn eval(&self, env: &HashMap<Var, i64>) -> i64 {
        let mut s = self.c;
        for (a, k) in &self.terms {
            s += k * match a {
                Atom::Var(x) => env[x],
                Atom::Div(l, d) => l.eval(env).div_euclid(*d),
                Atom::Mod(l, d) => l.eval(env).rem_euclid(*d),
            };
        }
        s
    }

    /// C source, with each variable printed by `name`.
    pub fn c(&self, name: &dyn Fn(Var) -> String) -> String {
        let mut s = String::new();
        for (a, k) in &self.terms {
            let at = match a {
                Atom::Var(x) => name(*x),
                Atom::Div(l, d) => format!("(({})/{d})", l.c(name)),
                Atom::Mod(l, d) => format!("(({})%{d})", l.c(name)),
            };
            if !s.is_empty() {
                s.push_str(if *k < 0 { " - " } else { " + " });
            } else if *k < 0 {
                s.push('-');
            }
            if k.abs() == 1 {
                s.push_str(&at);
            } else {
                let _ = write!(s, "{}*{at}", k.abs());
            }
        }
        if s.is_empty() {
            return self.c.to_string();
        }
        if self.c != 0 {
            let _ = write!(
                s,
                " {} {}",
                if self.c < 0 { '-' } else { '+' },
                self.c.abs()
            );
        }
        s
    }
}

impl Atom {
    fn bounds(&self, r: &Ranges) -> (i64, i64) {
        match self {
            Atom::Var(v) => (0, r.get(v).copied().unwrap_or(1) - 1),
            Atom::Div(l, k) => {
                let (lo, hi) = l.bounds(r);
                (lo.div_euclid(*k), hi.div_euclid(*k))
            }
            Atom::Mod(l, k) => {
                let (lo, hi) = l.bounds(r);
                if lo >= 0 && hi < *k {
                    (lo, hi)
                } else {
                    (0, k - 1)
                }
            }
        }
    }
}

/// Row-major linear index of `idx` in `shape`.
pub fn linearize(idx: &[Lin], shape: &[usize]) -> Lin {
    let mut l = Lin::konst(0);
    let mut stride = 1i64;
    for d in (0..shape.len()).rev() {
        l = l.add(&idx[d].scale(stride));
        stride *= shape[d] as i64;
    }
    l
}

/// Multi-index of linear index `l` in `shape`.
pub fn delinearize(l: &Lin, shape: &[usize], r: &Ranges) -> Vec<Lin> {
    let mut out = vec![Lin::konst(0); shape.len()];
    let mut stride = 1i64;
    for d in (0..shape.len()).rev() {
        out[d] = if d == 0 {
            l.div(stride, r)
        } else {
            l.div(stride, r).rem(shape[d] as i64, r)
        };
        stride *= shape[d] as i64;
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Un {
    Neg,
    Sqrt,
    Erf,
    Exp,
    Log,
    Abs,
    Tanh,
    Relu,
    Sigmoid,
    Recip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bin {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
}

/// What a load reads: a graph value (activation or constant) or a
/// row-local buffer of a row kernel.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Buf {
    Value(usize),
    Row(u32),
}

/// `lhs < bound`, a condition on indices (from Concat).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Cond {
    pub lhs: Lin,
    pub bound: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum E {
    Load(Buf, Lin),
    Const(f32),
    /// A row scalar of a row kernel, or the matmul accumulator.
    Scalar(u32),
    Un(Un, Box<E>),
    Bin(Bin, Box<E>, Box<E>),
    Sel(Cond, Box<E>, Box<E>),
}

impl E {
    /// Number of operations (for cost decisions).
    pub fn ops(&self) -> usize {
        match self {
            E::Load(..) | E::Const(_) | E::Scalar(_) => 0,
            E::Un(_, a) => 1 + a.ops(),
            E::Bin(_, a, b) => 1 + a.ops() + b.ops(),
            E::Sel(_, a, b) => 1 + a.ops() + b.ops(),
        }
    }

    /// Every buffer this expression reads.
    pub fn loads(&self, out: &mut Vec<Buf>) {
        match self {
            E::Load(b, _) => {
                if !out.contains(b) {
                    out.push(b.clone());
                }
            }
            E::Un(_, a) => a.loads(out),
            E::Bin(_, a, b) | E::Sel(_, a, b) => {
                a.loads(out);
                b.loads(out);
            }
            _ => {}
        }
    }

    pub fn is_pure_load(&self) -> bool {
        match self {
            E::Load(..) => true,
            E::Sel(_, a, b) => a.is_pure_load() && b.is_pure_load(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshape_and_transpose_indices_simplify_to_strides() {
        // [B*T, H*D] read as [B, T, H, D]: index (b, t, h, d).
        let (b, t, h, d) = (0, 1, 2, 3);
        let r: Ranges = [(b, 2), (t, 5), (h, 4), (d, 8)].into();
        let lin = linearize(
            &[Lin::var(b), Lin::var(t), Lin::var(h), Lin::var(d)],
            &[2, 5, 4, 8],
        );
        let idx = delinearize(&lin, &[10, 32], &r);
        assert_eq!(idx[0], Lin::var(b).scale(5).add(&Lin::var(t)));
        assert_eq!(idx[1], Lin::var(h).scale(8).add(&Lin::var(d)));
        // A real division survives when the remainder can overflow.
        let x: Ranges = [(0, 100)].into();
        let q = Lin::var(0).div(7, &x);
        assert!(matches!(q.terms[0].0, Atom::Div(..)));
        let env: HashMap<Var, i64> = [(0, 50)].into();
        assert_eq!(q.eval(&env), 7);
        assert_eq!(Lin::var(0).rem(7, &x).eval(&env), 1);
    }
}
