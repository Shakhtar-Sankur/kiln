//! kiln's graph IR: values (named, typed, shaped, possibly constant) and
//! the nodes that produce them, in topological order.

use crate::tensor::{DType, Tensor};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum Attr {
    Float(f32),
    Int(i64),
    Str(String),
    Tensor(Tensor),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
}

#[derive(Clone, Debug)]
pub struct Value {
    pub name: String,
    pub dtype: Option<DType>,
    pub shape: Option<Vec<usize>>,
    /// Known at compile time (an initializer, or folded).
    pub konst: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: String,
    pub name: String,
    /// Value ids; None for an omitted optional input.
    pub inputs: Vec<Option<usize>>,
    pub outputs: Vec<usize>,
    pub attrs: HashMap<String, Attr>,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub values: Vec<Value>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

impl Node {
    pub fn attr_int(&self, k: &str, default: i64) -> i64 {
        match self.attrs.get(k) {
            Some(Attr::Int(v)) => *v,
            _ => default,
        }
    }

    pub fn attr_float(&self, k: &str, default: f32) -> f32 {
        match self.attrs.get(k) {
            Some(Attr::Float(v)) => *v,
            Some(Attr::Int(v)) => *v as f32,
            _ => default,
        }
    }

    pub fn attr_ints(&self, k: &str) -> Option<Vec<i64>> {
        match self.attrs.get(k) {
            Some(Attr::Ints(v)) => Some(v.clone()),
            Some(Attr::Int(v)) => Some(vec![*v]),
            _ => None,
        }
    }

    /// The i-th input's value id (panics if absent).
    pub fn input(&self, i: usize) -> usize {
        self.inputs[i].unwrap_or_else(|| panic!("{} ({}): missing input {i}", self.name, self.op))
    }

    pub fn has_input(&self, i: usize) -> bool {
        self.inputs.get(i).is_some_and(Option::is_some)
    }
}

impl Graph {
    /// Adds a value and returns its id.
    pub fn add_value(&mut self, name: String) -> usize {
        self.values.push(Value {
            name,
            dtype: None,
            shape: None,
            konst: None,
        });
        self.values.len() - 1
    }

    pub fn shape(&self, v: usize) -> &[usize] {
        self.values[v]
            .shape
            .as_deref()
            .unwrap_or_else(|| panic!("value {} has no known shape", self.values[v].name))
    }

    pub fn konst(&self, v: usize) -> Option<&Tensor> {
        self.values[v].konst.as_ref()
    }

    /// Number of nodes reading each value (graph outputs count once).
    pub fn use_counts(&self) -> Vec<usize> {
        let mut c = vec![0; self.values.len()];
        for n in &self.nodes {
            for i in n.inputs.iter().flatten() {
                c[*i] += 1;
            }
        }
        for &o in &self.outputs {
            c[o] += 1;
        }
        c
    }

    /// The node producing each value, if any.
    pub fn producers(&self) -> Vec<Option<usize>> {
        let mut p = vec![None; self.values.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            for &o in &n.outputs {
                p[o] = Some(i);
            }
        }
        p
    }

    /// Counts of each op type, sorted by name.
    pub fn op_histogram(&self) -> Vec<(String, usize)> {
        let mut h: HashMap<&str, usize> = HashMap::new();
        for n in &self.nodes {
            *h.entry(n.op.as_str()).or_default() += 1;
        }
        let mut v: Vec<(String, usize)> = h.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        v.sort();
        v
    }
}
