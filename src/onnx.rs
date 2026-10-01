//! ONNX import: ModelProto bytes to a kiln Graph.

use crate::graph::{Attr, Graph, Node};
use crate::proto::Reader;
use crate::tensor::{DType, Tensor, numel};
use std::collections::HashMap;
use std::path::Path;

pub fn load(path: &Path) -> Result<Graph, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_model(&bytes)
}

pub fn parse_model(bytes: &[u8]) -> Result<Graph, String> {
    let mut r = Reader::new(bytes);
    while let Some((num, f)) = r.field()? {
        if num == 7 {
            return parse_graph(f.bytes());
        }
    }
    Err("no graph in model".into())
}

struct ValueInfo {
    name: String,
    dtype: Option<DType>,
    shape: Option<Vec<usize>>,
}

fn parse_graph(b: &[u8]) -> Result<Graph, String> {
    let mut nodes = Vec::new();
    let mut inits = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut infos = Vec::new();
    let mut r = Reader::new(b);
    while let Some((num, f)) = r.field()? {
        match num {
            1 => nodes.push(parse_node(f.bytes())?),
            5 => inits.push(parse_tensor(f.bytes())?),
            11 => inputs.push(parse_value_info(f.bytes())?),
            12 => outputs.push(parse_value_info(f.bytes())?),
            13 => infos.push(parse_value_info(f.bytes())?),
            _ => {}
        }
    }
    let mut g = Graph::default();
    let mut ids: HashMap<String, usize> = HashMap::new();
    let mut id = |g: &mut Graph, name: &str| -> usize {
        if let Some(&i) = ids.get(name) {
            return i;
        }
        let i = g.add_value(name.to_string());
        ids.insert(name.to_string(), i);
        i
    };
    for (name, t) in inits {
        let v = id(&mut g, &name);
        g.values[v].dtype = Some(t.dtype());
        g.values[v].shape = Some(t.shape.clone());
        g.values[v].konst = Some(t);
    }
    for vi in inputs.iter().chain(&outputs).chain(&infos) {
        let v = id(&mut g, &vi.name);
        let val = &mut g.values[v];
        if val.konst.is_none() {
            val.dtype = val.dtype.or(vi.dtype);
            if val.shape.is_none() {
                val.shape.clone_from(&vi.shape);
            }
        }
    }
    for vi in &inputs {
        let v = id(&mut g, &vi.name);
        if g.values[v].konst.is_none() {
            g.inputs.push(v);
        }
    }
    for vi in &outputs {
        let v = id(&mut g, &vi.name);
        g.outputs.push(v);
    }
    for n in nodes {
        let ins = n
            .inputs
            .iter()
            .map(|s| (!s.is_empty()).then(|| id(&mut g, s)))
            .collect();
        let outs = n.outputs.iter().map(|s| id(&mut g, s)).collect();
        g.nodes.push(Node {
            op: n.op,
            name: n.name,
            inputs: ins,
            outputs: outs,
            attrs: n.attrs,
        });
    }
    Ok(g)
}

struct RawNode {
    op: String,
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: HashMap<String, Attr>,
}

fn parse_node(b: &[u8]) -> Result<RawNode, String> {
    let mut n = RawNode {
        op: String::new(),
        name: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: HashMap::new(),
    };
    let mut domain = String::new();
    let mut r = Reader::new(b);
    while let Some((num, f)) = r.field()? {
        match num {
            1 => n.inputs.push(f.string()),
            2 => n.outputs.push(f.string()),
            3 => n.name = f.string(),
            4 => n.op = f.string(),
            5 => {
                let (k, v) = parse_attr(f.bytes())?;
                if let Some(v) = v {
                    n.attrs.insert(k, v);
                }
            }
            7 => domain = f.string(),
            _ => {}
        }
    }
    if !domain.is_empty() && domain != "ai.onnx" {
        return Err(format!(
            "{}: operator domain {domain:?} is not supported",
            n.op
        ));
    }
    Ok(n)
}

fn parse_attr(b: &[u8]) -> Result<(String, Option<Attr>), String> {
    let mut name = String::new();
    let (mut fl, mut i, mut s, mut t) = (None, None, None, None);
    let (mut floats, mut ints) = (Vec::new(), Vec::new());
    let mut ty = 0;
    let mut r = Reader::new(b);
    while let Some((num, f)) = r.field()? {
        match num {
            1 => name = f.string(),
            2 => fl = Some(f.float()),
            3 => i = Some(f.int()),
            4 => s = Some(f.string()),
            5 => t = Some(parse_tensor(f.bytes())?.1),
            7 => f.floats(&mut floats),
            8 => f.ints(&mut ints)?,
            20 => ty = f.int(),
            _ => {}
        }
    }
    let a = match ty {
        1 => fl.map(Attr::Float),
        2 => i.map(Attr::Int),
        3 => s.map(Attr::Str),
        4 => t.map(Attr::Tensor),
        6 => Some(Attr::Floats(floats)),
        7 => Some(Attr::Ints(ints)),
        _ => None,
    };
    Ok((name, a))
}

/// A TensorProto: its name and value.
fn parse_tensor(b: &[u8]) -> Result<(String, Tensor), String> {
    let mut dims = Vec::new();
    let mut ty = 0;
    let mut name = String::new();
    let mut raw: Option<&[u8]> = None;
    let (mut floats, mut ints, mut int32s) = (Vec::new(), Vec::new(), Vec::new());
    let mut external = false;
    let mut r = Reader::new(b);
    while let Some((num, f)) = r.field()? {
        match num {
            1 => f.ints(&mut dims)?,
            2 => ty = f.int(),
            4 => f.floats(&mut floats),
            5 => f.ints(&mut int32s)?,
            7 => f.ints(&mut ints)?,
            8 => name = f.string(),
            9 => raw = Some(f.bytes()),
            14 => external = f.int() == 1,
            _ => {}
        }
    }
    if external {
        return Err(format!("{name}: external tensor data is not supported"));
    }
    let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    let n = numel(&shape);
    let t = match DType::from_onnx(ty)? {
        DType::F32 => {
            let v = match raw {
                Some(b) => b
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect(),
                None => floats,
            };
            Tensor::f32(shape, v)
        }
        DType::I64 => {
            let v = match (raw, ty) {
                (Some(b), 7) => b
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| i64::from_le_bytes(*c))
                    .collect(),
                (Some(b), _) => b
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i64::from(i32::from_le_bytes(*c)))
                    .collect(),
                (None, 7) => ints,
                (None, _) => int32s,
            };
            Tensor::i64(shape, v)
        }
        DType::Bool => {
            let v = match raw {
                Some(b) => b.iter().map(|&x| x != 0).collect(),
                None => int32s.iter().map(|&x| x != 0).collect(),
            };
            Tensor::bool(shape, v)
        }
    };
    if t.len() != n {
        return Err(format!(
            "{name}: {} values for shape {:?}",
            t.len(),
            t.shape
        ));
    }
    Ok((name, t))
}

fn parse_value_info(b: &[u8]) -> Result<ValueInfo, String> {
    let mut vi = ValueInfo {
        name: String::new(),
        dtype: None,
        shape: None,
    };
    let mut r = Reader::new(b);
    while let Some((num, f)) = r.field()? {
        match num {
            1 => vi.name = f.string(),
            2 => {
                // TypeProto.tensor_type
                let mut tr = Reader::new(f.bytes());
                while let Some((tn, tf)) = tr.field()? {
                    if tn != 1 {
                        continue;
                    }
                    let mut rr = Reader::new(tf.bytes());
                    while let Some((n2, f2)) = rr.field()? {
                        match n2 {
                            1 => vi.dtype = DType::from_onnx(f2.int()).ok(),
                            2 => vi.shape = parse_shape(f2.bytes())?,
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(vi)
}

/// A TensorShapeProto with every dimension known, else None.
fn parse_shape(b: &[u8]) -> Result<Option<Vec<usize>>, String> {
    let mut dims = Vec::new();
    let mut known = true;
    let mut r = Reader::new(b);
    while let Some((num, f)) = r.field()? {
        if num != 1 {
            continue;
        }
        let mut d = None;
        let mut dr = Reader::new(f.bytes());
        while let Some((n2, f2)) = dr.field()? {
            if n2 == 1 {
                d = Some(f2.int() as usize);
            }
        }
        match d {
            Some(v) => dims.push(v),
            None => known = false,
        }
    }
    Ok(known.then_some(dims))
}
