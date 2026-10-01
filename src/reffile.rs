//! Reference files from scripts/export_models.py: named input and output
//! tensors computed by PyTorch.

use crate::tensor::Tensor;
use std::path::Path;

pub struct Reference {
    pub inputs: Vec<(String, Tensor)>,
    pub outputs: Vec<(String, Tensor)>,
}

pub fn load(path: &Path) -> Result<Reference, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !b.starts_with(b"KILNREF1") {
        return Err("not a kiln reference file".into());
    }
    let hl = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
    let head = std::str::from_utf8(&b[12..12 + hl]).map_err(|e| e.to_string())?;
    let mut pos = 12 + hl;
    let mut r = Reference {
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    // A tiny parser for the fixed header layout: {"inputs": [...], "outputs": [...]}
    for (kind, list) in [("inputs", 0), ("outputs", 1)] {
        let start = head.find(&format!("\"{kind}\"")).ok_or("bad header")?;
        let open = head[start..].find('[').unwrap() + start;
        let mut depth = 0;
        let mut close = open;
        for (i, c) in head[open..].char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        close = open + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        for item in head[open + 1..close]
            .split('}')
            .filter(|s| s.contains("name"))
        {
            let field = |k: &str| -> &str {
                let i = item.find(&format!("\"{k}\"")).unwrap() + k.len() + 2;
                item[i..].trim_start_matches([':', ' '])
            };
            let name = field("name")
                .trim_start_matches('"')
                .split('"')
                .next()
                .unwrap()
                .to_string();
            let dtype = field("dtype")
                .trim_start_matches('"')
                .split('"')
                .next()
                .unwrap()
                .to_string();
            let sh = field("shape");
            let sh = &sh[1..sh.find(']').unwrap()];
            let shape: Vec<usize> = sh
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse().unwrap())
                .collect();
            let n: usize = shape.iter().product();
            let t = match dtype.as_str() {
                "f32" => {
                    let v = b[pos..pos + 4 * n]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect();
                    pos += 4 * n;
                    Tensor::f32(shape, v)
                }
                _ => {
                    let v = b[pos..pos + 8 * n]
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|c| i64::from_le_bytes(*c))
                        .collect();
                    pos += 8 * n;
                    Tensor::i64(shape, v)
                }
            };
            if list == 0 {
                r.inputs.push((name, t));
            } else {
                r.outputs.push((name, t));
            }
        }
    }
    Ok(r)
}
