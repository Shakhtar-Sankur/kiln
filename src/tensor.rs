//! Dense tensors: float32, int64 and bool, row-major.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    I64,
    Bool,
}

impl DType {
    /// The ONNX TensorProto data type code.
    pub fn from_onnx(code: i64) -> Result<DType, String> {
        match code {
            1 => Ok(DType::F32),
            7 | 6 => Ok(DType::I64),
            9 => Ok(DType::Bool),
            c => Err(format!("unsupported tensor type {c}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::I64 => "i64",
            DType::Bool => "bool",
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Data {
    F32(Vec<f32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

#[derive(Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Data,
}

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major strides of `shape`.
pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

impl Tensor {
    pub fn f32(shape: Vec<usize>, v: Vec<f32>) -> Tensor {
        assert_eq!(numel(&shape), v.len(), "shape {shape:?}");
        Tensor {
            shape,
            data: Data::F32(v),
        }
    }

    pub fn i64(shape: Vec<usize>, v: Vec<i64>) -> Tensor {
        assert_eq!(numel(&shape), v.len(), "shape {shape:?}");
        Tensor {
            shape,
            data: Data::I64(v),
        }
    }

    pub fn bool(shape: Vec<usize>, v: Vec<bool>) -> Tensor {
        assert_eq!(numel(&shape), v.len(), "shape {shape:?}");
        Tensor {
            shape,
            data: Data::Bool(v),
        }
    }

    pub fn scalar_f32(v: f32) -> Tensor {
        Tensor::f32(vec![], vec![v])
    }

    pub fn dtype(&self) -> DType {
        match self.data {
            Data::F32(_) => DType::F32,
            Data::I64(_) => DType::I64,
            Data::Bool(_) => DType::Bool,
        }
    }

    pub fn len(&self) -> usize {
        numel(&self.shape)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(&self) -> &[f32] {
        match &self.data {
            Data::F32(v) => v,
            _ => panic!("expected f32, got {}", self.dtype().name()),
        }
    }

    pub fn as_i64(&self) -> &[i64] {
        match &self.data {
            Data::I64(v) => v,
            _ => panic!("expected i64, got {}", self.dtype().name()),
        }
    }

    pub fn as_bool(&self) -> &[bool] {
        match &self.data {
            Data::Bool(v) => v,
            _ => panic!("expected bool, got {}", self.dtype().name()),
        }
    }

    /// Values as i64 (for shapes, axes and indices), from any type.
    pub fn to_i64(&self) -> Vec<i64> {
        match &self.data {
            Data::F32(v) => v.iter().map(|&x| x as i64).collect(),
            Data::I64(v) => v.clone(),
            Data::Bool(v) => v.iter().map(|&x| x as i64).collect(),
        }
    }

    pub fn reshaped(&self, shape: Vec<usize>) -> Tensor {
        assert_eq!(numel(&shape), self.len());
        Tensor {
            shape,
            data: self.data.clone(),
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}>{:?}", self.dtype().name(), self.shape)
    }
}
