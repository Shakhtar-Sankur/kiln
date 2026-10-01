//! The protocol buffers wire format, enough to read ONNX files: varints,
//! fixed-width scalars and length-delimited fields (sub-messages, strings,
//! bytes and packed repeated scalars).

pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

/// One field of a message: its number and payload.
pub enum Field<'a> {
    Varint(u64),
    Fixed64(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Reader<'a> {
        Reader { buf, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.buf.get(self.pos).ok_or("truncated varint")?;
            self.pos += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("varint too long".into())
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or("truncated field")?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// The next field, or None at the end of the message.
    pub fn next(&mut self) -> Result<Option<(u32, Field<'a>)>, String> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let num = (key >> 3) as u32;
        let f = match key & 7 {
            0 => Field::Varint(self.varint()?),
            1 => Field::Fixed64(u64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            2 => {
                let n = self.varint()? as usize;
                Field::Bytes(self.take(n)?)
            }
            5 => Field::Fixed32(u32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            w => return Err(format!("unsupported wire type {w}")),
        };
        Ok(Some((num, f)))
    }
}

impl<'a> Field<'a> {
    pub fn int(&self) -> i64 {
        match *self {
            Field::Varint(v) | Field::Fixed64(v) => v as i64,
            Field::Fixed32(v) => i64::from(v as i32),
            Field::Bytes(_) => 0,
        }
    }

    pub fn float(&self) -> f32 {
        match *self {
            Field::Fixed32(v) => f32::from_bits(v),
            Field::Fixed64(v) => f64::from_bits(v) as f32,
            _ => 0.0,
        }
    }

    pub fn bytes(&self) -> &'a [u8] {
        match *self {
            Field::Bytes(b) => b,
            _ => &[],
        }
    }

    pub fn string(&self) -> String {
        String::from_utf8_lossy(self.bytes()).into_owned()
    }

    /// A repeated varint field, packed or not.
    pub fn ints(&self, out: &mut Vec<i64>) -> Result<(), String> {
        match self {
            Field::Bytes(b) => {
                let mut r = Reader::new(b);
                while r.pos < b.len() {
                    out.push(r.varint()? as i64);
                }
            }
            f => out.push(f.int()),
        }
        Ok(())
    }

    /// A repeated float field, packed or not.
    pub fn floats(&self, out: &mut Vec<f32>) {
        match self {
            Field::Bytes(b) => out.extend(
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
            ),
            f => out.push(f.float()),
        }
    }
}
