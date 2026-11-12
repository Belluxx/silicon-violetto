use std::io::Write;
use std::path::Path;

use crate::Res;

#[derive(Default)]
pub struct Writer {
    entries: Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
}

impl Writer {
    /// bf16 values given as their bits.
    pub fn bf16(&mut self, name: &str, shape: &[usize], data: &[u16]) {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}");
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.entries.push((name.to_string(), "BF16", shape.to_vec(), bytes));
    }

    pub fn u32(&mut self, name: &str, shape: &[usize], data: &[u32]) {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}");
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.entries.push((name.to_string(), "U32", shape.to_vec(), bytes));
    }

    pub fn save(&self, path: &Path) -> Res {
        let mut header = String::from("{");
        let mut off = 0;
        for (i, (name, dtype, shape, bytes)) in self.entries.iter().enumerate() {
            let shape: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            header += &format!(
                "{}\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}",
                if i > 0 { "," } else { "" },
                shape.join(","),
                off + bytes.len()
            );
            off += bytes.len();
        }
        header += "}";
        while header.len() % 8 != 0 {
            header.push(' ');
        }
        let write = || -> std::io::Result<()> {
            let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
            f.write_all(&(header.len() as u64).to_le_bytes())?;
            f.write_all(header.as_bytes())?;
            self.entries.iter().try_for_each(|(_, _, _, bytes)| f.write_all(bytes))?;
            f.flush()
        };
        write().map_err(|e| format!("{}: {e}", path.display()))
    }
}
