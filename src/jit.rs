//! Compiles generated C with the system C compiler into a shared library,
//! cached by content hash, and loads its kernels with dlopen.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::PathBuf;

pub type KernelFn = unsafe extern "C" fn(*const *mut f32, i64, i64);

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

const RTLD_NOW: c_int = 2;

pub struct Library {
    handle: *mut c_void,
    pub path: PathBuf,
    pub compile_seconds: f64,
    pub cached: bool,
}

unsafe impl Send for Library {}
unsafe impl Sync for Library {}

/// The SIMD width to generate for: 16 floats with AVX-512, else 8.
pub fn vector_width() -> usize {
    if let Ok(v) = std::env::var("KILN_VEC")
        && let Ok(w) = v.parse()
    {
        return w;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        return 16;
    }
    8
}

pub fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("KILN_CACHE") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".cache").join("kiln")
}

pub const CFLAGS: &[&str] = &[
    "-O3",
    "-march=native",
    "-ffp-contract=fast",
    "-fPIC",
    "-shared",
    "-fno-math-errno",
];

/// Compiles `src` (or reuses a cached build) and opens it.
pub fn compile(src: &str) -> Result<Library, String> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    compile_with(&cc, CFLAGS, &["-lm"], "c", src)
}

/// Compiles `src` with `cc` and `flags` into a cached shared library.
pub fn compile_with(
    cc: &str,
    flags: &[&str],
    libs: &[&str],
    ext: &str,
    src: &str,
) -> Result<Library, String> {
    let key = format!("{:016x}", fnv(&format!("{cc} {flags:?} {libs:?}\n{src}")));
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let so = dir.join(format!("k{key}.so"));
    let t = std::time::Instant::now();
    let cached = so.exists();
    if !cached {
        // Per-process and per-thread names: concurrent builds of the same
        // source must not read each other's half-written files.
        let tag = format!("{}.{:?}", std::process::id(), std::thread::current().id())
            .replace(|c: char| !c.is_ascii_alphanumeric() && c != '.', "");
        let c = dir.join(format!("k{key}.{tag}.{ext}"));
        std::fs::write(&c, src).map_err(|e| e.to_string())?;
        let tmp = dir.join(format!("k{key}.{tag}.tmp.so"));
        let out = std::process::Command::new(cc)
            .args(flags)
            .arg("-o")
            .arg(&tmp)
            .arg(&c)
            .args(libs)
            .output()
            .map_err(|e| format!("{cc}: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "{cc} failed on {}:\n{}",
                c.display(),
                String::from_utf8_lossy(&out.stderr)
                    .lines()
                    .take(30)
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        std::fs::rename(&tmp, &so).map_err(|e| e.to_string())?;
    }
    let path = CString::new(so.to_string_lossy().as_bytes()).unwrap();
    // SAFETY: a library kiln compiled; its constructors are empty.
    let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        // SAFETY: dlerror returns a static message after a failure.
        let msg = unsafe { CStr::from_ptr(dlerror()) }
            .to_string_lossy()
            .into_owned();
        return Err(format!("dlopen {}: {msg}", so.display()));
    }
    Ok(Library {
        handle,
        path: so,
        compile_seconds: t.elapsed().as_secs_f64(),
        cached,
    })
}

impl Library {
    /// Another handle to an already-loaded library (dlopen counts
    /// references, and kiln never unloads).
    pub fn alias(l: &Library) -> Library {
        let path = CString::new(l.path.to_string_lossy().as_bytes()).unwrap();
        // SAFETY: reopening a library this process already loaded.
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        Library {
            handle,
            path: l.path.clone(),
            compile_seconds: 0.0,
            cached: true,
        }
    }

    /// The address of symbol `name`.
    pub fn symbol(&self, name: &str) -> Result<*mut c_void, String> {
        let c = CString::new(name).unwrap();
        // SAFETY: looking up a symbol in a library this process loaded.
        let p = unsafe { dlsym(self.handle, c.as_ptr()) };
        if p.is_null() {
            return Err(format!("no symbol {name}"));
        }
        Ok(p)
    }

    pub fn kernel(&self, name: &str) -> Result<KernelFn, String> {
        let c = CString::new(name).unwrap();
        // SAFETY: looking up a symbol in a library this process loaded.
        let p = unsafe { dlsym(self.handle, c.as_ptr()) };
        if p.is_null() {
            return Err(format!("no kernel {name}"));
        }
        // SAFETY: every kernel kiln generates has this signature.
        Ok(unsafe { std::mem::transmute::<*mut c_void, KernelFn>(p) })
    }
}
