//! The CUDA driver API and NVRTC, loaded with dlopen (no build-time
//! dependency on CUDA): just the entry points kiln uses.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

fn open(names: &[String]) -> Option<*mut c_void> {
    for n in names {
        let c = CString::new(n.as_str()).ok()?;
        // SAFETY: loading a system library by name.
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if !h.is_null() {
            return Some(h);
        }
    }
    None
}

fn sym<T: Copy>(h: *mut c_void, name: &str) -> Result<T, String> {
    let c = CString::new(name).unwrap();
    // SAFETY: looking up a symbol of a loaded library.
    let p = unsafe { dlsym(h, c.as_ptr()) };
    if p.is_null() {
        return Err(format!("missing symbol {name}"));
    }
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    // SAFETY: T is the function pointer type of `name`.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
}

pub type CuResult = c_int;
pub type DevPtr = u64;
type Ctx = *mut c_void;
type Module = *mut c_void;
/// A loaded kernel.
#[derive(Clone, Copy, Debug)]
pub struct Function(*mut c_void);
pub type Stream = *mut c_void;
type Event = *mut c_void;
type Graph = *mut c_void;
/// An instantiated CUDA graph.
#[derive(Clone, Copy, Debug)]
pub struct GraphExec(*mut c_void);

#[allow(non_snake_case)]
pub struct Cuda {
    cuGetErrorString: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    cuMemAlloc: unsafe extern "C" fn(*mut DevPtr, usize) -> CuResult,
    cuMemAllocHost: unsafe extern "C" fn(*mut *mut c_void, usize) -> CuResult,
    cuMemcpyHtoDAsync: unsafe extern "C" fn(DevPtr, *const c_void, usize, Stream) -> CuResult,
    cuMemcpyDtoHAsync: unsafe extern "C" fn(*mut c_void, DevPtr, usize, Stream) -> CuResult,
    cuMemsetD32Async: unsafe extern "C" fn(DevPtr, c_uint, usize, Stream) -> CuResult,
    cuModuleLoadData: unsafe extern "C" fn(*mut Module, *const c_void) -> CuResult,
    cuModuleGetFunction: unsafe extern "C" fn(*mut *mut c_void, Module, *const c_char) -> CuResult,
    cuFuncSetAttribute: unsafe extern "C" fn(*mut c_void, c_int, c_int) -> CuResult,
    cuFuncGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void) -> CuResult,
    cuLaunchKernel: unsafe extern "C" fn(
        *mut c_void,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        Stream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    cuStreamCreate: unsafe extern "C" fn(*mut Stream, c_uint) -> CuResult,
    cuStreamSynchronize: unsafe extern "C" fn(Stream) -> CuResult,
    cuEventCreate: unsafe extern "C" fn(*mut Event, c_uint) -> CuResult,
    cuEventRecord: unsafe extern "C" fn(Event, Stream) -> CuResult,
    cuEventSynchronize: unsafe extern "C" fn(Event) -> CuResult,
    cuEventElapsedTime: unsafe extern "C" fn(*mut f32, Event, Event) -> CuResult,
    cuStreamBeginCapture: unsafe extern "C" fn(Stream, c_int) -> CuResult,
    cuStreamEndCapture: unsafe extern "C" fn(Stream, *mut Graph) -> CuResult,
    cuGraphInstantiateWithFlags: unsafe extern "C" fn(*mut *mut c_void, Graph, u64) -> CuResult,
    cuGraphLaunch: unsafe extern "C" fn(*mut c_void, Stream) -> CuResult,
    pub name: String,
    /// Compute capability (major, minor).
    pub cc: (i32, i32),
    pub sms: usize,
    pub stream: Stream,
}

// The context is bound per thread; kiln drives one device from one thread
// at a time.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

const ATTR_CC_MAJOR: c_int = 75;
const ATTR_CC_MINOR: c_int = 76;
const ATTR_SMS: c_int = 16;
const FUNC_MAX_DYN_SMEM: c_int = 8;
pub const FUNC_NUM_REGS: c_int = 4;

impl Cuda {
    /// The first CUDA device, or why there is none.
    #[allow(non_snake_case)]
    pub fn open() -> Result<Cuda, String> {
        let h = open(&["libcuda.so.1".into(), "libcuda.so".into()])
            .ok_or("no CUDA driver (libcuda.so.1)")?;
        let cuInit: unsafe extern "C" fn(c_uint) -> CuResult = sym(h, "cuInit")?;
        let cuDeviceGet: unsafe extern "C" fn(*mut c_int, c_int) -> CuResult =
            sym(h, "cuDeviceGet")?;
        let cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult =
            sym(h, "cuDeviceGetAttribute")?;
        let cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> CuResult =
            sym(h, "cuDeviceGetName")?;
        let cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut Ctx, c_int) -> CuResult =
            sym(h, "cuDevicePrimaryCtxRetain")?;
        let cuCtxSetCurrent: unsafe extern "C" fn(Ctx) -> CuResult = sym(h, "cuCtxSetCurrent")?;
        let mut c = Cuda {
            cuGetErrorString: sym(h, "cuGetErrorString")?,
            cuMemAlloc: sym(h, "cuMemAlloc_v2")?,
            cuMemAllocHost: sym(h, "cuMemAllocHost_v2")?,
            cuMemcpyHtoDAsync: sym(h, "cuMemcpyHtoDAsync_v2")?,
            cuMemcpyDtoHAsync: sym(h, "cuMemcpyDtoHAsync_v2")?,
            cuMemsetD32Async: sym(h, "cuMemsetD32Async")?,
            cuModuleLoadData: sym(h, "cuModuleLoadData")?,
            cuModuleGetFunction: sym(h, "cuModuleGetFunction")?,
            cuFuncSetAttribute: sym(h, "cuFuncSetAttribute")?,
            cuFuncGetAttribute: sym(h, "cuFuncGetAttribute")?,
            cuLaunchKernel: sym(h, "cuLaunchKernel")?,
            cuStreamCreate: sym(h, "cuStreamCreate")?,
            cuStreamSynchronize: sym(h, "cuStreamSynchronize")?,
            cuEventCreate: sym(h, "cuEventCreate")?,
            cuEventRecord: sym(h, "cuEventRecord")?,
            cuEventSynchronize: sym(h, "cuEventSynchronize")?,
            cuEventElapsedTime: sym(h, "cuEventElapsedTime")?,
            cuStreamBeginCapture: sym(h, "cuStreamBeginCapture_v2")?,
            cuStreamEndCapture: sym(h, "cuStreamEndCapture")?,
            cuGraphInstantiateWithFlags: sym(h, "cuGraphInstantiateWithFlags")?,
            cuGraphLaunch: sym(h, "cuGraphLaunch")?,
            name: String::new(),
            cc: (0, 0),
            sms: 0,
            stream: std::ptr::null_mut(),
        };
        let mut dev = 0;
        let mut ctx = std::ptr::null_mut();
        // SAFETY: plain driver API calls with valid out-pointers.
        unsafe {
            c.check(cuInit(0), "cuInit")?;
            c.check(cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
            let (mut a, mut b, mut s) = (0, 0, 0);
            c.check(cuDeviceGetAttribute(&mut a, ATTR_CC_MAJOR, dev), "attr")?;
            c.check(cuDeviceGetAttribute(&mut b, ATTR_CC_MINOR, dev), "attr")?;
            c.check(cuDeviceGetAttribute(&mut s, ATTR_SMS, dev), "attr")?;
            c.cc = (a, b);
            c.sms = s as usize;
            let mut buf = [0 as c_char; 256];
            c.check(cuDeviceGetName(buf.as_mut_ptr(), 256, dev), "name")?;
            c.name = CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned();
            c.check(
                cuDevicePrimaryCtxRetain(&mut ctx, dev),
                "cuDevicePrimaryCtxRetain",
            )?;
            c.check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            let mut st = std::ptr::null_mut();
            c.check((c.cuStreamCreate)(&mut st, 1), "cuStreamCreate")?;
            c.stream = st;
        }
        Ok(c)
    }

    pub fn check(&self, r: CuResult, what: &str) -> Result<(), String> {
        if r == 0 {
            return Ok(());
        }
        let mut p: *const c_char = std::ptr::null();
        // SAFETY: cuGetErrorString returns a static string.
        unsafe { (self.cuGetErrorString)(r, &mut p) };
        let msg = if p.is_null() {
            String::new()
        } else {
            // SAFETY: as above.
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        };
        Err(format!("{what}: CUDA error {r} {msg}"))
    }

    pub fn alloc(&self, bytes: usize) -> Result<DevPtr, String> {
        let mut p = 0;
        // SAFETY: valid out-pointer.
        self.check(
            unsafe { (self.cuMemAlloc)(&mut p, bytes.max(4)) },
            "cuMemAlloc",
        )?;
        Ok(p)
    }

    /// Page-locked host memory (for fast, asynchronous copies).
    pub fn alloc_host(&self, floats: usize) -> Result<*mut f32, String> {
        let mut p = std::ptr::null_mut();
        // SAFETY: valid out-pointer.
        self.check(
            unsafe { (self.cuMemAllocHost)(&mut p, floats.max(1) * 4) },
            "cuMemAllocHost",
        )?;
        Ok(p as *mut f32)
    }

    pub fn upload(&self, dst: DevPtr, src: &[f32]) -> Result<(), String> {
        // SAFETY: src is valid for its length; dst is a device allocation.
        unsafe {
            self.check(
                (self.cuMemcpyHtoDAsync)(
                    dst,
                    src.as_ptr() as *const c_void,
                    src.len() * 4,
                    self.stream,
                ),
                "upload",
            )?;
        }
        self.sync()
    }

    /// Asynchronous copies on kiln's stream (host memory must stay valid
    /// until the next sync).
    pub fn h2d_async(&self, dst: DevPtr, src: *const f32, n: usize) -> Result<(), String> {
        // SAFETY: caller guarantees src is valid for n floats until sync.
        self.check(
            unsafe { (self.cuMemcpyHtoDAsync)(dst, src as *const c_void, n * 4, self.stream) },
            "h2d",
        )
    }

    pub fn d2h_async(&self, dst: *mut f32, src: DevPtr, n: usize) -> Result<(), String> {
        // SAFETY: caller guarantees dst is valid for n floats until sync.
        self.check(
            unsafe { (self.cuMemcpyDtoHAsync)(dst as *mut c_void, src, n * 4, self.stream) },
            "d2h",
        )
    }

    pub fn fill_nan(&self, dst: DevPtr, n: usize) -> Result<(), String> {
        // SAFETY: dst is a device allocation of n floats.
        self.check(
            unsafe { (self.cuMemsetD32Async)(dst, 0x7fc0_0000, n, self.stream) },
            "memset",
        )
    }

    pub fn sync(&self) -> Result<(), String> {
        // SAFETY: kiln's stream.
        self.check(unsafe { (self.cuStreamSynchronize)(self.stream) }, "sync")
    }

    /// Loads a CUBIN and looks up its functions.
    pub fn load(&self, image: &[u8], names: &[String]) -> Result<Vec<Function>, String> {
        let mut m = std::ptr::null_mut();
        // SAFETY: image is a CUBIN produced by NVRTC.
        self.check(
            unsafe { (self.cuModuleLoadData)(&mut m, image.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        let mut out = Vec::new();
        for n in names {
            let c = CString::new(n.as_str()).unwrap();
            let mut f = std::ptr::null_mut();
            // SAFETY: m is a loaded module.
            self.check(
                unsafe { (self.cuModuleGetFunction)(&mut f, m, c.as_ptr()) },
                &format!("function {n}"),
            )?;
            out.push(Function(f));
        }
        Ok(out)
    }

    pub fn allow_smem(&self, f: Function, bytes: u32) -> Result<(), String> {
        // SAFETY: f is a loaded function.
        self.check(
            unsafe { (self.cuFuncSetAttribute)(f.0, FUNC_MAX_DYN_SMEM, bytes as c_int) },
            "shared memory limit",
        )
    }

    pub fn func_attr(&self, f: Function, attr: c_int) -> i32 {
        let mut v = 0;
        // SAFETY: f is a loaded function.
        unsafe { (self.cuFuncGetAttribute)(&mut v, attr, f.0) };
        v
    }

    pub fn launch(
        &self,
        f: Function,
        grid: [u32; 3],
        block: u32,
        smem: u32,
        args: &[DevPtr],
    ) -> Result<(), String> {
        let mut params: Vec<*mut c_void> = args
            .iter()
            .map(|a| a as *const DevPtr as *mut c_void)
            .collect();
        // SAFETY: params point at device pointers matching the kernel's
        // parameters, alive for the call.
        self.check(
            unsafe {
                (self.cuLaunchKernel)(
                    f.0,
                    grid[0],
                    grid[1],
                    grid[2],
                    block,
                    1,
                    1,
                    smem,
                    self.stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    pub fn begin_capture(&self) -> Result<(), String> {
        // Relaxed mode: kiln's thread makes no unsafe calls while capturing.
        // SAFETY: kiln's stream.
        self.check(
            unsafe { (self.cuStreamBeginCapture)(self.stream, 2) },
            "cuStreamBeginCapture",
        )
    }

    pub fn end_capture(&self) -> Result<GraphExec, String> {
        let mut g = std::ptr::null_mut();
        let mut e = std::ptr::null_mut();
        // SAFETY: a capture is in progress on kiln's stream.
        unsafe {
            self.check(
                (self.cuStreamEndCapture)(self.stream, &mut g),
                "cuStreamEndCapture",
            )?;
            self.check(
                (self.cuGraphInstantiateWithFlags)(&mut e, g, 0),
                "cuGraphInstantiate",
            )?;
        }
        Ok(GraphExec(e))
    }

    pub fn graph_launch(&self, e: GraphExec) -> Result<(), String> {
        // SAFETY: an instantiated graph.
        self.check(
            unsafe { (self.cuGraphLaunch)(e.0, self.stream) },
            "cuGraphLaunch",
        )
    }

    /// Milliseconds of GPU time `f` enqueues (it must only enqueue work).
    pub fn time(&self, f: &mut dyn FnMut() -> Result<(), String>) -> Result<f32, String> {
        let (mut a, mut b) = (std::ptr::null_mut(), std::ptr::null_mut());
        let mut ms = 0f32;
        // SAFETY: events on kiln's stream.
        unsafe {
            self.check((self.cuEventCreate)(&mut a, 0), "event")?;
            self.check((self.cuEventCreate)(&mut b, 0), "event")?;
            self.check((self.cuEventRecord)(a, self.stream), "record")?;
            f()?;
            self.check((self.cuEventRecord)(b, self.stream), "record")?;
            self.check((self.cuEventSynchronize)(b), "event sync")?;
            self.check((self.cuEventElapsedTime)(&mut ms, a, b), "elapsed")?;
        }
        Ok(ms)
    }
}

/// NVRTC: CUDA C++ to CUBIN.
#[allow(non_snake_case)]
pub struct Nvrtc {
    nvrtcCreateProgram: unsafe extern "C" fn(
        *mut *mut c_void,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int,
    nvrtcCompileProgram: unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int,
    nvrtcGetProgramLogSize: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
    nvrtcGetProgramLog: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
    nvrtcGetCUBINSize: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
    nvrtcGetCUBIN: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
    nvrtcDestroyProgram: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    pub version: (i32, i32),
}

unsafe impl Send for Nvrtc {}
unsafe impl Sync for Nvrtc {}

impl Nvrtc {
    /// NVRTC from $KILN_NVRTC, the loader path, or the CUDA toolkit.
    #[allow(non_snake_case)]
    pub fn open() -> Result<Nvrtc, String> {
        let mut names: Vec<String> = Vec::new();
        if let Ok(p) = std::env::var("KILN_NVRTC") {
            names.push(p);
        }
        for n in [
            "libnvrtc.so",
            "libnvrtc.so.12",
            "libnvrtc.so.13",
            "/usr/local/cuda/lib64/libnvrtc.so",
        ] {
            names.push(n.into());
        }
        let h = open(&names).ok_or("no NVRTC (set KILN_NVRTC to libnvrtc.so)")?;
        let nvrtcVersion: unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int =
            sym(h, "nvrtcVersion")?;
        let (mut a, mut b) = (0, 0);
        // SAFETY: valid out-pointers.
        unsafe { nvrtcVersion(&mut a, &mut b) };
        Ok(Nvrtc {
            nvrtcCreateProgram: sym(h, "nvrtcCreateProgram")?,
            nvrtcCompileProgram: sym(h, "nvrtcCompileProgram")?,
            nvrtcGetProgramLogSize: sym(h, "nvrtcGetProgramLogSize")?,
            nvrtcGetProgramLog: sym(h, "nvrtcGetProgramLog")?,
            nvrtcGetCUBINSize: sym(h, "nvrtcGetCUBINSize")?,
            nvrtcGetCUBIN: sym(h, "nvrtcGetCUBIN")?,
            nvrtcDestroyProgram: sym(h, "nvrtcDestroyProgram")?,
            version: (a, b),
        })
    }

    /// Compiles `src` for sm_{arch}; returns the CUBIN and the compiler log.
    pub fn compile(
        &self,
        src: &str,
        arch: u32,
        verbose: bool,
    ) -> Result<(Vec<u8>, String), String> {
        let csrc = CString::new(src).unwrap();
        let name = CString::new("kiln.cu").unwrap();
        let mut prog = std::ptr::null_mut();
        let mut opts = vec![
            format!("--gpu-architecture=sm_{arch}"),
            "--std=c++17".into(),
            "--fmad=true".into(),
        ];
        if verbose {
            opts.push("--ptxas-options=-v".into());
        }
        let copts: Vec<CString> = opts
            .iter()
            .map(|o| CString::new(o.as_str()).unwrap())
            .collect();
        let ptrs: Vec<*const c_char> = copts.iter().map(|c| c.as_ptr()).collect();
        // SAFETY: NVRTC calls with valid strings; the program is destroyed below.
        unsafe {
            let r = (self.nvrtcCreateProgram)(
                &mut prog,
                csrc.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            );
            if r != 0 {
                return Err(format!("nvrtcCreateProgram: {r}"));
            }
            let r = (self.nvrtcCompileProgram)(prog, ptrs.len() as c_int, ptrs.as_ptr());
            let mut n = 0;
            (self.nvrtcGetProgramLogSize)(prog, &mut n);
            let mut log = vec![0u8; n.max(1)];
            (self.nvrtcGetProgramLog)(prog, log.as_mut_ptr() as *mut c_char);
            let log = String::from_utf8_lossy(&log)
                .trim_end_matches('\0')
                .to_string();
            if r != 0 {
                (self.nvrtcDestroyProgram)(&mut prog);
                return Err(format!("NVRTC failed:\n{log}"));
            }
            let mut n = 0;
            (self.nvrtcGetCUBINSize)(prog, &mut n);
            let mut bin = vec![0u8; n];
            (self.nvrtcGetCUBIN)(prog, bin.as_mut_ptr() as *mut c_char);
            (self.nvrtcDestroyProgram)(&mut prog);
            Ok((bin, log))
        }
    }
}
