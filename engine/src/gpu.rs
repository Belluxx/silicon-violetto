use std::ffi::{c_char, c_void};
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

// The few IOKit and CoreFoundation calls `Gpu::core_count` needs.
type CFTypeRef = *const c_void;
const IO_MAIN_PORT_DEFAULT: u32 = 0;
const CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
const CF_NUMBER_SINT64_TYPE: isize = 4;

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const c_char) -> *mut c_void;
    fn IOServiceGetMatchingService(main_port: u32, matching: *mut c_void) -> u32;
    fn IORegistryEntryCreateCFProperty(entry: u32, key: CFTypeRef, allocator: CFTypeRef, options: u32) -> CFTypeRef;
    fn IOObjectRelease(object: u32) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(allocator: CFTypeRef, s: *const c_char, encoding: u32) -> CFTypeRef;
    fn CFGetTypeID(cf: CFTypeRef) -> usize;
    fn CFNumberGetTypeID() -> usize;
    fn CFNumberGetValue(number: CFTypeRef, kind: isize, value: *mut c_void) -> u8;
    fn CFRelease(cf: CFTypeRef);
}

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl Gpu {
    pub fn new(source: &str) -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("cannot create a command queue")?;
        let opts = MTLCompileOptions::new();
        opts.setLanguageVersion(MTLLanguageVersion::Version3_1);
        opts.setMathMode(MTLMathMode::Safe);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&opts))
            .map_err(|e| format!("shader compile failed: {}", e.localizedDescription()))?;
        Ok(Self { device, queue, library })
    }

    pub fn pipeline(&self, name: &str) -> Result<Pipeline, String> {
        let f = self.library.newFunctionWithName(&NSString::from_str(name)).ok_or_else(|| format!("no kernel {name}"))?;
        self.device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| format!("pipeline {name}: {}", e.localizedDescription()))
    }

    /// A pipeline that must run threadgroups of `threads`.
    pub fn pipeline_for(&self, name: &str, threads: usize) -> Result<Pipeline, String> {
        let p = self.pipeline(name)?;
        let max = p.maxTotalThreadsPerThreadgroup();
        if max < threads {
            return Err(format!("{name} needs threadgroups of {threads} threads; this GPU allows {max}"));
        }
        Ok(p)
    }

    /// Shared-storage buffer, zero-filled.
    pub fn buffer(&self, bytes: usize) -> Result<Buffer, String> {
        self.device
            .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| format!("cannot allocate a {:.1} MB buffer", bytes as f64 / 1e6))
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// GPU cores, which Metal doesn't expose: `gpu-core-count` of the IORegistry's
    /// AGXAccelerator entry. None if the entry or the key is missing.
    pub fn core_count(&self) -> Option<usize> {
        // Safety: every object created here is released here, and the property is
        // checked to be a CFNumber before it is read.
        unsafe {
            let matching = IOServiceMatching(c"AGXAccelerator".as_ptr());
            if matching.is_null() {
                return None;
            }
            let service = IOServiceGetMatchingService(IO_MAIN_PORT_DEFAULT, matching); // consumes `matching`
            if service == 0 {
                return None;
            }
            let key = CFStringCreateWithCString(std::ptr::null(), c"gpu-core-count".as_ptr(), CF_STRING_ENCODING_UTF8);
            let value = if key.is_null() {
                std::ptr::null()
            } else {
                let value = IORegistryEntryCreateCFProperty(service, key, std::ptr::null(), 0);
                CFRelease(key);
                value
            };
            IOObjectRelease(service);
            if value.is_null() {
                return None;
            }
            let mut n: i64 = 0;
            let ok = CFGetTypeID(value) == CFNumberGetTypeID()
                && CFNumberGetValue(value, CF_NUMBER_SINT64_TYPE, (&raw mut n).cast()) != 0;
            CFRelease(value);
            (ok && n > 0).then_some(n as usize)
        }
    }

    pub fn command_buffer(&self) -> CommandBuffer {
        CommandBuffer(self.queue.commandBufferWithUnretainedReferences().expect("command buffer"))
    }

    /// Encodes with `f`, runs it to completion and returns the GPU time in seconds.
    pub fn run(&self, f: impl FnOnce(&Encoder)) -> Result<f64, String> {
        let cb = self.command_buffer();
        let e = cb.encoder();
        f(&e);
        e.end();
        cb.commit();
        cb.wait()
    }
}

pub struct CommandBuffer(Retained<ProtocolObject<dyn MTLCommandBuffer>>);

impl CommandBuffer {
    pub fn encoder(&self) -> Encoder {
        Encoder(self.0.computeCommandEncoder().expect("compute encoder"))
    }

    pub fn commit(&self) {
        self.0.commit();
    }

    /// Waits for completion; returns the GPU time in seconds.
    pub fn wait(&self) -> Result<f64, String> {
        self.0.waitUntilCompleted();
        if self.0.status() == MTLCommandBufferStatus::Error {
            let msg = self.0.error().map(|e| e.localizedDescription().to_string()).unwrap_or_default();
            return Err(format!("command buffer failed: {msg}"));
        }
        Ok(self.0.GPUEndTime() - self.0.GPUStartTime())
    }
}

/// A compute encoder. Dispatches run serially, in order.
pub struct Encoder(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>);

impl Encoder {
    pub fn pipeline(&self, p: &Pipeline) {
        self.0.setComputePipelineState(p);
    }

    pub fn buf(&self, index: usize, b: &Buffer, offset: usize) {
        // Safety: the engine owns every buffer and keeps it alive past the command
        // buffers that use it; offsets are computed in range.
        unsafe { self.0.setBuffer_offset_atIndex(Some(b), offset, index) }
    }

    pub fn bytes<T: Copy>(&self, index: usize, v: &T) {
        // Safety: Metal copies the bytes during the call; T is a #[repr(C)] mirror of the
        // kernel's argument.
        unsafe { self.0.setBytes_length_atIndex(NonNull::from(v).cast::<c_void>(), size_of::<T>(), index) }
    }

    /// Threadgroup memory for the kernel's `[[threadgroup(index)]]` argument (a multiple of
    /// 16 bytes).
    pub fn tg_mem(&self, index: usize, bytes: usize) {
        // Safety: a plain length; the pipeline's limit is checked by Metal's validation.
        unsafe { self.0.setThreadgroupMemoryLength_atIndex(bytes.next_multiple_of(16), index) }
    }

    pub fn dispatch(&self, groups: (usize, usize, usize), threads: usize) {
        self.0.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: groups.0, height: groups.1, depth: groups.2 },
            MTLSize { width: threads, height: 1, depth: 1 },
        );
    }

    pub fn end(self) {
        self.0.endEncoding();
    }
}

pub fn contents<T>(b: &Buffer) -> *mut T {
    b.contents().as_ptr().cast()
}

pub fn read<T: Copy>(b: &Buffer, offset: usize, len: usize) -> Vec<T> {
    assert!((offset + len) * size_of::<T>() <= b.length());
    // Safety: in range (checked above) of a shared buffer the GPU isn't writing.
    unsafe { std::slice::from_raw_parts(contents::<T>(b).add(offset), len).to_vec() }
}

pub fn write<T: Copy>(b: &Buffer, offset: usize, data: &[T]) {
    assert!((offset + data.len()) * size_of::<T>() <= b.length());
    // Safety: in range (checked above) of a shared buffer the GPU isn't reading.
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), contents::<T>(b).add(offset), data.len()) }
}
