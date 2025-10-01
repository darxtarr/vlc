# NVIDIA GPU Vulkan Support in WSL2 (2025) - Research Report

**Date**: 2025-10-01
**Status**: NVIDIA Vulkan in WSL2 is NON-FUNCTIONAL
**Recommendation**: Use CUDA or native platform alternatives

---

## Executive Summary

**BAD NEWS**: Native NVIDIA Vulkan support in WSL2 is **NOT WORKING** as of 2025, and there's no clear timeline for when it will be fixed. Your RTX 4080 GPU is accessible via CUDA but **not via Vulkan**.

**GOOD NEWS**: You have multiple viable alternatives for GPU compute workloads in Rust.

---

## Current Status: Why NVIDIA Vulkan Fails in WSL2

### The Root Cause

Your system configuration is actually **correct**. The problem is with the NVIDIA WSL2 driver implementation itself:

1. **NVIDIA WSL2 drivers are CUDA-only stubs** - They're specifically designed for compute workloads via CUDA, not graphics APIs like Vulkan
2. **The Vulkan ICD exists but is non-functional** - `/usr/share/vulkan/icd.d/nvidia_icd.json` points to `libGLX_nvidia.so.0`, which contains Vulkan ICD symbols (`vk_icdGetInstanceProcAddr`) but they return NULL when called
3. **It's a stub implementation** - The NVIDIA WSL2 driver includes the Vulkan ICD interface for compatibility but doesn't actually implement the Vulkan API

### What We Found on the System

```bash
# The ICD file exists and points to the right library
/usr/share/vulkan/icd.d/nvidia_icd.json → libGLX_nvidia.so.0

# The library exists and has Vulkan symbols
/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.580.65.06
  - Contains: vk_icdGetInstanceProcAddr
  - Contains: vk_icdGetPhysicalDeviceProcAddr
  - Contains: vk_icdNegotiateLoaderICDInterfaceVersion

# But when called, it fails to return vkCreateInstance
ERROR: Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr'
```

This is **by design** - NVIDIA's WSL2 driver doesn't implement Vulkan compute functionality.

---

## Why This Happens: WSL2 GPU Architecture

### Microsoft's Approach: D3D12 Translation Layer

WSL2 uses GPU-PV (GPU paravirtualization), not direct hardware passthrough:

1. **CUDA Path**: Windows NVIDIA driver → WSL2 CUDA stub (`libcuda.so`) → Works ✅
2. **Vulkan Path (Microsoft's "Dozen" driver)**: Should use `dzn` (Vulkan-on-D3D12) → **Missing/broken** ❌
3. **NVIDIA Direct Vulkan**: Native Linux NVIDIA Vulkan driver → **Not included in WSL2 driver** ❌

### The Missing Piece: Mesa "Dozen" (dzn)

Microsoft's solution for Vulkan in WSL2 is the **Dozen (dzn)** driver - a Mesa implementation that translates Vulkan to DirectX 12. However:

- Users report missing `dzn_icd.x86_64.json` and `libvulkan_dzn.so` files
- Even when present, dzn doesn't work reliably with NVIDIA GPUs
- It's marked as "non-conformant, testing use only"
- Multiple reports from 2024-2025 confirm it's still broken for NVIDIA on Ubuntu 24.04/25.04

---

## Confirmed Non-Solutions

These approaches were researched - **none work for NVIDIA Vulkan compute in WSL2**:

### ❌ Installing Mesa PPAs
```bash
# This was suggested in forums but doesn't help NVIDIA
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update && sudo apt upgrade
```
**Result**: Only helps Intel/AMD GPUs, doesn't fix NVIDIA Vulkan

### ❌ Setting VK_ICD_FILENAMES
```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```
**Result**: Same error - the ICD exists but doesn't implement Vulkan

### ❌ Installing Linux NVIDIA Drivers
**NVIDIA explicitly states**: Do NOT install Linux NVIDIA drivers in WSL2 - it will break CUDA

### ❌ Symlink Workarounds
Creating symlinks to different library paths doesn't help when the library itself lacks Vulkan implementation

---

## VIABLE ALTERNATIVES (Recommended Solutions)

### **Option 1: Use Rust-CUDA (BEST for WSL2 + NVIDIA)**

**GitHub**: https://github.com/Rust-GPU/Rust-CUDA

**Why this is perfect for the use case**:
- ✅ Direct CUDA support (which **does work** in WSL2)
- ✅ Pure Rust - no shader languages needed
- ✅ Full access to CUDA features: shared memory, streams, unified memory
- ✅ Better performance than Vulkan compute for well-written kernels
- ✅ RTX 4080 is fully accessible via nvidia-smi/CUDA

**Install**:
```bash
# CUDA toolkit (if not installed)
# Download from NVIDIA's WSL-Ubuntu page

# Add to Cargo.toml
[dependencies]
cust = "0.3"  # CUDA runtime
```

**Example**:
```rust
use cust::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    rustacuda::init(CudaFlags::empty())?;

    let device = Device::get_device(0)?;
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
        device,
    )?;

    // Your GPU compute kernel here
    // Compiles to PTX and runs on RTX 4080

    Ok(())
}
```

### **Option 2: Run Native Windows + wgpu with DX12 Backend**

Since running on Windows with WSL2, run Rust code **natively on Windows** instead:

**Why this works**:
- ✅ wgpu supports DirectX 12 backend on Windows
- ✅ DX12 has full NVIDIA driver support
- ✅ No WSL2 limitations
- ✅ Same Rust code, just change target platform

**Setup**:
```powershell
# In Windows PowerShell (not WSL2)
# Install Rust for Windows
rustup-init.exe

# Existing wgpu code should work
$env:WGPU_BACKEND="dx12"
cargo run --release
```

### **Option 3: CubeCL (Multi-backend Rust GPU)**

**Link**: https://github.com/tracel-ai/cubecl

**Why consider this**:
- ✅ Single Rust codebase
- ✅ Targets CUDA, ROCm, and wgpu
- ✅ Can switch backends at runtime
- ✅ Higher-level API than raw CUDA

**Example**:
```rust
use cubecl::prelude::*;

// Same kernel code works on CUDA, wgpu, or ROCm
#[cube(launch)]
fn my_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    // Your compute logic
}

// Select runtime at compile time
#[cfg(target_os = "linux")]
let runtime = CudaRuntime::new(0);  // Use CUDA in WSL2
```

### **Option 4: Native Linux (Dual Boot or Separate Machine)**

If Vulkan compute is absolutely required:
- ✅ Native Ubuntu + NVIDIA drivers = full Vulkan support
- ✅ Can dual-boot or use separate machine
- ✅ Jetson Nano/Orin has native Linux with proper drivers
- ❌ Requires leaving WSL2 environment or additional hardware

---

## Package Installation (If You Want to Try Mesa Dozen Anyway)

Despite not working reliably for NVIDIA, here's how to attempt the Mesa Dozen setup:

```bash
# Update to latest Mesa (includes dzn driver)
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt upgrade mesa-vulkan-drivers

# Check if dzn ICD appeared
ls /usr/share/vulkan/icd.d/dzn*

# Test (will likely show dzn but still fall back to software)
vulkaninfo --summary
```

**Expected result**: You'll see "Microsoft Direct3D12 (NVIDIA GeForce RTX 4080)" but with warnings about non-conformance, or it still won't work.

---

## Official Microsoft/NVIDIA Documentation Status

### Microsoft WSL Documentation
- ✅ Confirms CUDA support is official and production-ready
- ⚠️ Mentions Vulkan support exists but provides no NVIDIA-specific guidance
- ❌ No official statement on NVIDIA Vulkan compute timelines

### NVIDIA WSL Documentation
- ✅ Comprehensive CUDA on WSL2 guides
- ✅ Explicitly states: "Do not install Linux NVIDIA drivers in WSL2"
- ❌ **Zero mention of Vulkan support** - this is telling

### Community Consensus (2024-2025)
- GitHub issue #7790 (WSL): Still open, no resolution
- GitHub issue #1254 (WSLg): Still open, dzn issues ongoing
- NVIDIA forums: Multiple threads, no working solutions for NVIDIA+Vulkan+WSL2

---

## VLC Project Recommendations

**For the vlc project (M2 GPU compute workloads)**:

### **Immediate Solution: Rust-CUDA**
1. Install CUDA Toolkit in WSL2 (if not installed):
   ```bash
   # Check if already installed
   nvcc --version

   # If not, follow: https://developer.nvidia.com/cuda-downloads
   # Select: Linux → x86_64 → WSL-Ubuntu → 2.0 → deb (local)
   ```

2. Modify project to use `cust` crate instead of wgpu for compute
3. Keep existing compute kernel logic, just wrap it in CUDA instead of WGSL

### **Why This is Better Than Waiting for Vulkan**:
- CUDA is **actually faster** for compute workloads than Vulkan on NVIDIA hardware
- Get access to NVIDIA-specific optimizations
- It's production-ready in WSL2 right now
- RTX 4080 will perform better with CUDA than it would with Vulkan

### **Migration Path**:
```rust
// Before (wgpu)
let device = adapter.request_device(...).await?;
let shader = device.create_shader_module(wgsl! { ... });

// After (Rust-CUDA)
let device = Device::get_device(0)?;
let module = Module::from_ptx(PTX_CODE, &[])?;
let kernel = module.get_function("my_kernel")?;
```

---

## Timeline Expectations

Based on research:

- **Short term (2025)**: NVIDIA Vulkan in WSL2 will **remain broken**
- **Medium term (2026)**: Microsoft *might* fix dzn for NVIDIA, but no promises
- **Long term**: Uncertain - Microsoft may focus on DirectX 12 instead

**Don't wait for this to be fixed** - use CUDA or switch to native Windows/Linux.

---

## Additional Resources

### Working Examples
- **Rust-CUDA examples**: https://github.com/Rust-GPU/Rust-CUDA/tree/master/examples
- **CUDA on WSL2 guide**: https://docs.nvidia.com/cuda/wsl-user-guide/
- **wgpu on native Windows**: https://sotrh.github.io/learn-wgpu/

### Issue Tracking
- WSL Vulkan support: https://github.com/microsoft/WSL/issues/7790
- WSLg Vulkan: https://github.com/microsoft/wslg/issues/1254
- NVIDIA forums: https://forums.developer.nvidia.com/t/enabling-nvidia-support-for-vulkan-on-ubuntu-22-04-through-wsl2/244176

---

## Summary Table

| Solution | Works in WSL2? | Performance | Ease of Migration | Recommended? |
|----------|----------------|-------------|-------------------|--------------|
| **Rust-CUDA** | ✅ Yes | ⭐⭐⭐⭐⭐ Excellent | 🔨 Moderate | **✅ YES** |
| **wgpu + native Windows DX12** | ✅ Yes | ⭐⭐⭐⭐ Very Good | 🔨 Easy | **✅ YES** |
| **CubeCL** | ✅ Yes | ⭐⭐⭐⭐ Very Good | 🔨 Moderate | ✅ Consider |
| **wgpu + NVIDIA Vulkan (WSL2)** | ❌ **NO** | N/A | N/A | **❌ NO** |
| **Mesa Dozen (dzn)** | ⚠️ Partial | ⭐⭐ Poor | 🔨 Hard | ❌ Not worth it |
| **Native Linux dual-boot** | ✅ Yes | ⭐⭐⭐⭐⭐ Excellent | 🔨 Hard | ⚠️ If essential |
| **Jetson Nano/Orin** | ✅ Yes | ⭐⭐⭐⭐ Very Good | 🔨 Hard | ✅ For deployment |

---

## Conclusion

NVIDIA Vulkan compute in WSL2 is fundamentally broken and unlikely to be fixed soon. The recommended path forward is:

1. **Accept software renderer validation** for correctness testing (already proven at 1.21x speedup)
2. **Focus on M3 implementation** (maintenance + retrieval) to complete the project
3. **Consider CUDA backend** for production GPU acceleration in WSL2
4. **Target Jetson Nano/Orin** for native Linux deployment with full GPU support

The VLC architecture is sound and validated. GPU acceleration can be added later via CUDA or on native hardware.

---

*Generated: 2025-10-01*
