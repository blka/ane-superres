# Running realesrgan-ncnn-vulkan on macOS (2026)

The ncnn/Real-ESRGAN codebase ships macOS support that has bit-rotted: as of
macOS 27, the release binary segfaults on launch, and even a clean source build
cannot see MoltenVK. Two bugs, both with small patches. If you need ncnn on a
modern Mac anyway, here they are — but first read [benchmarks.md](benchmarks.md):
on Apple Silicon, ncnn+MoltenVK is ~32× slower than CoreML/ANE for video SR.

## Build from source

```zsh
git clone https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan /tmp/resrgan-src
cd /tmp/resrgan-src
git submodule update --init --recursive --depth 1
cd src && git submodule update --init --recursive --depth 1   # ncnn lives here
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_WEBP=OFF \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j $(sysctl -n hw.ncpu)
```

Note the CMakeLists is in `src/`, not the repo root. `CMAKE_POLICY_VERSION_MINIMUM`
is needed because the vendored CMake syntax predates CMake 4.

Then apply the patches:

```zsh
cd src && git apply ../patches/ncnn/0001-*.patch
cd ncnn && git apply ../../patches/ncnn/0002-*.patch   # relative to ncnn submodule
# and rebuild
```

## Patch 1: exec path segfault (`0001`)

`filesystem_utils.h` resolves the executable directory to locate shader files:

```c
readlink("/proc/self/exe", filepath, 256);
```

`/proc/self/exe` is **Linux-only**. On macOS the call fails and leaves `filepath`
uninitialized; `strrchr` then dereferences garbage → segfault at startup. lldb
shows the crash inside `sanitize_filepath`.

Fix: `_NSGetExecutablePath` from `<mach-o/dyld.h>` under `#if __APPLE__`.

## Patch 2: MoltenVK invisible (`0002`)

With a working binary, ncnn still fails with `invalid gpu device` or
`vkCreateInstance failed -9` when only MoltenVK is installed.

Vulkan on macOS requires the loader to enumerate the ICD through the portability
mechanism: the instance must request the `VK_KHR_portability_enumeration`
extension and set `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`. ncnn
does neither, so MoltenVK never appears in its device list.

Fix in `ncnn/src/gpu.cpp`: scan the instance extension properties for
`VK_KHR_portability_enumeration` and set the flag when creating the instance.

## Device selection afterwards

With both patches, `vulkaninfo --summary` shows three ICDs: MoltenVK, Mesa
KosmicKrisp, and llvmpipe. ncnn picks by `-g <device_id>` — check the ID order
first, and avoid KosmicKrisp entirely (see benchmarks). Even on MoltenVK, expect
seconds per frame, not milliseconds.

## Why we moved on

32× slower than CoreML/ANE meant ~300 h per film instead of ~9 h. The patches
remain useful for single images or non-Apple-Silicon GPUs; for video on Apple
Silicon, CoreML wins and it isn't close.