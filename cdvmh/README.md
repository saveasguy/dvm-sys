# Building Guide

## DVM System Build

The CDVMH converted is a part of the DVM system. See [DVM](http://dvm-system.org) for details.

## Standalone Build

You can use CMake to configure and build CDVMH converter outside of the DVM system build tree.
A supported LLVM version has to be previously installed.

```bash
cd path/to/cdvmh-clang
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<path> -DLLVM_VERSION=x.x.x -DLLVM_DIR=<path/to/cmake/llvm>
cmake --build . -- -j <jobs_number>
cmake --build . --target install/fast
```

The mentioned CMake variables are optional:

- `LLVM_VERSION` specifies the LLVM version to use,
- `LLVM_DIR` specifies a path to the LLVM configuration files. (actually <build_path>\lib\cmake\llvm)

## SAPFOR Build

It is also possible to build the converter as a part of the SAPFOR project.
In this case, necessary LLVM libraries may be built as a part of SAPFOR.
See [SAPFOR](https://github.com/dvm-system) for details.
