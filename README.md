# rocdis

## POC ROCm dissassembler

### Usage:

    rocdis [options] <input file>
    -d                       Disassemble
    -e                       Extract AMDGPU object from x86-64 binary
    -h                       Show usage
    -l                       List kernels
    -s                       Print source (if available)
    -n                       Print line numbers (if available)
    -k <index>               Disassemble specified kernel
    -o <output file>         used with -e

### Dependencies
    Python3
    ROCm SDK    (for AMDGPU llvm-objdump)
    llvm        (for llvm-cxxfilt)
    g++         (for objdump)
