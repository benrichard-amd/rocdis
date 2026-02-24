# rocdis

## POC ROCm dissassembler

### Usage
```
   rocdis.py [options] <input file>
   -d                       Disassemble
   -e                       Extract AMDGPU object from x86-64 binary
   -h                       Show usage
   -l                       List kernels
   -s                       Print source (if available)
   -n                       Print line numbers (if available)
   -k <index>               Disassemble specified kernel
   -o <output file>         used with -e
```

### Dependencies
```
    Python3
    ROCm SDK      (for AMDGPU llvm-objdump)
    clang-tools   (for llvm-cxxfilt)
    g++           (for objdump)
```

The tool will look for the path to `llvm-cxxfilt` in the following order:
1. `LLVM_CXXFILT` environment variable
2. System path
3. `/opt/rocm/llvm/bin/llvm-cxxfilt`

### Samples

```
make -C samples
```
#### List kernels
```
rocdis.py -l samples/vec_add.o
```
```
0  vec_add(float*, float*, float*, int)
```
### Show disassembly

```
rocdis.py -d -s -k 0 samples/vec_add.o
```
```
<_Z7vec_addPfS_S_i>:
 _Z7vec_addPfS_S_i():
 __DEVICE__ unsigned int __hip_get_block_dim_x() { return __ockl_get_local_size(0); }
        s_clause                         0x1
        s_load_b32                       s2, s[0:1], 0x2c
        s_load_b32                       s3, s[0:1], 0x18
        s_waitcnt                        lgkmcnt(0)
        s_and_b32                        s2, s2, 0xffff
        s_delay_alu                      instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
 int gid = blockDim.x * blockIdx.x + threadIdx.x;
        v_mad_u64_u32                    v[1:2], null, s15, s2, v[0:1]
 if(gid < n) {
        s_mov_b32                        s2, exec_lo
        v_cmpx_gt_i32_e64                s3, v1
        s_cbranch_execz                  27
 __DEVICE__ unsigned int __hip_get_block_dim_x() { return __ockl_get_local_size(0); }
        s_load_b128                      s[4:7], s[0:1], null
 z[gid] = x[gid] + y[gid];
        v_ashrrev_i32_e32                v2, 31, v1
 __DEVICE__ unsigned int __hip_get_block_dim_x() { return __ockl_get_local_size(0); }
        s_load_b64                       s[0:1], s[0:1], 0x10
        s_delay_alu                      instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
 z[gid] = x[gid] + y[gid];
        v_lshlrev_b64                    v[0:1], 2, v[1:2]
        s_waitcnt                        lgkmcnt(0)
        v_add_co_u32                     v2, vcc_lo, s4, v0
        s_delay_alu                      instid0(VALU_DEP_2)
        v_add_co_ci_u32_e32              v3, vcc_lo, s5, v1, vcc_lo
        v_add_co_u32                     v4, vcc_lo, s6, v0
        v_add_co_ci_u32_e32              v5, vcc_lo, s7, v1, vcc_lo
        v_add_co_u32                     v0, vcc_lo, s0, v0
        global_load_b32                  v2, v[2:3], off
        global_load_b32                  v3, v[4:5], off
        v_add_co_ci_u32_e32              v1, vcc_lo, s1, v1, vcc_lo
        s_waitcnt                        vmcnt(0)
        v_add_f32_e32                    v2, v2, v3
        global_store_b32                 v[0:1], v2, off
 }
        s_nop                            0
        s_sendmsg                        sendmsg(MSG_DEALLOC_VGPRS)
        s_endpgm
```
    
