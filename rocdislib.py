import subprocess
import os
import random

import disassembly

LLVM_OBJDUMP_PATH='/opt/rocm/llvm/bin/llvm-objdump'
LLVM_CXXFILT_PATH='/usr/bin/llvm-cxxfilt'

def run_cmd(args):
    output = subprocess.getoutput(' '.join(args))
    return output.splitlines()

def get_file_format(file_path):
    args = [LLVM_OBJDUMP_PATH, '-f', file_path]

    objdump_out = run_cmd(args)
    for line in objdump_out:
        if line.startswith('architecture:'):
            return line.split(' ')[1]

    return None

# Represents an AMDGPU code object file
# If provided bytes, will write to tmp file
class AMDGPUCodeObject:

    fname = ''

    def __init__(self, filepath):
        self.fname = filepath

    @classmethod
    def from_bytes(cls, data):

        # Generate temp file
        tmpname = os.path.join('/tmp', 'rocdis-' + str(random.randint(0, 2**64)) + '.co')

        with open(tmpname, 'wb') as f:
            f.write(data)
        
        return cls(tmpname)

# Extract code objects from x86 binary/object file
def extract_code_objects(file_path):

    executable = file_path

    # GNU objdump
    cmd = ['objdump', '-h', executable]

    lines = run_cmd(cmd)

    # Check that there is a HIP binary embedded in this exe
    found = False
    for line in lines:

        # Parse out the bundle size and offset
        if '.hip_fatbin' in line:
            # size VMA LMA Offset
            words = line.strip().split()
            size = int(words[2], 16)
            hip_bin_offset = int(words[5], 16)

            found = True

    if not found:
        print('Unable to find symbols in {}'.format(executable))
        return None

    # Read the HIP binary
    with open(executable, 'rb') as f:
        f.seek(hip_bin_offset)
        data = f.read(size)

    # First 24 bytes are magic string
    if data[:24] != b'__CLANG_OFFLOAD_BUNDLE__':
        print('Not a clang offload bundle')
        return None

    # Get number of code objects
    num_entries = int.from_bytes(data[24:][:8], 'little')

    bundle_info = []
    code_objects = []

    start = 24 + 8
    cur_offset = start
    for idx in range(0, num_entries):
        entry_offset = int.from_bytes(data[cur_offset:][:8], 'little')
        cur_offset += 8
    
        entry_size = int.from_bytes(data[cur_offset:][:8], 'little')
        cur_offset += 8
        
        triple_size = int.from_bytes(data[cur_offset:][:8], 'little')
        cur_offset += 8

        triple = data[cur_offset:][:triple_size].decode('utf-8')

        cur_offset += triple_size

        bundle_info.append({
            'entry_offset':entry_offset,
            'entry_size':entry_size,
            'triple':triple
        })

        if entry_size > 0:
            bundle_bytes = data[entry_offset:][:entry_size]
            code_objects.append(AMDGPUCodeObject.from_bytes(bundle_bytes))

    return code_objects

def load_code_object(filename):
    executable = filename

    format = get_file_format(executable)
    if format == None:
        print('Unknown file format')
        return None
    elif format == 'x86_64':
        objs = extract_code_objects(executable)

        if len(objs) > 1:
            print('Found {} code objects. Using first one'.format(len(objs)))
        
        code_object = objs[0]
    else:
        code_object = AMDGPUCodeObject(executable)

    return code_object

def demangle_name(name):
    args = [LLVM_CXXFILT_PATH, name]

    demangled = run_cmd(args)[0]

    return demangled

class AMDGPUDisassembly:

    _kernels = []
    _data = []

    def __init__(self, data, kernel_names):

        self._data = data
        self._kernels = []

        for i in range(0, len(self._data)):
            if self._data[i]['type'] == 'symbol':
                symbol = self._data[i]['symbol']
                if symbol.startswith('<') and symbol.endswith('>'):
                    kernel_name_mangled = symbol[1:-1]

                    # Kernel with preloaded arguments
                    if kernel_name_mangled.endswith('_preloaded'):
                        kernel_name_mangled = kernel_name_mangled[:-10]

                    if kernel_name_mangled in kernel_names:
                        k = {}
                        k['mangled'] = kernel_name_mangled
                        k['name'] = demangle_name(kernel_name_mangled)
                        k['start'] = i

                        if len(self._kernels) > 0:
                            self._kernels[-1]['end'] = i - 1
                        self._kernels.append(k)

        # Update end of last kernel
        self._kernels[-1]['end'] = len(self._data) - 1
    
    def kernels(self):
        return self._kernels

# Parse the .rodata section and get names from kernel descriptors
def get_kernel_names(obj):
    args = [LLVM_OBJDUMP_PATH, '--disassemble', '--section=.rodata', obj.fname]
    objdump_out = run_cmd(args)

    kernels = []

    for line in objdump_out:
        if disassembly.is_symbol(line):
            sym = disassembly.parse_symbol(line)
            name = sym['symbol']
            if name.startswith('<') and name.endswith('>'):

                if name[-4:-1] == '.kd':
                    kernels.append(name[1:-4])
 
    return kernels


def disassemble(obj):

    kernels = get_kernel_names(obj)

    args = [LLVM_OBJDUMP_PATH, '--source', '--disassemble', '--line-numbers', obj.fname]
    lines = run_cmd(args)

    data = []
    for i in range(0, len(lines)):
        line = lines[i]

        if len(line) == 0:
            continue

        # Skip objdump messages
        if 'Disassembly of' in line:
            continue
        if 'file format' in line:
            continue

        # Used to pad shader buffer on RDNA
        if 's_code_end' in line:
            continue

        data.append(disassembly.parse_asm_line(line))


    return AMDGPUDisassembly(data, kernels)