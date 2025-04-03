#!/usr/bin/env python

import sys
import rocdislib
import getopt

_print_lines = False
_print_source = False

def print_kernels(dis):

    kernels = dis.kernels()

    for i in range(0, len(kernels)):
        print('{}  {}'.format(i, kernels[i]['name']))

def pretty_print(dis, start, end):

    for i in range(start, end+1):
        d = dis._data[i]

        if d['type'] == 'opcode':
            print('    {0:32} {1}'.format(d['opcode'], d['args']))
        elif d['type'] == 'source':
            if _print_source:
                print(' {}'.format(d['value']))
        elif d['type'] == 'symbol':
            print('{}:'.format(d['symbol']))
        elif d['type'] == 'line_number':
            if _print_lines:
                print('{}:{}'.format(d['file'], d['line']))
        else:
            print('UNKNOWN {}'.format(d))

def print_disassembly(dis, k_idx):

    start = 0
    end = len(dis._data) - 1

    if k_idx >= 0:
        start = dis.kernels()[k_idx]['start']
        end = dis.kernels()[k_idx]['end']

    pretty_print(dis, start, end)



def usage():
    print('ROCm disassembler')
    print('')
    print('USAGE: rocdis [options] <input file>')
    print('')
    print('-d             Disassemble')
    print('-h             Show usage')
    print('-l             List kernels')
    print('-k <index>     Disassemble specified kernel')
    print('')

def main():

    try:
        opts, operands = getopt.getopt(sys.argv[1:], 'dk:l')
    except getopt.GetoptError as err:
        print(err)
        return 1

    l_flag = False
    d_flag = False
    k_idx = -1
  
    for opt, arg in opts:
        if opt == '-l':
            l_flag = True
        elif opt == '-d':
            d_flag = True
        elif opt == '-k':
            k_idx = int(arg)
        else:
            print('error: unknown argument \'{}\''.format(opt))
            return 1

    if len(operands) == 0:
        usage()
        return 1

    if len(operands) > 1:
        print('Only 1 file can be disassembled at a time')
        return 1

    executable = operands[0]

    obj = rocdislib.load_code_object(executable)

    dis = rocdislib.disassemble(obj)
    
    if k_idx >= len(dis.kernels()):
        print('Invalid kernel index')
        return 1

    if l_flag:
        print_kernels(dis)
    elif d_flag:
        print_disassembly(dis, k_idx)

if __name__ == "__main__":
    ret = main()
    sys.exit(ret)