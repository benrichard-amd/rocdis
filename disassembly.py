
def is_hex(s):
    try:
        int(s, 16)
        return True
    except:
        return False

def is_symbol(s):
    
    parts = s.split(' ')

    if len(parts) != 2:
        return False

    if len(parts[0]) != 16:
        return False

    if not is_hex(parts[0]):
        return False

    return True


def parse_symbol(s):

    # Hex digits followed by symbol
    parts = s.split()

    # Hex address
    addr = int(parts[0], 16)

    # Symbol
    sym = parts[1].strip(':')

    return {'type':'symbol',
    'address':addr,
    'symbol':sym
    }


# True if in format abs_path:number
def is_line_number(s):
    if s[0] != '/':
        return False
    
    parts = s.split(':')

    if len(parts) != 2:
        return False

    if not parts[1].isnumeric():
        return False

    return True

def parse_line_number(s):
    parts = s.split(':')

    return {'type':'line_number',
    'file':parts[0],
    'line':int(parts[1])
    }

def parse_source_code(source):

    return {
        'type':'source',
        'value':source
    }

def parse_opcode(s):

    # Format:
    # <opcode> //  <address> <encoding>
    parts = s.split('//')
    opcode = parts[0].strip()
    annotation = parts[1]

    parts = annotation.split(':')
    addr = int(parts[0],16)
    encoding = parts[1].strip()

    parts = opcode.split(' ', 1)
    opcode = parts[0]

    if len(parts) > 1:
        args = parts[1]
    else:
        args = ''

    return {
        'type':'opcode',
        'opcode':opcode,
        'addr':addr,
        'args':args,
        'encoding':encoding
    }

def parse_asm_line(line):

    try:
        if line[0] == ';':

            # Strip semicolon and spaces
            line = line[1:].strip()

            if is_line_number(line):
                return parse_line_number(line)
            else:
                return parse_source_code(line)

        elif is_symbol(line):
            return parse_symbol(line)

        elif line[0] == '\t':

            if len(line) >= 2 and line[0] == '\t' and line[1] == '\t':
                return {'type':'endprog'}

            # opcode
            return parse_opcode(line)
        else:
            return {'type':'UNKNOWN',
            'value':line
            }
    except:
        print('Error parsing line:')
        print(line)
        return {'type':'error'}