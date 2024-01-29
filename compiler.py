from enum import IntEnum
from typing import Optional

class Tag(IntEnum):
    U = 0
    I = 1
    A = 2
    X = 3
    F = 5

class OpCode(IntEnum):
    Label = 1
    FuncInfo = 2
    IntCodeEnd = 3
    Return = 19
    Move = 64
    GcBif2 = 125

class Expr:
    pass

class Var(Expr):
    def __init__(self, text):
        self.text = text

class Number(Expr):
    def __init__(self, value):
        self.value = value

class Binop(Expr):
    def __init__(self, kind, lhs, rhs):
        self.kind = kind
        self.lhs = lhs
        self.rhs = rhs

class BinopKind:
    Sum = "Sum"
    Sub = "Sub"

class Type(IntEnum):
    Int = 1

class Param:
    def __init__(self, name, type, index):
        self.name = name
        self.type = type
        self.index = index

class Func:
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class CompiledFunc:
    def __init__(self, label, arity):
        self.label = label
        self.arity = arity

class Module:
    def __init__(self):
        self.funcs = {}


def encode_arg(tag, n):
    if n < 0:
        raise NotImplementedError("Negative numbers not supported yet")
    elif n < 16:
        return [(n << 4) | tag]
    elif n < 0x800:
        a = (((n >> 3) & 0b11100000) | tag | 0b00001000).to_bytes(1, 'big')
        b = (n & 0xFF).to_bytes(1, 'big')
        return list(a + b)
    else:
        raise NotImplementedError("Large numbers not supported yet")

def pad_chunk(word_size, chunk):
    len_chunk = len(chunk)
    new_len = (len_chunk + word_size - 1) // word_size * word_size
    chunk.extend([0] * (new_len - len_chunk))

def encode_chunk(tag, chunk):
    result = bytearray()
    result.extend(tag)
    result.extend(len(chunk).to_bytes(4, 'big'))
    result.extend(chunk)
    pad_chunk(4, result)
    return result



def compile_expr(expr, atoms, imports, code, params, stack_size):
    stack_start = len(params)
    if isinstance(expr, Var):
        name = expr.text
        param = params.get(name)
        if param is not None:
            code.append(OpCode.Move.value)
            code.extend(encode_arg(Tag.X, param.index))
            code.extend(encode_arg(Tag.X, stack_start + stack_size[0]))
            stack_size[0] += 1
            return True
        else:
            print(f"ERROR: Unknown variable {name}")
            return False
    elif isinstance(expr, Number):
        x = expr.value
        code.append(OpCode.Move.value)
        code.extend(encode_arg(Tag.I, x))
        code.extend(encode_arg(Tag.X, stack_start + stack_size[0]))
        stack_size[0] += 1
        return True
    elif isinstance(expr, Binop):
        kind = expr.kind
        lhs = expr.lhs
        rhs = expr.rhs

        if not compile_expr(lhs, atoms, imports, code, params, stack_size) or \
           not compile_expr(rhs, atoms, imports, code, params, stack_size):
            return False

        assert stack_size[0] >= 2

        code.append(OpCode.GcBif2.value)
        code.extend(encode_arg(Tag.F, 0))  # Lbl
        code.extend(encode_arg(Tag.U, 2))  # Live

        bif2 = imports.get(resolve_function_signature(atoms, "erlang", "+", 2))
        bif2 = bif2 if kind == BinopKind.Sum else imports.get(resolve_function_signature(atoms, "erlang", "-", 2))

        if bif2 is not None:
            code.extend(encode_arg(Tag.U, bif2))
            code.extend(encode_arg(Tag.X, stack_start + stack_size[0] - 2))  # Arg1
            code.extend(encode_arg(Tag.X, stack_start + stack_size[0] - 1))  # Arg2
            code.extend(encode_arg(Tag.X, stack_start + stack_size[0] - 2))  # Res
            stack_size[0] -= 1
            return True
        else:
            return False

    return False



def encode_code_chunk(module, imports, atoms, labels):
    label_count = 0
    function_count = 0
    code = bytearray()

    for _, func in module.funcs.items():
        function_count += 1

        label_count += 1
        code.append(OpCode.Label)
        code.extend(encode_arg(Tag.U, label_count))

        code.append(OpCode.FuncInfo)
        code.extend(encode_arg(Tag.A, atoms.get_id("bada")))
        name_id = atoms.get_id(func.name)
        code.extend(encode_arg(Tag.A, name_id))
        code.extend(encode_arg(Tag.U, len(func.params)))

        label_count += 1
        code.append(OpCode.Label)
        code.extend(encode_arg(Tag.U, label_count))
        labels[name_id] = CompiledFunc(label_count, len(func.params))

        stack_size = [0]
        compile_expr(func.body, atoms, imports, code, func.params, stack_size)

        if len(func.params) > 0:
            code.append(OpCode.Move)
            code.extend(encode_arg(Tag.X, len(func.params)))
            code.extend(encode_arg(Tag.X, 0))

        code.append(OpCode.Return)

    code.append(OpCode.IntCodeEnd)

    label_count += 1
    sub_size = 16
    instruction_set = 0
    opcode_max = 169

    chunk = bytearray()
    chunk.extend(sub_size.to_bytes(4, 'big'))
    chunk.extend(instruction_set.to_bytes(4, 'big'))
    chunk.extend(opcode_max.to_bytes(4, 'big'))
    chunk.extend(label_count.to_bytes(4, 'big'))
    chunk.extend(function_count.to_bytes(4, 'big'))
    chunk.extend(code)

    return encode_chunk(b"Code", chunk)
def encode_atom_chunk(atoms):
    chunk = bytearray()
    chunk.extend(len(atoms.names).to_bytes(4, 'big'))
    for atom in atoms.names:
        chunk.extend(len(atom).to_bytes(1, 'big'))
        chunk.extend(atom.encode('utf-8'))

    return encode_chunk(b"AtU8", chunk)

def resolve_function_signature(atoms, module, func, arity):
    return (
        atoms.get_id(module),
        atoms.get_id(func),
        arity
    )

def encode_imports_chunk(atoms, imports):
    chunk = bytearray()
    import_count = 2
    chunk.extend(import_count.to_bytes(4, 'big'))

    module, func, arity = resolve_function_signature(atoms, "erlang", "+", 2)
    chunk.extend(module.to_bytes(4, 'big'))
    chunk.extend(func.to_bytes(4, 'big'))
    chunk.extend(arity.to_bytes(4, 'big'))
    imports[(module, func, arity)] = 0

    module, func, arity = resolve_function_signature(atoms, "erlang", "-", 2)
    chunk.extend(module.to_bytes(4, 'big'))
    chunk.extend(func.to_bytes(4, 'big'))
    chunk.extend(arity.to_bytes(4, 'big'))
    imports[(module, func, arity)] = 1

    return encode_chunk(b"ImpT", chunk)

def encode_exports_chunk(labels):
    chunk = bytearray()
    export_count = len(labels)
    chunk.extend(export_count.to_bytes(4, 'big'))

    for name_id, compiled_func in labels.items():
        chunk.extend(name_id.to_bytes(4, 'big'))
        chunk.extend(compiled_func.arity.to_bytes(4, 'big'))
        chunk.extend(compiled_func.label.to_bytes(4, 'big'))

    return encode_chunk(b"ExpT", chunk)

class Atoms:
    def __init__(self):
        self.names = []

    def get_id(self, needle):
        result = next((index + 1 for index, name in enumerate(self.names) if name == needle), None)
        if result is not None:
            return result
        else:
            self.names.append(needle)
            return len(self.names)

def encode_string_chunk():
    return encode_chunk(b"StrT", [])

def compile_beam_module(module):
    atoms = Atoms()
    labels = {}
    imports = {}

    # TODO: get module name from the stem of the input file
    _ = atoms.get_id("bada")

    beam = bytearray()
    beam.extend(b"BEAM")
    beam.extend(encode_imports_chunk(atoms, imports))
    beam.extend(encode_code_chunk(module, imports, atoms, labels))
    beam.extend(encode_exports_chunk(labels))
    beam.extend(encode_string_chunk())
    beam.extend(encode_atom_chunk(atoms))

    return beam
def generate_output_bytes(module):
    beam = compile_beam_module(module)
    bytes_data = bytearray()
    bytes_data.extend(b"FOR1")
    bytes_data.extend(len(beam).to_bytes(4, 'big'))
    bytes_data.extend(beam)
    return bytes_data
