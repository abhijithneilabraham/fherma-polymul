"""Feed exact scaled residues into the same cooperative CRT used by CUDA."""
import ctypes
from math import prod
from pathlib import Path
import random
import subprocess
from tempfile import TemporaryDirectory

root = Path(__file__).resolve().parent
widths = (1536, 1280, 1024)
moduli = [2**h + 1 for h in widths]
product = prod(moduli)
q = 2**1747 - 9699327
bound = 65536*(q-1)**2
rng = random.Random(263)
values = {0, 1, q, q-1, q+1, 2*q-1, bound, bound-1, (1 << 3494)-1}
for shift in range(0, 3510, 32):
    values.update((2**shift-1,2**shift,2**shift+1))
for h, m in zip(widths, moduli):
    for top in range(9):
        for offset in (-1,0,1,2):
            values.add(((top << (h-3))+offset)*(product//m)%m)
values.update(rng.randrange(bound+1) for _ in range(256))
values = sorted({sign*v for v in values for sign in [-1,1] if v<=bound})
rng.shuffle(values)

U = ctypes.c_uint32
with TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp/'cuda_runtime.h').touch()
    library = tmp/'block.so'
    subprocess.run(['c++','-O2','-std=c++17','-Wno-deprecated-declarations','-shared','-fPIC',
                    '-I',str(tmp),str(root/'blocks.cpp'),'-o',str(library)],check=True)
    block = ctypes.CDLL(str(library)).final_block
    block.argtypes = [ctypes.POINTER(U)]*4
    block.restype = None
    for begin in range(0,len(values),32):
        batch = values[begin:begin+32]
        batch += [0]*(32-len(batch))
        inputs = []
        for h, m in zip(widths,moduli):
            inverse = pow(product//m,-1,m)
            residues = [v*inverse%m for v in batch]
            packed = b''.join(((r-1)%2**h).to_bytes(h//8,'little') + int(r==0).to_bytes(4,'little') for r in residues)
            inputs.append((U*(len(packed)//4)).from_buffer_copy(packed))
        result = (U*(32*55))()
        block(*inputs,result)
        word_bytes = bytes(result)
        for i, v in enumerate(batch):
            actual = int.from_bytes(word_bytes[i*220:(i+1)*220],'little')
            assert actual == v%q,(begin+i,v,actual,v%q)
print(f'PASS native cooperative CRT for {len(values)} signed coefficients')
