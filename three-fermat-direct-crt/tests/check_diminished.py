import ctypes
from pathlib import Path
import random
import subprocess
import tempfile

path=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as temp:
    library=Path(temp)/'d1.so'
    subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC',str(path/'check_diminished.cpp'),'-o',str(library)],check=True)
    lib=ctypes.CDLL(str(library))
    op=lib.operation
    U=ctypes.c_uint32
    op.argtypes=[U,ctypes.POINTER(U),ctypes.POINTER(U),U,U,U,ctypes.POINTER(U)]
    op.restype=None
    lib.product.argtypes=[U,ctypes.POINTER(U),ctypes.POINTER(U),ctypes.POINTER(U)]
    lib.product.restype=None
    for words in (32,40,48):
        bits=32*words
        modulus=2**bits+1
        rng=random.Random(bits)
        def encode(x):
            raw=((x-1)%(1<<bits)).to_bytes(4*words,'little')
            return (U*(words+1)).from_buffer_copy(raw+int(x==0).to_bytes(4,'little'))
        def verify(x,y,ex,ey,sub):
            out=(U*(words+1))()
            op(words,encode(x),encode(y),ex,ey,sub,out)
            got=0 if out[words] else int.from_bytes(bytes(out)[:-4],'little')+1
            assert got==(x*pow(2,ex,modulus)+(1-2*sub)*y*pow(2,ey,modulus))%modulus
        for _ in range(300):
            x,y=rng.randrange(modulus),rng.randrange(modulus)
            ex,ey=rng.randrange(2*bits),rng.randrange(2*bits)
            for sub in (0,1):verify(x,y,ex,ey,sub)
        # Force carries through every limb, exact cancellation, zero operands and wrap at every bit.
        for ex in range(2*bits):
            for sub in (0,1):verify(modulus-1,1,ex,ex,sub)
        for x in [0,1,modulus-1]+[2**j for j in range(0,bits,32)]:
            for y in (0,1,modulus-1,modulus-x if x else 0):
                for sub in (0,1):verify(x,y,0,0,sub)
        for x in [0,1,modulus-1,modulus//2,2**(bits//2)-1]+[rng.randrange(modulus) for _ in range(128)]:
            for y in [0,1,2,modulus-1,rng.randrange(modulus)]:
                out=(U*(words+1))()
                lib.product(words,encode(x),encode(y),out)
                got=0 if out[words] else int.from_bytes(bytes(out)[:-4],'little')+1
                assert got==x*y%modulus
        if words in (40,48):
            factor=2**(256 if words==40 else 512)+1
            out=(U*(words+1))()
            lib.product(words,encode(factor),encode(modulus//factor),out)
            assert out[words]==1
        print(f'PASS {bits}-bit butterflies, every rotation and exact products',flush=True)
