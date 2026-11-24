from pathlib import Path
import ctypes
import math
import random
import subprocess
import tempfile

root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as temp:
    binary=Path(temp)/'recover.so'
    subprocess.run(['c++','-O2','-std=c++17','-shared','-fPIC',str(root/'check_recovery.cpp'),'-o',str(binary)],check=True)
    fun=ctypes.CDLL(str(binary)).recover
    Word=ctypes.c_uint32
    fun.argtypes=[ctypes.POINTER(Word)]*4
    fun.restype=None
    basis=(1536,1280,1024)
    m=math.prod(2**h+1 for h in basis)
    q=2**1747-9699327
    rng=random.Random(7042)
    values=[0,1,q-1,q,q+1,m//2,m//2+1,m-1,-1,-q+1,-q]
    for h in basis:values.extend([2**h-1,2**h,2**h+1,2**h+2])
    values += [rng.randrange(m) for _ in range(2048)]
    for value in values:
        data=[]
        for h in basis:
            v=value%(2**h+1)
            data.append((Word*(h//32+1)).from_buffer_copy(((v-1)%2**h).to_bytes(h//8,'little')+int(v==0).to_bytes(4,'little')))
        out=(Word*55)()
        fun(*data,out)
        integer=value%m
        assert int.from_bytes(bytes(out),'little')==(integer if 2*integer<=m else integer-m)%q
    print(f'PASS {len(values)} native sparse CRT and final reductions')
