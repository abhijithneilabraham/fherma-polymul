"""Compare all four CUDA blocks, including subgroup votes and shuffles, with integer DFTs."""
import ctypes
from pathlib import Path
import random
import subprocess
import tempfile

root=Path(__file__).resolve().parent
def reverse(i,bits):
    return int(f'{i:0{bits}b}'[::-1],2)

def direct(values,mod,bits,inverse,negacyclic):
    n=len(values)
    root=pow(2,2*bits//n,mod)
    twist=pow(2,bits//n,mod) if negacyclic else 1
    weight=1
    if inverse:
        values=[values[reverse(i,n.bit_length()-1)] for i in range(n)]
        root=pow(root,-1,mod)
    else:
        values=[v*pow(twist,i,mod)%mod for i,v in enumerate(values)]
    output=[]
    for index in range(n):
        acc=0
        for v in reversed(values):
            acc=(acc*weight+v)%mod
        output.append(acc)
        weight=weight*root%mod
    if inverse:
        return [v*pow(n*pow(twist,i,mod),-1,mod)%mod for i,v in enumerate(output)]
    return [output[reverse(i,n.bit_length()-1)] for i in range(n)]

with tempfile.TemporaryDirectory() as temp:
    temp=Path(temp);(temp/'cuda_runtime.h').write_text('')
    binary=temp/'blocks.so'
    subprocess.run(['c++','-O2','-std=c++17','-Wno-deprecated-declarations','-shared','-fPIC','-I',str(temp),str(root/'blocks.cpp'),'-o',str(binary)],check=True)
    operation=ctypes.CDLL(str(binary)).block
    U=ctypes.c_uint32
    operation.argtypes=[U,U,ctypes.POINTER(U),ctypes.POINTER(U),ctypes.POINTER(U)]
    operation.restype=None
    for words in (32,40,48):
        bits=32*words;mod=2**bits+1;rng=random.Random(27+words)
        def pack(values,wide=False):
            if wide: raw=b''.join(v.to_bytes(220,'little') for v in values)
            else:raw=b''.join(((v-1)%2**bits).to_bytes(4*words,'little')+int(v==0).to_bytes(4,'little') for v in values)
            return (U*(len(raw)//4)).from_buffer_copy(raw)
        for stage in range(4):
            size=512 if stage==2 else 256
            x=[rng.randrange(2**1747-9699327 if stage==0 else mod) for _ in range(size)]
            x[:4]=[0,1,mod-1,mod] if stage==0 else [0,1,mod-1,0]
            y=[rng.randrange(mod) for _ in range(size)];y[:4]=[0,mod-1,1,0]
            if stage==0:ref=direct([v%mod for v in x],mod,bits,False,True)
            elif stage==1:ref=direct(x+[0]*256,mod,bits,False,False)
            elif stage==2:
                terms=direct([a*b%mod for a,b in zip(x,y)],mod,bits,True,False)
                twist=pow(2,(2*reverse(37,8)+1)*(bits//256),mod)
                ref=[(a+twist*b)%mod for a,b in zip(terms[:256],terms[256:])]
            else:ref=direct(x,mod,bits,True,True)
            output=(U*(len(ref)*(words+1)))()
            operation(words,stage,pack(x,stage==0),pack(y),output)
            for i,expected in enumerate(ref):
                j=i*(words+1)
                actual=0 if output[j+words] else int.from_bytes(bytes((U*words)(*output[j:j+words])),'little')+1
                assert actual==expected,(words,stage,i)
        print(f'PASS all four submitted blocks at {bits} bits against independent integer transforms',flush=True)
