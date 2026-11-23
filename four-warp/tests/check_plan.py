"""Check packed coefficients, reconstruction and quotient rounding using arbitrary precision."""
from math import prod
from pathlib import Path
from random import Random
import struct
import subprocess
import tempfile

root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); exe=tmp/'plan'
    subprocess.run(['c++','-O2','-std=c++17',root/'plan_bridge.cpp','-o',exe],check=True)
    for n,w,gap,count in [(32768,868,43646975,16),(65536,1747,9699327,30)]:
        subprocess.run([exe,str(w)],cwd=tmp,check=True)
        def words(name):
            return [x[0] for x in struct.iter_unpack('<I',(tmp/name).read_bytes())]
        primes=words('primes'); inv=words('crt_inverse'); recip=words('reciprocal48')
        twists=words('twists');recover=words('recover')
        assert len(set(primes))==count
        q=(1<<w)-gap; M=prod(primes)
        cof=[M//p for p in primes]
        assert inv==[pow(c,-1,p) for c,p in zip(cof,primes)]
        assert recip==[(1<<48)//p for p in primes]
        pos=[(i*w+3)//4 for i in range(5)]
        bound=[]
        for k in range(4):
            total=0
            for i in range(4):
                j=(k-i)%4
                a=(1<<(pos[i+1]-pos[i]))-1
                b=(1<<(pos[j+1]-pos[j]))-1
                wrap=i+j>=4
                total+=((a*b)<<(pos[i]+pos[j]-pos[k]-wrap*w))*(gap if wrap else 1)
            bound.append(total*n)
        assert M>4*max(bound)
        packed=(tmp/'encode').read_bytes()
        for j,p in enumerate(primes):
            r=twists[j]
            assert pow(r,4,p)*pow(2,w,p)%p==gap%p
            chunks=2 if w==1747 else 1
            for digit,start in enumerate(pos[:-1]):
                scale=pow(2,start,p)*pow(r,digit,p)%p
                assert recover[4*j+digit]*4*scale%p==1
                for byte in range(32*chunks):
                    value=0
                    for limb in range(4):
                        lane=4*(4*(j%2)+limb)+(byte%16)//4
                        word=((((digit*chunks+byte//32)*(count//2)+j//2)*2+(byte%32)//16)*32+lane)
                        value|=packed[4*word+byte%4]<<(8*limb)
                    assert value==scale*pow(2,-(start%32),p)*pow(256,byte,p)%p
        weights=[(c<<pos[d])%q for d in range(4) for c in cof]+[(-M<<pos[d])%q for d in range(4)]
        kpad=32*((count*4+4+7)//8); limbs=(w+31)//32
        matrix=(tmp/'crt').read_bytes()
        assert len(matrix)==4*limbs*kpad
        for k in range(kpad):
            col=int.from_bytes(matrix[k::kpad],'little')
            assert col==((weights[k//4]<<(8*(k%4)))%q if k//4<len(weights) else 0)
        packed=(tmp/'packed').read_bytes()
        rows=7*((limbs+13)//14)
        header=128*((4*(rows+1)+127)//128)
        payload=128*((header+8*rows*((16*count+4+31)//32)+127)//128)
        columns=list(range(16*count))+[16*count+4*j for j in range(4)]
        offsets=struct.unpack_from(f'<{rows+1}I',packed)
        assert offsets[0]==0 and 0<offsets[-1]<rows*(kpad//32)
        assert len(packed)==payload+256*offsets[-1]
        for row,(start,end) in enumerate(zip(offsets,offsets[1:])):
            dense=bytearray(8*len(columns))
            recovered=bytearray(len(dense))
            for c in range(8):
                out=8*row+c
                for i,k in enumerate(columns):
                    if k is not None and out<4*limbs:dense[c*len(columns)+i]=matrix[out*kpad+k]
            used=set()
            for tile in range(start,end):
                for slot,word in enumerate(packed[header+8*tile:header+8*tile+8]):
                    mapped=[]
                    for c in range(8):
                        index=payload+256*tile+128*(slot//4)+4*(4*c+slot%4)
                        mapped.append(packed[index:index+4])
                    if not any(any(chunk) for chunk in mapped):continue
                    assert word not in used and 4*word<len(columns)
                    used.add(word)
                    for c,chunk in enumerate(mapped):
                        index=c*len(columns)+4*word
                        recovered[index:index+4]=chunk
            assert dense==recovered
        print(f'PASS W={w}: {offsets[-1]} packed dot tiles and every omitted word')
        rng=Random(w)
        for case in range(128):
            vals=[(bound[d] if case==0 else -bound[d] if case==1 else rng.randint(-bound[d],bound[d])) for d in range(4)]
            signed_terms=[]; quotients=[]
            for d,x in enumerate(vals):
                res=[x*inv[j]%p for j,p in enumerate(primes)]
                quotient=(sum(a*b for a,b in zip(res,recip))+(1<<47))>>48
                assert sum(a*b for a,b in zip(res,cof))-quotient*M==x
                signed_terms.extend(res);quotients.append(quotient)
            assert sum(a*b for a,b in zip(signed_terms+quotients,weights))%q==sum(x<<pos[d] for d,x in enumerate(vals))%q
        print(f'PASS W={w}: {len(matrix)} reconstruction bytes, 128 signed joint reconstructions')
