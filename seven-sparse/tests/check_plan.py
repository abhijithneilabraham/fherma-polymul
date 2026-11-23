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
    for n,w,gap,count in [(32768,868,43646975,10),(65536,1747,9699327,18)]:
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
        pos=[(i*w+6)//7 for i in range(8)]
        bound=[]
        for k in range(7):
            total=0
            for i in range(7):
                j=(k-i)%7
                a=(1<<(pos[i+1]-pos[i]))-1
                b=(1<<(pos[j+1]-pos[j]))-1
                wrap=i+j>=7
                total+=((a*b)<<(pos[i]+pos[j]-pos[k]-wrap*w))*(gap if wrap else 1)
            bound.append(total*n)
        assert M>4*max(bound)
        packed=(tmp/'encode').read_bytes()
        for j,p in enumerate(primes):
            table=[twists[7*j+d] for d in range(7)]
            r=table[1]*pow(2,-pos[1],p)%p
            assert pow(r,7,p)*pow(2,w,p)%p==gap%p
            for d,t in enumerate(table):
                assert t==pow(2,pos[d],p)*pow(r,d,p)%p
                assert recover[7*j+d]*7*t%p==1
            for b in range(32):
                v=sum(packed[(4*j+i)*32+b]<<(8*i) for i in range(4))
                assert v==pow(256,b,p)
        weights=[(c<<pos[d])%q for d in range(7) for c in cof]+[(-M<<pos[d])%q for d in range(7)]
        kpad=32*((count*7+14)//8); limbs=(w+31)//32
        matrix=(tmp/'crt').read_bytes()
        assert len(matrix)==4*limbs*kpad
        for k in range(kpad):
            col=int.from_bytes(matrix[k::kpad],'little')
            assert col==((weights[k//4]<<(8*(k%4)))%q if k//4<len(weights) else 0)
        packed=(tmp/'packed').read_bytes()
        tile_rows=7*((limbs+13)//14)
        header_bytes=128*((2*tile_rows+31)//32)
        cursor=0
        for row in range(tile_rows):
            mask,first=struct.unpack_from('<II',packed,8*row)
            assert first==cursor and mask>>(kpad//32)==0
            for ktile in range(kpad//32):
                live=bool(mask & (1<<ktile))
                tile=packed[header_bytes+256*cursor:header_bytes+256*(cursor+1)] if live else bytes(256)
                assert len(tile)==256 and any(tile)==live
                for lane in range(32):
                    col=8*row+lane//4
                    for half in range(2):
                        start=col*kpad+32*ktile+16*half+4*(lane%4)
                        expected=matrix[start:start+4] if col<4*limbs else bytes(4)
                        assert tile[128*half+4*lane:128*half+4*lane+4]==expected
                cursor+=live
        assert cursor=={868:86,1747:225}[w] and len(packed)==header_bytes+256*cursor
        print(f'PASS W={w}: {cursor} live tiles of {tile_rows*kpad//32}')
        rng=Random(w)
        for case in range(128):
            vals=[(bound[d] if case==0 else -bound[d] if case==1 else rng.randint(-bound[d],bound[d])) for d in range(7)]
            signed_terms=[]; quotients=[]
            for d,x in enumerate(vals):
                res=[x*inv[j]%p for j,p in enumerate(primes)]
                quotient=(sum(a*b for a,b in zip(res,recip))+(1<<47))>>48
                assert sum(a*b for a,b in zip(res,cof))-quotient*M==x
                signed_terms.extend(res);quotients.append(quotient)
            assert sum(a*b for a,b in zip(signed_terms+quotients,weights))%q==sum(x<<pos[d] for d,x in enumerate(vals))%q
        print(f'PASS W={w}: {len(matrix)} reconstruction bytes, 128 signed joint reconstructions')
