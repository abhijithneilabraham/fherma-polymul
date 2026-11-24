"""Run every native decoder lane on a boundary tile and compare all output limbs."""
from math import prod
from pathlib import Path
from random import Random
import re
import subprocess
import tempfile


root=Path(__file__).resolve().parent
source=root.parent
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    code=(source/'warp-alpha.cuh').read_text().split('inline void launch_g1_',1)[0]+'\n}\n'
    code=code.replace('#include "core/g1-crt.cuh"','')
    pattern=r'asm volatile\(\s*"mma\.sync\.aligned\.m16n8k32[^;]*;\\n".*?\);'
    calls=re.findall(pattern,code,re.S)
    assert len(calls)==2
    for start,call in zip((0,4),calls):
        args=re.findall(r'"\+?r"\(([a-z]\d)\)',call)
        assert args==[f'c{i}' for i in range(start,start+4)]+[f'a{i}' for i in range(start,start+4)]+['b0','b1']
        code=code.replace(call,'emulate_mma('+','.join(args)+');',1)
    (tmp/'decoder-emulated.hpp').write_text(code)
    exe=tmp/'decode'
    subprocess.run(['c++','-std=c++20','-O2','-pthread','-I',str(tmp),'-I',str(source),root/'decoder_test.cpp','-o',exe],check=True)
    for n,w,gap in [(32768,868,43646975),(65536,1747,9699327)]:
        rng=Random(w)
        spec='large' if w==1747 else 'small'
        section=(source/'two-primes.hpp').read_text().split(spec+'[] = {',1)[1].split('};',1)[0]
        constants=[tuple(map(int,pair)) for pair in re.findall(r'\{(\d+)u,\s*(\d+)u\}',section)]
        assert len(constants)==(56 if w==1747 else 30)
        ps=[p for p,_ in constants];m=prod(ps);q=(1<<w)-gap
        base=1<<(880 if w==1747 else 440)
        inputs=[w];expected=[]
        for row in range(32):
            limit=(m-1)//4
            def value(index):
                extremes=[limit,-limit,0,-1,1]
                return extremes[index] if index<len(extremes) else rng.randint(-limit,limit)
            low=value(row);high=value(31-row)
            xy=[(low*pow(m//p,-1,p)%p,high*pow(m//p,-1,p)%p) for p in ps]
            for sign in [1,-1]:
                inputs.extend((x+sign*theta*y)*pow(2,-1,p)%p for (x,y),(p,theta) in zip(xy,constants))
            expected.append((low+base*high)%q)
        result=subprocess.run([exe],input=' '.join(map(str,inputs)),text=True,capture_output=True,check=True,timeout=120)
        actual=[sum(int(v)<<(32*j) for j,v in enumerate(line.split())) for line in result.stdout.splitlines()]
        assert expected==actual
        print(f'PASS full {w}-bit decoder block with {32*((w+31)//32)} output words')
