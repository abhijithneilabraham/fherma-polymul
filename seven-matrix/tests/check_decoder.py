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
    code=(source/'seven-decode.cuh').read_text()
    pattern=r'asm volatile\(\s*"mma\.sync\.aligned\.m16n8k32[^;]*;\\n".*?\);'
    calls=re.findall(pattern,code,re.S)
    assert len(calls)==3
    for start,call in zip((0,0,4),calls):
        args=re.findall(r'"\+?r"\(([a-z]\d)\)',call)
        assert args==[f'c{i}' for i in range(start,start+4)]+[f'a{i}' for i in range(start,start+4)]+['b0','b1']
        code=code.replace(call,'emulate_mma('+','.join(args)+');',1)
    (tmp/'decoder-emulated.hpp').write_text(code)
    exe=tmp/'decode'
    subprocess.run(['c++','-std=c++20','-O2','-pthread','-I',str(tmp),'-I',str(source),root/'decoder_test.cpp','-o',exe],check=True)
    for n,w,gap in [(32768,868,43646975),(65536,1747,9699327)]:
        rng=Random(w)
        params=(source/'prime-roots.hpp').read_text().split(f'roots{w}[] = {{',1)[1].split('};',1)[0]
        ps=[int(x) for x in re.findall(r'\{(\d+)u,',params)]
        assert len(ps)==(18 if n==65536 else 10)
        m=prod(ps);q=(1<<w)-gap
        s={'bounds':[(m-1)//4]*7,'positions':[(i*w+6)//7 for i in range(8)]}
        inv=[pow(m//p,-1,p) for p in ps]
        inputs=[w];expected=[]
        for i in range(32):
            digits=[(b if i==0 else -b if i==1 else 0 if i==2 else -1 if i==3 else rng.randint(-b,b))for b in s['bounds']]
            inputs.extend(x*c%p for x in digits for c,p in zip(inv,ps))
            expected.append(sum(x<<shift for x,shift in zip(digits,s['positions']))%q)
        result=subprocess.run([exe],input=' '.join(map(str,inputs)),text=True,capture_output=True,check=True,timeout=120)
        actual=[sum(int(v)<<(32*j) for j,v in enumerate(line.split())) for line in result.stdout.splitlines()]
        assert expected==actual
        print(f'PASS full {w}-bit decoder block with {32*((w+31)//32)} output words')
