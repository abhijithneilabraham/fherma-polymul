#!/usr/bin/env python3
"""Run the actual finish kernel with 32 CPU coroutines per block and Python modulo."""
from pathlib import Path
import random
import subprocess
import tempfile


def function(text, marker):
    begin=text.index(marker)
    brace=text.index('{',begin);depth=1;end=brace+1
    while depth:
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        end+=1
    return text[begin:end]


def check():
    root=Path(__file__).resolve().parents[1]
    core=(root/'core/g1-crt.cuh').read_text()
    fold=function(core,'template <unsigned W, unsigned L>\n__device__ __forceinline__ void fold_near_power')
    reduce=function(core,'template <unsigned W, unsigned L>\n__device__ __forceinline__ void reduce_near_power')
    finish=function((root/'split-crt.cuh').read_text(),'template<unsigned W, unsigned L>\n__global__ void finish_crt')
    finish=finish.replace('__global__ void','Thread').replace('__shared__','static')
    finish=finish.replace('__syncthreads();','co_await std::suspend_always{};')
    source='namespace bytecrt { namespace detail {\n'+fold+'\n'+reduce+'\n}\n'+finish+'\n}\n'
    rng=random.Random(792053);lines=[];expected=[];cases=0
    for w,L in [(1747,55),(868,28)]:
        for gap in [1,9699327 if w==1747 else 43646975,(1<<32)-1]:
            q=(1<<w)-gap
            values=[0,1,q-1,q,q+1,(1<<w)-1,1<<w,(1<<(w+38))-1]
            values += [rng.randrange(1<<(w+38)) for _ in range(96-len(values))]
            lines.append(f'{w} 96 {gap}')
            lines.append(' '.join(str((q>>(32*i))&0xffffffff) for i in range(L)))
            lines.append(' '.join(str((x>>(32*i))&0xffffffff) for x in values for i in range(L+2)))
            expected.append([(x%q>>(32*i))&0xffffffff for x in values for i in range(L)])
            cases+=len(values)
    with tempfile.TemporaryDirectory(prefix='split-finish-') as temporary:
        build=Path(temporary);(build/'finish-kernel.inc').write_text(source)
        subprocess.run(['c++','-std=c++20','-O2','-Wall','-Wextra','-I',str(build),
                        str(root/'tests/finish_bridge.cpp'),'-o',str(build/'contract')],check=True)
        result=subprocess.run([str(build/'contract')],input='\n'.join(lines)+'\n',
                              text=True,capture_output=True,check=True,timeout=30)
    actual=[list(map(int,row.split())) for row in result.stdout.splitlines()]
    assert actual==expected
    print(f'PASS {cases} exact finish outputs; three-block barriers/guards/input ownership checked; GPU executions=0')


if __name__=='__main__':check()
