#!/usr/bin/env python3
"""Actual warp sum versus arbitrary-precision centered CRT, both full plans."""
from pathlib import Path
import importlib.util
import math
import random
import subprocess
import tempfile


def check():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('bytecrt_oracle',root/'tests/bytecrt_reference.py')
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    rng=random.Random(286410)
    expected=[];rows=[]
    for n,w,gap in [(65536,1747,9699327),(32768,868,43646975)]:
        q=(1<<w)-gap
        product,primes,bases,inverses,_=oracle.plan(n,w,q)
        bound=n*(q-1)**2
        assert product>4*bound
        fixtures=[-bound,-bound+1,-1,0,1,bound-1,bound]
        fixtures += [rng.randrange(-bound,bound+1) for _ in range(512)]
        for coefficient in fixtures:
            terms=[(coefficient%p)*inverse%p for p,inverse in zip(primes,inverses)]
            numerator=sum(t*base for t,base in zip(terms,bases))
            alpha=(2*numerator+product)//(2*product)
            assert numerator-alpha*product==coefficient
            expected.append(alpha)
            rows += [str(len(primes)),' '.join(map(str,primes)),' '.join(map(str,terms))]
        unit=2**-24
        assert len(primes)*(16*unit/(1-16*unit))<.000109
    with tempfile.TemporaryDirectory(prefix='alpha-contract-') as temporary:
        binary=Path(temporary)/'alpha'
        subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra',
                        str(root/'tests/alpha_bridge.cpp'),'-o',str(binary)],check=True)
        result=subprocess.run([str(binary)],input='\n'.join(rows)+'\n',text=True,
                              capture_output=True,check=True,timeout=60)
    actual=list(map(int,result.stdout.split()))
    assert actual==expected
    assert sorted(row for warp in range(7) for row in range(warp,32,7))==list(range(32))
    for stride in [124,68]:
        assert 32//len({row*stride%32 for row in range(32)})==4
        assert len({lane%32 for lane in range(32)})==32
    print(f'PASS {len(expected)} exact centered-CRT alpha checks; native GPU executions=0')


if __name__=='__main__':check()
