"""Exercise the production warp reducer against an independent full-width oracle."""
from pathlib import Path
from random import Random
import subprocess
import tempfile

def fixtures(w,L,gap):
    rng=Random(34167)
    base=1<<32
    q=(1<<w)-gap
    # Construct exact values by splitting the most significant limb into
    # low/high. These include both folds, the canonical subtraction and zero.
    for x in (0,1,q-1,q,q+1,(1<<w)-1,1<<w,(1<<w)+1,2*q-1,2*q,2*q+1,(1<<(w+37))-1):
        words=[(x>>(32*j))&(base-1) for j in range(L)]
        high=[0]*L
        high[-1]=x>>(32*L)
        yield words,high
    for stop in range(1,L+1):
        # The carry starts at word one and crosses the first warp tile as needed.
        lo=[base-1 if 1<=j<stop else 0 for j in range(L)]
        high=[1 if j==0 else 0 for j in range(L)]
        yield lo,high
    for _ in range(64):
        yield [rng.randrange(base) for _ in range(L)],[rng.randrange(1<<15) for _ in range(L)]

def main():
    lines=[];expected=[]
    for w,L,official in [(1747,55,9699327),(868,28,43646975)]:
        cases=[(g,a,b) for g in [1,official,0xffffffff] for a,b in fixtures(w,L,g)]
        lines.append(f'{w} {len(cases)}')
        for g,a,b in cases:
            lines.append(' '.join(map(str,[g]+a+b)))
            total=sum((x+(y<<32))<<(32*j) for j,(x,y) in enumerate(zip(a,b)))
            expected.append(total%((1<<w)-g))
    with tempfile.TemporaryDirectory() as tmp:
        exe=Path(tmp)/'warp'
        subprocess.run(['c++','-std=c++20','-O2','-pthread',str(Path(__file__).with_name('carry_bridge.cpp')),'-o',exe],check=True)
        result=subprocess.run([exe],input='\n'.join(lines)+'\n',capture_output=True,text=True,check=True)
    actual=[sum(int(x)<<(32*j) for j,x in enumerate(row.split())) for row in result.stdout.splitlines()]
    assert len(actual)==len(expected)
    for i,(a,b) in enumerate(zip(actual,expected)):
        assert a==b,(i,a,b)
    print(f'PASS 100000 warp carry masks and {len(expected)} full-width canonical outputs')

if __name__=='__main__':
    main()
