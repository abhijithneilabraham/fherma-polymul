"""All 128 folded planes versus a full-width negacyclic product."""
from pathlib import Path
from random import Random
import subprocess
import tempfile

n,w=65536,1747
q=(1<<w)-9699327
rng=Random(612657)
a=[rng.randrange(q) for _ in range(n)]
a[:4]=[0,1,q-1,q-2]
b=[q-1,rng.randrange(q),rng.randrange(q)]
expected=[(a[i]*b[0]+(a[i-1] if i else -a[-1])*b[1]+
           (a[0] if i==n-1 else -a[i+1])*b[2])%q for i in range(n)]
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);exe=root/'ring'
    subprocess.run(['c++','-std=c++17','-O2',str(Path(__file__).with_name('ring.cpp')),'-o',exe],check=True)
    (root/'input').write_bytes(b''.join(x.to_bytes(220,'little') for x in a+b))
    subprocess.run([exe,root/'input',root/'output'],check=True,timeout=240)
    out=(root/'output').read_bytes()
    assert len(out)==n*220
    for i,x in enumerate(expected):assert int.from_bytes(out[220*i:220*(i+1)],'little')==x,i
print('PASS 65536 full-width coefficients across all 128 folded odd-root planes and signed wraparound')
