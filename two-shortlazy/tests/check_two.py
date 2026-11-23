"""Audit the candidate's actual packed input and output matrices and its split."""
from math import prod
from pathlib import Path
import random
import struct
import subprocess
import tempfile

def check(exe,alpha_exe,work,n,w,gap):
    L=(w+31)//32
    split=110 if n==65536 else 55
    enc_pad=128 if n==65536 else 64
    crt_pad=480 if n==65536 else 256
    rng=random.Random(139711)
    q=(1<<w)-gap
    base=1<<(8*split)
    vals=[0,1,base-1,base,base+1,q-1]+[rng.randrange(q) for _ in range(128)]
    words=lambda x,l:[(x>>(32*j))&0xffffffff for j in range(l)]
    cases='\n'.join(' '.join(map(str,words(x,L))) for x in vals)
    result=subprocess.run([exe,str(n)],cwd=work,input=str(len(vals))+'\n'+cases+'\n',
                          capture_output=True,check=True,text=True)
    padding=8*(split%4)
    expected=[words(x%base,enc_pad//4)+words((x//base)<<padding,enc_pad//4) for x in vals]
    assert [list(map(int,row.split())) for row in result.stdout.splitlines()]==expected
    def read(name):
        return (work/name).read_bytes()
    def integers(name):
        return [x[0] for x in struct.iter_unpack('<I',read(name))]
    primes=integers('primes'); count=len(primes); M=prod(primes)
    theta=integers('theta'); inverse_theta=integers('inverse_theta')
    encode=bytearray(8*count*enc_pad)
    for j,p in enumerate(primes):
        assert p%(2*n)==1 and pow(theta[2*j]<<padding,2,p)==(base*base%q)%p
        assert (theta[2*j]<<padding)*inverse_theta[2*j]%p==1
        assert theta[2*j+1]==(theta[2*j]<<32)//p and inverse_theta[2*j+1]==(inverse_theta[2*j]<<32)//p
        for half,length in enumerate((split,(w+7)//8-4*(split//4))):
            for k in range(length):
                weight=pow(256,k,p)*(theta[2*j] if half else 1)%p
                for byte in range(4):
                    tile=(k//32)*(count//2)+j//2
                    word=128*tile+64*half+32*((k%32)//16)+4*(4*(j%2)+byte)+(k%16)//4
                    encode[4*word+k%4]=(weight>>(8*byte))&255
    assert encode==read('encode')
    inv=[pow(2*(M//p),-1,p) for p in primes]
    assert integers('crt_inverse')==inv+inv
    cofactor=[M//p for p in primes]
    weights=cofactor+[base*c for c in cofactor]
    bias=((-count*M*(1+base))%q).to_bytes(4*L,'little')
    assert bias==read('crt_bias')
    decode=bytearray(4*L*crt_pad)
    for j,c in enumerate(weights):
        for byte in range(4):
            weight=((c<<(8*byte))%q).to_bytes(4*L,'little')
            for k,b in enumerate(weight): decode[k*crt_pad+4*j+byte]=b
    assert decode==read('decode')
    full_bound=n*max((base-1)**2+(base*base%q)*((1<<w)//base-1)**2,
                     2*(base-1)*((1<<w)//base-1))
    assert M>4*full_bound
    values=[0,1,-1,full_bound,-full_bound]+[rng.randrange(-full_bound,full_bound+1) for _ in range(100)]
    rows=[]; alpha=[]
    for x in values:
        terms=[(x%p)*pow(c,-1,p)%p for c,p in zip(cofactor,primes)]
        numerator=sum(t*c for t,c in zip(terms,cofactor))
        rounded=(2*numerator+M)//(2*M)
        assert numerator-rounded*M==x
        rows.extend([str(count),' '.join(map(str,primes)),' '.join(map(str,terms))])
        alpha.append(rounded)
    out=subprocess.run([alpha_exe],input='\n'.join(rows)+'\n',capture_output=True,text=True,check=True)
    assert list(map(int,out.stdout.split()))==alpha
    # Check the absorbed quotient against exact signed CRT for both digit channels.
    for lo,hi in zip(values,reversed(values)):
        adjusted=0
        for weight, value in ((1,lo),(base,hi)):
            residues=[value*pow(c,-1,p)%p for c,p in zip(cofactor,primes)]
            full=sum(r*c for r,c in zip(residues,cofactor))
            pivot=(2*full+M)//(2*M)
            assert 0<=pivot<=count and full-pivot*M==value
            adjusted+=weight*sum((r+(p if j>=pivot else 0))*cofactor[j]
                                 for j,(p,r) in enumerate(zip(primes,residues)))
        assert (adjusted+int.from_bytes(bias,'little'))%q==(lo+base*hi)%q
    print(f'PASS N={n}: {len(vals)} input splits, {len(encode)+len(decode)} matrix bytes, and all CRT and twist weights')

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp);exe=d/'bridge'
        subprocess.run(['c++','-std=c++17','-O2',str(Path(__file__).with_name('two_bridge.cpp')),'-o',exe],check=True)
        alpha=d/'alpha'
        subprocess.run(['c++','-std=c++17','-O2',str(Path(__file__).with_name('alpha_bridge.cpp')),'-o',alpha],check=True)
        check(exe,alpha,d,32768,868,43646975)
        check(exe,alpha,d,65536,1747,9699327)
