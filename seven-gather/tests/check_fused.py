"""Substitute the sole PTX matrix instruction for the cooperative CPU test."""
from pathlib import Path
import re
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
s=(root/'seven-encode.cuh').read_text()
s=s.replace('#include "core/rns-encode-mma.cuh"','')
assert s.count('extern __shared__ U storage[];') == 1
s=s.replace('extern __shared__ U storage[];', 'alignas(128) static U storage[shared_bytes<W>()/sizeof(U)];')
pat=r'asm volatile\("mma\.sync\.aligned\.m16n8k32\.row\.col\.s32\.u8\.u8\.s32 ".*?\);'
matches=re.findall(pat,s,re.S)
assert len(matches)==1
args=re.findall(r'"\+?r"\(([^)]+)\)',matches[0])
assert args==['c[0]','c[1]','c[2]','c[3]','a0','a1','a2','a3','b0','b1']
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    (tmp/'emulated-tile.hpp').write_text(re.sub(pat,'emulate_mma(c,a0,a1,a2,a3,b0,b1);',s,flags=re.S))
    exe=tmp/'check'
    subprocess.run(['c++','-O2','-std=c++17','-Wno-deprecated-declarations',
                    '-I',tmp,'-I',root,root/'tests/fused_tile.cpp','-o',exe],check=True)
    subprocess.run([exe],check=True,timeout=90)
