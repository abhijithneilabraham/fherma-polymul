"""Substitute the sole PTX matrix instruction for the cooperative CPU test."""
from pathlib import Path
import re
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
s=(root/'two-encode.cuh').read_text()
assert s.count('extern __shared__ uint32_t storage[];') == 1
s=s.replace('extern __shared__ uint32_t storage[];', 'alignas(128) static uint32_t storage[tile_bytes<N,Pad>()/sizeof(uint32_t)];')
pat=r'asm volatile\(\s*"mma\.sync\.aligned\.m16n8k32\.row\.col\.s32\.u8\.u8\.s32 ".*?\);'
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
    subprocess.run([exe],check=True,timeout=180)
