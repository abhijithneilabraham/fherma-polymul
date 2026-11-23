"""Substitute the sole PTX matrix instruction for the cooperative CPU test."""
from pathlib import Path
import re
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
s=(root/'two-encode.cuh').read_text()
pat=r'asm volatile\("\bmma\.sync\.aligned\.m16n8k32\.row\.col\.s32\.u8\.u8\.s32 ".*?\);'
matches=re.findall(pat,s,re.S)
assert len(matches)==1
args=re.findall(r'"\+?r"\(([^)]+)\)',matches[0])
assert args==['c[0]','c[1]','c[2]','c[3]','a0','a1','a2','a3','b0','b1']
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    (tmp/'emulated-tile.hpp').write_text(re.sub(pat,'emulate_mma(c,a0,a1,a2,a3,b0,b1);',s,flags=re.S))
    tensor_source=(root/'tensor-warp-ops.cuh').read_text()
    tensor_calls=re.findall(pat,tensor_source,re.S)
    assert len(tensor_calls)==1
    tensor_args=re.findall(r'"\+?r"\(([^)]+)\)',tensor_calls[0])
    assert tensor_args==['c[0]','c[1]','c[2]','c[3]','a[0]','a[1]','a[2]','a[3]','b[0]','b[4]']
    (tmp/'tensor-warp-ops.cuh').write_text(re.sub(
        pat,'emulate_mma(c,a[0],a[1],a[2],a[3],b[0],b[4]);',tensor_source,flags=re.S))
    stages=(root/'warp-stages.cuh').read_text()
    (tmp/'emulated-stages.hpp').write_text(stages)
    exe=tmp/'check'
    subprocess.run(['c++','-O2','-std=c++17','-Wno-deprecated-declarations',
                    '-I',tmp,'-I',root,root/'tests/fused_tile.cpp','-o',exe],check=True)
    subprocess.run([exe],check=True,timeout=90)
