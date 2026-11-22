#!/usr/bin/env python3
"""Compile the actual paired-lane helper against an independent integer oracle."""
from pathlib import Path
import subprocess
import tempfile


def check():
    root = Path(__file__).resolve().parents[1]
    source = (root / 'core/rns-encode-mma.cuh').read_text()
    begin = source.index('__device__ __forceinline__ uint32_t encode_mma_reduce(')
    end = source.index('\n}\n', begin) + 2
    with tempfile.TemporaryDirectory(prefix='warp-encoding-') as temporary:
        build = Path(temporary)
        (build / 'encode-reducer.inc').write_text(
            'namespace bytecrt::detail {\n' + source[begin:end] + '\n}\n')
        binary = build / 'warp-contract'
        subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                        '-I', str(build), str(root / 'tests/warp_contract.cpp'),
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=120)


if __name__ == '__main__':
    check()
