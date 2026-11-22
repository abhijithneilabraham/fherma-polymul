#!/usr/bin/env python3
"""Compile the actual paired-lane helper against an independent integer oracle."""
from pathlib import Path
import subprocess
import tempfile


def check():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='warp-encoding-') as temporary:
        build = Path(temporary)
        binary = build / 'warp-contract'
        subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                        '-I', str(build), str(root / 'tests/warp_contract.cpp'),
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=120)


if __name__ == '__main__':
    check()
