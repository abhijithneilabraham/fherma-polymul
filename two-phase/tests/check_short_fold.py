"""Compare the actual CRT reducer with Python integer remainder."""
from pathlib import Path
import random
import subprocess
import tempfile

def cases():
    rng = random.Random(346349)
    mask = (1 << 32) - 1
    for width, official_gap in [(1747, 9699327), (868, 43646975)]:
        fence = 1 << width
        for gap in (1, official_gap, 1 << 31, mask):
            q = fence - gap
            for high in (0, 1, mask, (1 << 38) - 1):
                cross = fence - high*gap
                lows = [0, q-1, q, q+1, fence-1, cross-1, cross, cross+1]
                # Force carries that stop in every possible limb position.
                for limb in range(1, (width+31)//32):
                    lows.extend(((1 << (32*limb))-1, (1 << (32*limb))-high*gap))
                lows.extend(rng.randrange(fence) for _ in range(100))
                for low in lows:
                    if 0 <= low < fence:
                        yield width, gap, high*fence + low
            for _ in range(300):
                yield width, gap, rng.randrange(fence << 38)

def main():
    rows, expected = [], []
    for w,gap,x in cases():
        limbs = (w+31)//32 + 2
        rows.append(' '.join(map(str, [w,gap]+[(x >> (32*i)) & 0xffffffff for i in range(limbs)])))
        expected.append(x % ((1 << w)-gap))
    with tempfile.TemporaryDirectory() as directory:
        binary = Path(directory)/'reduce'
        subprocess.run(['c++','-std=c++17','-O2',str(Path(__file__).with_name('fold_bridge.cpp')),'-o',str(binary)],check=True)
        result = subprocess.run([str(binary)],input='\n'.join(rows),text=True,capture_output=True,check=True)
    actual = [sum(int(x) << (32*i) for i,x in enumerate(line.split())) for line in result.stdout.splitlines()]
    assert actual == expected
    print(f'PASS {len(expected)} full-width CRT fold comparisons')

if __name__=='__main__':
    main()
