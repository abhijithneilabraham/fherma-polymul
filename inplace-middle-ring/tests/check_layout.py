"""Check each lane's addresses across all blocks of one plane at both sizes."""

def check(n):
    inner_size = n // 32
    producer = bytearray(n)
    consumer = bytearray(n)
    for start in range(0, inner_size, 4):
        for part in range(4):
            addresses = []
            for lane in range(32):
                frequency = lane // 4 + 8 * part
                index = start * 32 + 4 * frequency + lane % 4
                producer[index] += 1
                addresses.append(index)
            assert len({index // 8 for index in addresses}) == 4
    for first in range(0, 32, 2):
        for row in range(inner_size // 256):
            for warp in range(16):
                addresses = []
                for lane in range(32):
                    tid = warp * 32 + lane
                    side = (tid // 4) % 2
                    u = 4 * (tid // 8) + tid % 4
                    column = 256 * row + u
                    address = (column // 4) * 128 + 4 * (first + side) + column % 4
                    consumer[address] += 1
                    addresses.append(address)
                assert len({index // 8 for index in addresses}) == 4
    assert min(producer) == max(producer) == min(consumer) == max(consumer) == 1
    print(f"PASS complete writer and reader partition, four sectors per warp at N={n}")

for size in (32768,65536):
    check(size)
