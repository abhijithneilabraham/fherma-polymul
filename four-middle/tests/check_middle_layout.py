"""Enumerate the shared banks, input sectors and write coverage for each shape."""
for n in (32768, 65536):
    j = n // 32
    s = j // 256
    for outer in range(0, 32, 2):
        output = set()
        for warp in range(16):
            for chunk in range(s):
                banks, words = [], []
                for tid in range(32*warp, 32*warp+32):
                    side = (tid//4) % 2
                    u = 4*(tid//8) + tid%4
                    logical = chunk*256+u
                    source = outer*j + chunk*512 + tid
                    assert source == outer*j + (logical//4)*8 + side*4 + logical%4
                    words.append(source)
                    bank = u ^ (16*side)
                    banks.append(bank % 32)
                    dst = (logical//4)*128 + (outer+side)*4 + logical%4
                    assert dst not in output
                    output.add(dst)
                assert len({word//8 for word in words}) == 4
                assert len(set(banks)) == 32
        assert len(output) == 2*j
    for side in (0,1):
        for part in range(8):
            assert len({((32*part+lane)^(16*side))%32 for lane in range(32)}) == 32
    print(f"N={n} middle input ownership, output uniqueness and all shared bank mappings passed")
