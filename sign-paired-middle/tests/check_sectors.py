"""Enumerate 32-byte sector requests at the two sides of reconstruction."""

def sectors(words):
    return len({word // 8 for word in words})

for n in (32768, 65536):
    inner = n // 32
    for outer in range(32):
        for start in range(0, inner, 16):
            paired = [2 * ((i // 2) * 64 + outer * 2 + i % 2) + sign
                      for i in range(start,start+16) for sign in range(2)]
            control = [(i // 4) * 128 + (outer // 2 * 2 + side) * 4 + i % 4
                       for i in range(start,start+16) for side in range(2)]
            assert sectors(control) == 4 and sectors(paired) == 8
    for i in range(inner):
        control = [(i // 4) * 128 + lane * 4 + i % 4 for lane in range(32)]
        paired = [2 * ((i // 2) * 64 + lane * 2 + i % 2) + sign
                  for lane in range(32) for sign in range(2)]
        assert sectors(control) == 16 and sectors(paired) == 16
    print(f"PASS N={n}, sectors per 32 words: store 4->8, load 16->8")
