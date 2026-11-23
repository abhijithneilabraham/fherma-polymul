"""Check the middle writer and both decoder address maps on all coefficients."""
for n, primes in ((32768, 10), (65536, 18)):
    visited = set()
    parent_sectors = quad_sectors = 0
    for group in range(n // 128):
        packets = [4 * (32 * group + lane) for lane in range(32)]
        assert all(word % 4 == 0 for word in packets)
        group_words = []
        for column in range(4):
            old = [(group * 4 + column) // 4 * 128 + lane * 4 + column for lane in range(32)]
            new = [packet + column for packet in packets]
            assert old == new
            parent_sectors += len({word // 8 for word in old})
            group_words.extend(new)
        quad_sectors += len({word // 8 for word in group_words})
        assert not visited.intersection(group_words)
        visited.update(group_words)
    assert visited == set(range(n))
    assert parent_sectors == 4 * quad_sectors
    factor = 32 * 7 * primes * 128 / (1 << 30)
    print(f"N={n} decoder input: {parent_sectors*factor:g} to {quad_sectors*factor:g} GiB of requested sectors")
