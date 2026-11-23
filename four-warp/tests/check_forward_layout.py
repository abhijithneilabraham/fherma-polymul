"""Verify warp ownership, bank usage and output sectors for every forward block."""
for n,count in ((32768,16),(65536,30)):
    j=n//32
    seen=set()
    for col in range(4):
        for half in range(2):
            for h in range(2):
                for digit in range(4):
                    words=[]
                    for lane in range(0,32,2):
                        side=(lane%4)//2
                        index=(16*half+lane//4+8*h)^(8*col)^(8*side)
                        words.append(((2*digit+side)*4+col)*32+index)
                    assert len({w%32 for w in words})==16
                    assert seen.isdisjoint(words)
                    seen.update(words)
    assert len(seen)==1024
    for side in range(2):
        for digit in range(4):
            for part in range(4):
                words=[]
                for lane in range(32):
                    col,row=lane%4,lane//4+8*part
                    words.append(((2*digit+side)*4+col)*32+(row^(8*col)^(8*side)))
                assert set(words)<=seen and len({w%32 for w in words})==32
    spectrum=set()
    for base in range(0,j,4):
        for part in range(4):
            words=[]
            for lane in range(32):
                col,row=lane%4,lane//4+8*part
                dst=(row//2)*(2*j)+2*base+4*(row%2)+col
                words.append(dst)
            assert len({w//8 for w in words})==4
            assert spectrum.isdisjoint(words)
            spectrum.update(words)
    assert spectrum==set(range(n))
    print(f"N={n} verified four-sector output writes, full plane coverage and private tile banks")
