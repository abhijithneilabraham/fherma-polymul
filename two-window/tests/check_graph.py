"""Check the captured graph's dependencies at each reuse of the two workspaces."""
from pathlib import Path
import re

source = (Path(__file__).resolve().parents[1] / 'solve.cu').read_text()
loop = source.split('for (unsigned product = 0; product < r.batch; ++product) {',1)[1]
start, rest = loop.split('for (unsigned group=0;group<groups;++group) {',1)
group, rest = rest.split('        }\n',1)
end, tail = rest.split('check(cudaGetLastError(), "resident product launches");\n    }',1)
tail = tail.split('} // namespace',1)[0]
pattern = re.compile(r'(?:if\s*\((?P<guard>[^()]*)\)\s*)?(?:check\()?'
                     r'(?P<call>cudaStreamWaitEvent|cudaEventRecordWithFlags|cudaEventRecord|'
                     r'two_digit::tile_forward<[^>]*>|bytecrt::inverse_crt<[^>]*>|bytecrt::warp_ntt::middle<[^>]*>)'
                     r'(?P<args>[^;]*);',re.S)
for line in ('const unsigned step=product*groups+group;', 'const unsigned window=step%2;',
             'uint32_t* spectrum=r.forward+window*slot_words;'):
    assert line in group
assert 'unsigned slot=product%2;' in start and 'uint32_t* partial=r.middle+slot*planes;' in start
assert '2*count' in group and 'r.warp_mods,first,count' in group
assert 'r.warp_mods,first' in group and 'partial,r.warp_roots,r.warp_mods' in end


def check(batch, groups):
    streams = {name:set() for name in ('stream','transform_stream','crt_stream')}
    signals, ancestors = {}, {}

    def execute(text, product, step):
        guards = {None:True, '!product':product==0, 'product >= 2':product>=2, 'step >= 2':step>=2}
        next_slot = min(batch-1,product)%2
        for m in pattern.finditer(text):
            guard, call, args = (m[x] for x in ('guard','call','args'))
            if not guards[guard.strip() if guard else None]:
                continue
            stream, = re.findall(r'r\.(crt_stream|transform_stream|stream)\b',args)
            ordered = streams[stream]
            if call.startswith('cuda'):
                if 'spectrum_consumed[window]' in args:
                    event = ('spectrum',step%2)
                elif 'partial_consumed[' in args:
                    event = ('partial',next_slot)
                    assert '[slot]' in args or '[(r.batch-1)%2]' in args
                else:
                    event = re.search(r'r\.(forward_ready|partial_ready|stages\[\d+\])',args)[1]
                if call=='cudaStreamWaitEvent':
                    ordered.update(signals[event])
                else:
                    signals[event] = ordered.copy()
            else:
                stage = 'forward' if 'tile_forward' in call else 'middle' if '::middle' in call else 'crt'
                node = (stage,product if stage=='crt' else step)
                assert node not in ancestors
                ancestors[node] = ordered.copy()
                ordered.add(node)

    for product in range(batch):
        execute(start,product,product*groups)
        for j in range(groups):
            execute(group,product,product*groups+j)
        execute(end,product,(product+1)*groups-1)
    execute(tail,batch,batch*groups)
    for step in range(batch*groups):
        forward, middle = ('forward',step), ('middle',step)
        assert forward in ancestors[middle]
        assert middle in ancestors[('crt',step//groups)]
        if step>=2:
            assert ('middle',step-2) in ancestors[forward]
        if step//groups>=2:
            assert ('crt',step//groups-2) in ancestors[middle]
        if step:
            assert ('middle',step-1) not in ancestors[forward]
    assert set(ancestors)==signals['stages[3]']==streams['stream']

for groups in (3,5):
    for batch in (1,2,3,57,128):
        check(batch,groups)
print('PASS both prime-group counts, spectrum and inverse slot lifetimes, and final join')
