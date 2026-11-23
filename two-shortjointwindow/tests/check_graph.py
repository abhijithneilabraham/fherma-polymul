"""Check the capture's window dependencies, reuse and final join."""
from pathlib import Path
import re

source = (Path(__file__).resolve().parents[1] / 'solve.cu').read_text()
loop = source.split('for (unsigned product = 0; product < r.batch; ++product) {', 1)[1]
start, group_and_tail = loop.split('for (unsigned group=0;group<2;++group) {', 1)
group, remaining = group_and_tail.split('        }\n', 1)
end, tail = remaining.split('check(cudaGetLastError(), "resident product launches");\n    }', 1)
tail = tail.split('} // namespace', 1)[0]
pattern = re.compile(r'(?:if\s*\((?P<guard>[^()]*)\)\s*)?(?:check\()?'
                     r'(?P<call>cudaStreamWaitEvent|cudaEventRecordWithFlags|cudaEventRecord|'
                     r'two_digit::tile_forward<[^>]*>|bytecrt::launch_g1_\w+|bytecrt::warp_ntt::\w+<[^>]*>)'
                     r'(?P<args>[^;]*);', re.S)

assert 'spectrum = r.forward + group*slot_words' in group
assert 'short_window::words(N, P/2)' in source
assert 'dim3(16,2*live)' in group
assert group.count('r.warp_mods,first,live') == 2
assert 'spectrum,partial,r.warp_inverse' in group
assert 'partial = r.middle + slot*planes' in start
assert 'partial,r.crt_weights' in end


def check(batch):
    streams = {name: set() for name in ('stream', 'transform_stream', 'crt_stream')}
    signals, ancestors = {}, {}

    def execute(text, product, window):
        guards = {None: True, 'product': product != 0, '!product': product == 0,
                  '!product && group==0': product == 0 and window == 0,
                  '!product && group==1': product == 0 and window == 1,
                  'product >= 2': product >= 2}
        for match in pattern.finditer(text):
            guard, call, args = (match[x] for x in ('guard', 'call', 'args'))
            if not guards[guard]:
                continue
            stream, = re.findall(r'r\.(crt_stream|transform_stream|stream)\b', args)
            completed = streams[stream]
            if call.startswith('cuda'):
                if 'spectrum_consumed[group]' in args:
                    event = ('window_consumed', window)
                elif 'middle_consumed[' in args:
                    slot = (batch - 1 if product == batch else product) % 2
                    event = ('partial_consumed', slot)
                else:
                    event = re.search(r'r\.(forward_ready|middle_ready|stages\[\d+\])', args)[1]
                if call == 'cudaStreamWaitEvent':
                    completed.update(signals[event])
                else:
                    signals[event] = completed.copy()
            else:
                if 'tile_forward' in call:
                    stage = 'forward'
                elif '::middle' in call:
                    stage = 'middle'
                else:
                    assert 'launch_g1' in call
                    stage = 'crt'
                node = (product, stage, window if stage in ('forward', 'middle') else None)
                assert node not in ancestors
                ancestors[node] = completed.copy()
                completed.add(node)

    for product in range(batch):
        execute(start, product, None)
        for window in range(2):
            execute(group, product, window)
        execute(end, product, None)
    execute(tail, batch, None)

    assert len(ancestors) == 5*batch
    for product in range(batch):
        crt = (product, 'crt', None)
        assert crt in signals['stages[3]']
        for window in range(2):
            forward = (product, 'forward', window)
            middle = (product, 'middle', window)
            assert forward in ancestors[middle]
            assert middle in ancestors[crt]
            if product:
                assert (product-1, 'middle', window) in ancestors[forward]
            if product >= 2:
                assert (product-2, 'crt', None) in ancestors[middle]
            if product:
                assert (product-1, 'crt', None) not in ancestors[middle]
        assert (product, 'middle', 0) not in ancestors[product, 'forward', 1]
    assert set(ancestors) == signals['stages[3]'] == streams['stream']


for batch in (1, 2, 3, 57, 128):
    check(batch)
print('PASS both forward window lifetimes, partial-spectrum reuse and final join')
