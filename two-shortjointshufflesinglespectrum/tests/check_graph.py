"""Exercise the dependency order in the submitted enqueue loop on the CPU."""
from pathlib import Path
import re

src = (Path(__file__).resolve().parents[1] / 'solve.cu').read_text()
loop = src.split('for (unsigned product = 0; product < r.batch; ++product) {',1)[1]
body, tail = loop.split('check(cudaGetLastError(), "resident product launches");',1)
tail = tail.split('} // namespace',1)[0]
calls = re.compile(r'(?:if\s*\((?P<guard>[^()]*)\)\s*)?(?:check\()?'
                   r'(?P<call>cudaStreamWaitEvent|cudaEventRecordWithFlags|cudaEventRecord|'
                   r'two_digit::tile_forward<[^>]*>|bytecrt::launch_\w+|bytecrt::warp_ntt::\w+<[^>]*>)'
                   r'(?P<args>[^;]*);',re.S)

def scan(text):
    result=[]
    for m in calls.finditer(text):
        args=m['args']
        streams=re.findall(r'r\.(crt_stream|stream)\b',args)
        assert len(streams)==1, (m['call'],args)
        result.append((m['guard'],m['call'],streams[0],args))
    return result

def check(batch):
    frontier={s:set() for s in ('stream','crt_stream')}
    events={}
    ancestors={}
    labels=[]
    def execute(ops, product):
        for guard,call,stream,args in ops:
            if guard:
                assert guard in ('!product','product','product >= 2'), guard
                if not {'!product':product==0,'product':product!=0,'product >= 2':product>=2}[guard]:continue
            if call.startswith('cuda'):
                if 'middle_consumed[' in args:
                    slot=(batch-1 if product==batch else product)%2
                    event=f'middle_consumed[{slot}]'
                else:
                    event=re.search(r'r\.(middle_ready|stages\[\d\])',args)[1]
                if call=='cudaStreamWaitEvent':frontier[stream].update(events[event])
                else:events[event]=frontier[stream].copy()
            else:
                if 'tile_forward' in call:
                    assert 'r.a+offset,r.b+offset,spectrum' in args
                    stage='forward'
                elif 'launch_g1_crt' in call:
                    assert 'partial,r.crt_weights' in args and 'r.out+offset' in args
                    stage='crt'
                else:
                    stage=call.split('::')[-1].split('<')[0]
                    if stage=='inverse': assert 'r.middle,r.weighted' in args
                node=(product,stage)
                assert node not in ancestors
                ancestors[node]=frontier[stream].copy()
                frontier[stream].add(node)
                labels.append(node)
    for product in range(batch): execute(scan(body),product)
    execute(scan(tail),batch)
    assert len(labels)==3*batch
    for i in range(batch):
        stages=['forward','middle','crt']
        for a,b in zip(stages,stages[1:]):assert (i,a) in ancestors[i,b]
        if i:
            assert (i-1,'middle') in ancestors[i,'forward']
            for earlier,later in [('crt','middle'),('crt','forward')]:
                assert (i-1,earlier) not in ancestors[i,later]
                assert (i,later) not in ancestors[i-1,earlier]
        if i>=2:
            assert (i-2,'middle') in ancestors[i,'forward']
            assert (i-2,'crt') in ancestors[i,'middle']
        assert (i,'crt') in events['stages[4]']
    assert set(labels)==frontier['stream']
assert 'partial=r.middle+slot*planes;' in body and 'unsigned slot=product%2;' in body
assert 'spectrum,partial,r.warp_inverse' in body and 'r.middle_consumed[(r.batch-1)%2]' in tail
assert 'spectrum=r.forward;' in body
assert 'cudaMalloc(&r->forward,2*bytes)' in src
for b in (1,2,3,57,128):check(b)
print('PASS actual enqueue dependency order for 1, 2, 3, 57 and 128 products')
