"""Exercise the dependency order in the submitted enqueue loop on the CPU."""
from pathlib import Path
import re

src = (Path(__file__).resolve().parents[1] / 'solve.cu').read_text()
loop = src.split('for (unsigned product = 0; product < r.batch; ++product) {',1)[1]
body, tail = loop.split('check(cudaGetLastError(), "resident product launches");',1)
tail = tail.split('} // namespace',1)[0]
calls = re.compile(r'(?:if\s*\((?P<guard>[^()]*)\)\s*)?(?:check\()?'
                   r'(?P<call>cudaStreamWaitEvent|cudaEventRecordWithFlags|cudaEventRecord|'
                   r'bytecrt::launch_\w+|bytecrt::warp_ntt::\w+<[^>]*>)'
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
    frontier={'stream':set(),'crt_stream':set()}
    events={}
    ancestors={}
    labels=[]
    def execute(ops, product):
        for guard,call,stream,args in ops:
            if guard:
                assert guard in ('!product','product','product == 1'), guard
                if not {'!product':product==0,'product':product!=0,'product == 1':product==1}[guard]:continue
            if call.startswith('cuda'):
                event = re.search(r'r\.(weighted_ready|weighted_consumed|stages\[\d\]|first_crt_start|next_encode_start|next_encode_end)',args)[1]
                if call=='cudaStreamWaitEvent':frontier[stream].update(events[event])
                else:events[event]=frontier[stream].copy()
            else:
                if 'encode_mma' in call:stage='encode_a' if 'r.a + offset' in args else 'encode_b'
                elif 'g1_crt' in call:
                    assert 's.weighted' in args and 'r.out + offset' in args
                    stage='crt'
                else:
                    stage=call.split('::')[-1].split('<')[0]
                    if stage=='inverse': assert 's.weighted' in args
                node=(product,stage)
                assert node not in ancestors
                ancestors[node]=frontier[stream].copy()
                frontier[stream].add(node)
                labels.append(node)
    for product in range(batch): execute(scan(body),product)
    execute(scan(tail),batch)
    assert len(labels)==6*batch
    for i in range(batch):
        stages=['encode_a','encode_b','forward','middle','inverse','crt']
        for a,b in zip(stages,stages[1:]):assert (i,a) in ancestors[i,b]
        if i:
            assert (i-1,'crt') in ancestors[i,'inverse']
            for stage in stages[:4]:
                assert (i-1,'crt') not in ancestors[i,stage]
                assert (i,stage) not in ancestors[i-1,'crt']
        assert (i,'crt') in events['stages[6]']
    assert set(labels)==frontier['stream']
for b in (1,2,3,57,128):check(b)
print('PASS actual enqueue dependency order for 1, 2, 3, 57 and 128 products')
