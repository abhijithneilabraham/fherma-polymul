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
        streams=re.findall(r'r\.(crt_stream|transform_stream|a_stream|stream)\b',args)
        assert len(streams)==1, (m['call'],args)
        result.append((m['guard'],m['call'],streams[0],args))
    return result

def check(batch):
    frontier={s:set() for s in ('stream','transform_stream','crt_stream','a_stream')}
    events={}
    ancestors={}
    labels=[]
    def execute(ops, product):
        for guard,call,stream,args in ops:
            if guard:
                assert guard in ('!product','product','product == 1'), guard
                if not {'!product':product==0,'product':product!=0,'product == 1':product==1}[guard]:continue
            if call.startswith('cuda'):
                event = re.search(r'r\.(encoded_a_ready|forward_a_done|first_a_start|first_a_end|first_b_start|forward_ready|spectrum_consumed|first_middle_start|weighted_ready|weighted_consumed|stages\[\d\]|first_crt_start|next_encode_start|next_encode_end)',args)[1]
                if call=='cudaStreamWaitEvent':frontier[stream].update(events[event])
                else:events[event]=frontier[stream].copy()
            else:
                if 'encode_mma' in call:stage='encode_a' if 'r.a + offset' in args else 'encode_b'
                elif 'g1_crt' in call:
                    assert 's.weighted' in args and 'r.out + offset' in args
                    stage='crt'
                else:
                    stage=call.split('::')[-1].split('<')[0]
                    if stage=='forward':
                        assert 'dim3(32,P)' in args
                        if 's.ab_t+planes,s.ab+planes' in args:stage='forward_b'
                        else:
                            assert 's.ab_t,s.ab,' in args
                            stage='forward_a'
                    if stage=='inverse': assert 's.weighted' in args
                node=(product,stage)
                assert node not in ancestors
                ancestors[node]=frontier[stream].copy()
                frontier[stream].add(node)
                labels.append(node)
    for product in range(batch): execute(scan(body),product)
    execute(scan(tail),batch)
    assert len(labels)==7*batch
    for i in range(batch):
        def precedes(a,b):
            assert (i,a) in ancestors[i,b], (i,a,b)
        def concurrent(a,b):
            assert (i,a) not in ancestors[i,b] and (i,b) not in ancestors[i,a]
        for operand in ('a','b'):
            precedes('encode_'+operand,'forward_'+operand)
            precedes('forward_'+operand,'middle')
        concurrent('forward_a','encode_b')
        concurrent('forward_a','forward_b')
        precedes('middle','inverse')
        precedes('inverse','crt')
        if i:
            for operand in ('a','b'):
                assert (i-1,'forward_'+operand) in ancestors[i,'encode_'+operand]
                assert (i-1,'middle') in ancestors[i,'forward_'+operand]
            assert (i-1,'inverse') in ancestors[i,'middle']
            assert (i-1,'crt') in ancestors[i,'inverse']
            for stage in ('encode_a','encode_b','forward_a','forward_b','middle'):
                assert (i-1,'crt') not in ancestors[i,stage]
                assert (i,stage) not in ancestors[i-1,'crt']
        assert (i,'crt') in events['stages[6]']
    assert set(labels)==frontier['stream']
for b in (1,2,3,57,128):check(b)
print('PASS actual enqueue dependency order for 1, 2, 3, 57 and 128 products')
