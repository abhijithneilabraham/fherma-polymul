"""Exercise the dependency order in the submitted enqueue loop on the CPU."""
from pathlib import Path
import re

src = (Path(__file__).resolve().parents[1] / 'solve.cu').read_text()
loop = src.split('for (unsigned product = 0; product < r.batch; ++product) {',1)[1]
body, tail = loop.split('check(cudaGetLastError(), "resident product launches");',1)
tail = tail.split('} // namespace',1)[0]
calls = re.compile(r'(?:if\s*\((?P<guard>[^()]*)\)\s*)?(?:check\()?'
                   r'(?P<call>cudaStreamWaitEvent|cudaEventRecordWithFlags|cudaEventRecord|'
                   r'two_digit::encode<[^>]*>|bytecrt::launch_\w+|bytecrt::warp_ntt::\w+<[^>]*>)'
                   r'(?P<args>[^;]*);',re.S)

def scan(text):
    result=[]
    for m in calls.finditer(text):
        args=m['args']
        streams=re.findall(r'r\.(crt_stream|transform_stream|forward_stream|stream)\b',args)
        assert len(streams)==1, (m['call'],args)
        result.append((m['guard'],m['call'],streams[0],args))
    return result

assert 'const unsigned slot=product%2;' in body
assert 'uint32_t* encoded=r.encoded+size_t(slot)*2*planes;' in body
assert 'std::min(2u,r->batch)*2*bytes' in src

def check(batch):
    frontier={s:set() for s in ('stream','transform_stream','crt_stream','forward_stream')}
    events={}
    ancestors={}
    labels=[]
    def execute(ops, product):
        for guard,call,stream,args in ops:
            if guard:
                assert guard in ('!product','product','product == 1','product >= 2'), guard
                if not {'!product':product==0,'product':product!=0,'product == 1':product==1,'product >= 2':product>=2}[guard]:continue
            if call.startswith('cuda'):
                event = re.search(r'r\.(encoded_ready|encoded_consumed|first_forward_start|forward_ready|spectrum_consumed|first_middle_start|weighted_ready|weighted_consumed|stages\[\d\]|first_crt_start|next_encode_start|next_encode_end)',args)[1]
                if event=='encoded_consumed':
                    assert '[slot]' in args
                    event+=str(product%2)
                if call=='cudaStreamWaitEvent':frontier[stream].update(events[event])
                else:events[event]=frontier[stream].copy()
            else:
                if 'two_digit::encode' in call:
                    if 'r.a+offset,encoded,' in args:stage='encode_a'
                    else:
                        assert 'r.b+offset,encoded+planes,' in args
                        stage='encode_b'
                elif 'g1_crt' in call:
                    assert 'r.weighted' in args and 'r.out+offset' in args
                    stage='crt'
                else:
                    stage=call.split('::')[-1].split('<')[0]
                    if stage=='forward':assert 'encoded,r.forward' in args
                    if stage=='inverse': assert 'r.middle,r.weighted' in args
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
            for node in ('encode_a','encode_b'):
                assert (i-1,'forward') not in ancestors[i,node]
                assert (i,node) not in ancestors[i-1,'forward']
            if i>=2:assert (i-2,'forward') in ancestors[i,'encode_a']
            assert (i-1,'middle') in ancestors[i,'forward']
            assert (i-1,'inverse') in ancestors[i,'middle']
            assert (i-1,'crt') in ancestors[i,'inverse']
            for earlier in ('middle','inverse'):
                for encoder in ('encode_a','encode_b'):
                    assert (i-1,earlier) not in ancestors[i,encoder]
                    assert (i,encoder) not in ancestors[i-1,earlier]
            for stage in stages[:4]:
                assert (i-1,'crt') not in ancestors[i,stage]
                assert (i,stage) not in ancestors[i-1,'crt']
        assert (i,'crt') in events['stages[6]']
    assert set(labels)==frontier['stream']
for b in (1,2,3,57,128):check(b)
print('PASS actual enqueue dependency order for 1, 2, 3, 57 and 128 products')
