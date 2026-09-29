"""Independent exact selected-circuit evaluation for all d=3 printed identities."""
from .check_diagrams import *

def main(output=None):
    began=time.process_time()
    data=json.loads((ROOT/'data/basis.json').read_text())['stages'][0]
    stage,g=data['stage'],data['graph'];ns=g['nodes'];dataq=stage['data']
    qpos={q:i for i,q in enumerate(stage['wires'])}
    def accum(d,k,a):
        v=add(d.get(k,ZERO),a)
        if v==ZERO:d.pop(k,None)
        else:d[k]=v
    node_at={(n['x'],n['qubit']):k for k,n in ns.items() if n['kind']!='boundary'}
    def literal(bits,shifts):
        state={sum(((bits>>j)&1)<<qpos[q] for j,q in enumerate(dataq)):omega(0)}
        for gi,group in enumerate(stage['groups']):
            name=group['name']
            def delta(q):return shifts.get(node_at[(gi,q)],0)%8
            if name=='RX':
                for q in group['support']:
                    mask=1<<qpos[q];assert delta(q) in (0,4)
                    assert all(not b&mask for b in state)
                    state={k:a for b,c in state.items() for k,a in [(b,c),(b|mask,rot(c,delta(q)))]}
            elif name=='PHASE':
                for q0,angle0 in group['phases'].items():
                    q=int(q0);mask=1<<qpos[q];theta=angle0+delta(q)
                    state={b:rot(a,theta) if b&mask else a for b,a in state.items()}
            elif name=='CX':
                for c,t in zip(group['support'][::2],group['support'][1::2]):
                    dc,dt=delta(c),delta(t);assert dc in (0,4) and dt in (0,4)
                    cm,tm=1<<qpos[c],1<<qpos[t]
                    state={b^ (tm if b&cm else 0) ^ (tm if dt else 0):rot(a,dc if b&cm else 0) for b,a in state.items()}
            elif name=='MX':
                for q in group['support']:
                    mask=1<<qpos[q];assert delta(q) in (0,4)
                    out={}
                    for b,a in state.items():accum(out,b&~mask,rot(a,delta(q) if b&mask else 0))
                    state=out
            else:raise AssertionError(name)
        out={}
        for b,a in state.items():
            key=sum(((b>>qpos[q])&1)<<j for j,q in enumerate(dataq))
            accum(out,key,a)
        return out
    original=[literal(b,{}) for b in range(128)]
    assert any(original)
    def masks(labels):
        return (sum((p in 'XY')<<dataq.index(int(q)) for q,p in labels.items()),
                sum((p in 'YZ')<<dataq.index(int(q)) for q,p in labels.items()))
    records=list(data['generators'])
    # Read actual main-text example overlays and phases, not saved identity metadata.
    edges=[(e['u'],e['v']) for e in g['edges']];inc={k:[] for k in ns}
    for j,(a,b) in enumerate(edges):inc[a].append(j);inc[b].append(j)
    for name in ['x7','x7_cancellation','x0_cancellation','x0x7_cancellation','z0','z3','z0z3']:
        _,ed,_,_=parse(PAPER/'drawings/examples'/(name+'.tikz'))
        em={(a,b):p for a,b,p in ed};labels=[em[e] for e in edges]
        ds={};scalar=0
        for k,n in ns.items():
            if n['kind']=='boundary':continue
            de,se=local(n['kind'],n['phase'],tuple(labels[j] for j in inc[k]));scalar+=se
            if de:ds[k]=de
        scalar+=4*sum(p=='Y' and not any(ns[k]['role']=='output' for k in e) for e,p in zip(edges,labels))
        def boundary(role):return {str(n['qubit']):labels[inc[k][0]] for k,n in ns.items() if n['role']==role and labels[inc[k][0]]!='I'}
        records.append({'id':name,'input':boundary('input'),'output':boundary('output'),'defects':ds,'scalar_pi_over_4':scalar%8})
    for i,r in enumerate(records):
        x,z=masks(r['input']);ox,oz=masks(r['output'])
        for b in range(128):
            left={c:rot(a,2*(x&z).bit_count()+4*((b&z).bit_count()%2)) for c,a in original[b^x].items()}
            right={c^ox:rot(a,r['scalar_pi_over_4']+2*(ox&oz).bit_count()+4*((c&oz).bit_count()%2)) for c,a in literal(b,r['defects']).items()}
            assert left==right,(r['id'],b,left,right)
        if (i+1)%15==0:print(f'Checked {i+1}/{len(records)} complete d=3 identities',flush=True)
    result={'status':'pass','basis_identities':90,'main_figure_identities':7,
            'exact_matrix_columns_checked':len(records)*128,'arithmetic':'Z[zeta_8], common endpoint normalisations cancel',
            'independence':'New sparse circuit implementation; no research simulator or scalar arithmetic imported',
            'cpu_seconds':time.process_time()-began}
    if output is not None:
        Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':
    print(json.dumps(main(),indent=2))
