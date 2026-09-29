"""Independent readback of the submitted TikZ; no basis solver is imported.

Exact arithmetic is Z[zeta_8], represented by four integer coefficients.
The only imported input data are frozen graph/source descriptions, compared
with both the actual circuit and actual staged drawings before use.
"""
from pathlib import Path
from functools import lru_cache
import re, json, hashlib, time

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
ZERO = (0,0,0,0)
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def neg(a): return tuple(-x for x in a)
def omega(k):
    k %= 8
    return tuple((1 if k<4 else -1) if j==k%4 else 0 for j in range(4))
def rot(a,k):
    out=ZERO
    for j,c in enumerate(a):
        out=add(out,tuple(c*v for v in omega(j+k)))
    return out
BITS={'I':(0,0),'X':(1,0),'Y':(1,1),'Z':(0,1)}

@lru_cache(None)
def local(kind,theta,labels):
    n=len(labels); mask=(1<<n)-1
    xs=[BITS[p][0] for p in labels]; zs=[BITS[p][1] for p in labels]
    opposite=xs if kind=='Z' else zs
    assert len(set(opposite))==1, (kind,theta,labels,'not a semiweb')
    same=zs if kind=='Z' else xs
    delta=(4*(sum(same)%2)-2*theta*opposite[0])%8
    def vector(t):
        if kind=='Z':
            return [omega(0) if b==0 else omega(t) if b==mask else ZERO for b in range(1<<n)]
        # Common factor 2^(-n/2) cancels in the local identity.
        return [add(omega(0),rot(omega(t),4*(b.bit_count()%2))) for b in range(1<<n)]
    v,w=vector(theta),vector(theta+delta)
    x=sum(a<<j for j,a in enumerate(xs)); z=sum(a<<j for j,a in enumerate(zs))
    moved=[ZERO]*(1<<n)
    for b,a in enumerate(v):
        moved[b^x]=rot(a,2*(x&z).bit_count()+4*((b&z).bit_count()%2))
    scalars=[s for s in range(8) if moved==[rot(a,s) for a in w]]
    assert len(scalars)==1, (kind,theta,labels,delta,scalars)
    return delta,scalars[0]

def rank(rows):
    piv={}
    for r in rows:
        while r:
            k=r.bit_length()-1
            if k not in piv:
                piv[k]=r; break
            r^=piv[k]
    return len(piv)

NRE=re.compile(r'\\node \[style=(none|[XZ](?: phase)? dot)\] \((n\d+)\) at \(([^,]+),([^\)]+)\) \{(.*?)\};')
ERE=re.compile(r'\\draw(?: \[style=([XYZ]) Web\])? \((n\d+)\.center\) to \((n\d+)\.center\);')
PH={'':0,r'$\frac{\pi}{4}$':1,r'$-\frac{\pi}{4}$':-1}
def parse(path):
    text=path.read_text()
    nodes={k:{'kind':'boundary' if s=='none' else s[0], 'phase':PH[label],
              'x':float(x),'y':float(y)} for s,k,x,y,label in NRE.findall(text)}
    edges=[(a,b,p or 'I') for p,a,b in ERE.findall(text)]
    assert len(edges)==len({(a,b) for a,b,p in edges})
    meta=[json.loads(m) for m in re.findall(r'^% defect: (.*)$',text,re.M)]
    return nodes,edges,meta,text

def angle(t):
    return {0:'0',1:r'\frac{\pi}{4}',-1:r'-\frac{\pi}{4}',2:r'\frac{\pi}{2}',-2:r'-\frac{\pi}{2}',3:r'\frac{3\pi}{4}',-3:r'-\frac{3\pi}{4}',4:r'\pi'}[t]

def source_check(s):
    stage=s['stage']; graph=s['graph']; path=ROOT/stage['source_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==stage['source_sha256']
    lines=path.read_text().splitlines()
    lo,hi=stage['source_lines']; parsed=[]
    for line in lines[lo-1:hi]:
        m=re.fullmatch(r'(RX|MX|CX|T|T_DAG)(?:\([^)]*\))? (.*)',line)
        if not m: continue
        name,operands=m.groups(); q=list(map(int,operands.split()))
        if name in ('T','T_DAG'):
            if not parsed or parsed[-1][0]!='PHASE': parsed.append(('PHASE',{}))
            parsed[-1][1].update({str(i):1 if name=='T' else -1 for i in q})
        else: parsed.append((name,q))
    expected=[(g['name'],g.get('phases',g.get('support'))) for g in stage['groups']]
    assert parsed==expected,('source group mismatch',stage['distance'])
    # Reconstruct graph connectivity from physical time-ordered source operations.
    ns=graph['nodes']; used=set(); last={}; edges=[]
    def node(gi,q,kind,role,phase=0):
        candidates=[k for k,n in ns.items() if n['x']==gi and n['qubit']==q and n['kind']==kind and n['role']==role and n['phase']==phase]
        assert len(candidates)==1,(gi,q,kind,role)
        k=candidates[0]; assert k not in used; used.add(k); return k
    for q in stage['data']: last[q]=node(0,q,'boundary','input')
    middle=next(i for i,g in enumerate(parsed) if g[0]=='MX')
    for gi,(name,op) in enumerate(parsed):
        if name=='RX':
            for q in op:
                assert q not in last; last[q]=node(gi,q,'Z','prepare')
        elif name=='PHASE':
            for q,p in sorted((int(q),p) for q,p in op.items()):
                k=node(gi,q,'Z','entry' if gi<middle else 'exit',p)
                edges.append((last[q],k));last[q]=k
        elif name=='CX':
            for c,t in zip(op[::2],op[1::2]):
                a=node(gi,c,'Z','control');b=node(gi,t,'X','target')
                edges.extend([(last[c],a),(last[t],b),(a,b)]);last[c]=a;last[t]=b
        else:
            for q in op:
                k=node(gi,q,'Z','measure');edges.append((last.pop(q),k))
    for q in stage['data']:
        k=node(len(parsed)-1,q,'boundary','output');edges.append((last.pop(q),k))
    assert not last and used==set(ns)
    assert edges==[(e['u'],e['v']) for e in graph['edges']]

def main(output=None):
    start=time.process_time()
    data=json.loads((ROOT/'data/basis.json').read_text())
    pages=(PAPER/'drawings/pages.tex').read_text()
    pageblocks={m.group(1):m.group(2) for m in re.finditer(r'\\pdfbookmark\[2\].*?\{(d[35]_\d+)\}(.*?)(?=\\pdfbookmark\[2\]|\Z)',pages,re.S)}
    reports=[]; checks=0; y_corrections=0; star_count=0; hashes={}
    audited={}
    for s in data['stages']:
        source_check(s)
        stage=s['stage'];g=s['graph'];ns=g['nodes']; canonical=[(e['u'],e['v']) for e in g['edges']]
        nbits=2*len(canonical);inc={k:[] for k in ns}
        for j,(a,b) in enumerate(canonical):inc[a].append(j);inc[b].append(j)
        rows=[];web_extra=[];input_rows=[];closed_rows=[]
        for k,node in ns.items():
            ix=inc[k]
            if node['kind']=='boundary':
                assert len(ix)==1
                pair=[1<<(2*ix[0]),1<<(2*ix[0]+1)]
                closed_rows+=pair
                if node['role']=='input': input_rows+=pair
                continue
            opp=0 if node['kind']=='Z' else 1
            for j in ix[1:]: rows.append((1<<(2*ix[0]+opp))^(1<<(2*j+opp)))
            if node['phase']%2: web_extra.append(1<<(2*ix[0]+opp))
            assert node['phase'] in (-1,0,1)
            web_extra.append(sum(1<<(2*j+1-opp) for j in ix))
        vectors=[];parts={'incoming':[],'output':[],'internal':[]}
        for record in s['generators']:
            path=PAPER/'drawings/basis'/(record['id']+'.tikz')
            nodes,ed,metadata,text=parse(path)
            assert set(nodes)==set(ns),record['id']
            for k,n in nodes.items():
                assert n['kind']==ns[k]['kind'] and n['phase']==ns[k]['phase']
                assert n['y']==ns[k]['y']
            em={(a,b):p for a,b,p in ed}
            assert set(em)==set(canonical)
            # Each physical wire has the original node order, despite spread CNOT columns.
            for q in stage['wires']:
                sequence=sorted([k for k,n in ns.items() if n['qubit']==q],key=lambda k:(ns[k]['x'],int(k[1:])))
                assert all(nodes[a]['x']<=nodes[b]['x'] for a,b in zip(sequence,sequence[1:]))
            labels=[em[e] for e in canonical]
            assert labels==record['edge_labels']
            v=sum((BITS[p][0]+2*BITS[p][1])<<(2*j) for j,p in enumerate(labels))
            assert all((v&r).bit_count()%2==0 for r in rows)
            vectors.append(v);parts[record['kind']].append(v)
            def boundary(role):
                return {str(n['qubit']):labels[inc[k][0]] for k,n in ns.items() if n['role']==role and labels[inc[k][0]]!='I'}
            assert boundary('input')==record['input'] and boundary('output')==record['output']
            exponent=0; ds={}
            for k,n in ns.items():
                if n['kind']=='boundary':continue
                delta,scalar=local(n['kind'],n['phase'],tuple(labels[j] for j in inc[k]))
                if delta: ds[k]=delta
                exponent+=scalar; checks+=1
            for j,(a,b) in enumerate(canonical):
                if labels[j]=='Y' and not any(ns[k]['role']=='output' for k in (a,b)):
                    exponent+=4;y_corrections+=1
            assert ds=={k:v%8 for k,v in record['defects'].items()}
            assert exponent%8==record['scalar_pi_over_4']
            assert {m['node']:m['delta_pi_over_4']%8 for m in metadata}==ds
            assert len(metadata)==len(ds)
            for i,m in enumerate(metadata,1):
                assert m['marker']==i and m['phase_pi_over_4']==ns[m['node']]['phase']
                assert m['qubit']==ns[m['node']]['qubit'] and m['role']==ns[m['node']]['role']
                assert (m['changed_phase_pi_over_4']-m['phase_pi_over_4'])%8==m['delta_pi_over_4']%8
                assert re.search(r'\(defect-'+m['node']+r'\).*?\(\$\('+m['node']+r'\.center\)',text)
                assert r'\ast_{'+str(i)+'}' in text
            star_count+=len(ds)
            block=pageblocks[record['id']]
            for name,role in [('in','input'),('out','output')]:
                label=''.join(f'{p}_{{{q}}}' for q,p in sorted(boundary(role).items(),key=lambda x:int(x[0]))) or 'I'
                assert f'P_{{\\rm {name}}}={label}' in block,(record['id'],role,label)
            scalar={0:'1',1:r'e^{i\frac{\pi}{4}}',2:'i',3:r'e^{i\frac{3\pi}{4}}',4:'-1',5:r'e^{i\frac{5\pi}{4}}',6:'-i',7:r'e^{i\frac{7\pi}{4}}'}[exponent%8]
            # Accept equivalent typographical forms, but not another phase.
            printed=re.search(r'\\lambda=(.*?)\$',block).group(1)
            assert printed==scalar,(record['id'],printed,scalar)
            for m in metadata:
                assert f"\\delta_{{{m['marker']}}}={angle(m['delta_pi_over_4'])}" in block
            audited[record['id']]={'labels':labels,'defects':ds,'scalar':exponent%8}
            hashes[str(path.relative_to(PAPER))]=hashlib.sha256(path.read_bytes()).hexdigest()
        dims={'distance':stage['distance'],'nodes':len(ns),'edges':len(canonical),
              'semiweb_dimension':nbits-rank(rows),'basis_rank':rank(vectors),
              'no_boundary_dimension':nbits-rank(rows+closed_rows),
              'no_input_dimension':nbits-rank(rows+input_rows),
              'web_dimension':nbits-rank(rows+web_extra),
              'no_boundary_web_dimension':nbits-rank(rows+web_extra+closed_rows),
              'blocks':{k:len(v) for k,v in parts.items()},'source_sha256':stage['source_sha256']}
        assert dims['basis_rank']==dims['semiweb_dimension']==len(vectors)
        assert rank(parts['internal'])==dims['no_boundary_dimension']
        assert all((v&r).bit_count()%2==0 for v in parts['internal'] for r in closed_rows)
        assert all((v&r).bit_count()%2==0 for v in parts['output'] for r in input_rows)
        assert (dims['semiweb_dimension'],dims['no_boundary_dimension'],dims['web_dimension'],dims['no_boundary_web_dimension'])=={3:(90,62,12,6),5:(264,188,37,19)}[stage['distance']]
        reports.append(dims)
    # Main example and full-circuit cancellation figures must use these same graphs.
    s=data['stages'][0];g=s['graph'];ns=g['nodes'];inc={k:[] for k in ns}
    edges=[(e['u'],e['v']) for e in g['edges']]
    for j,(a,b) in enumerate(edges):inc[a].append(j);inc[b].append(j)
    example={}
    for name in ['x7','x7_cancellation','x0_cancellation','x0x7_cancellation','z0','z3','z0z3']:
        n,ed,metadata,text=parse(PAPER/'drawings/examples'/(name+'.tikz'))
        assert set(n)==set(ns)
        assert all(n[k]['phase']==ns[k]['phase'] and n[k]['kind']==ns[k]['kind'] for k in ns)
        em={(a,b):p for a,b,p in ed}; assert set(em)==set(edges)
        labels=[em[e] for e in edges];ds={};ex=0
        for k,node in ns.items():
            if node['kind']=='boundary':continue
            de,se=local(node['kind'],node['phase'],tuple(labels[j] for j in inc[k]))
            if de:ds[k]=de
            ex+=se
        ex+=4*sum(p=='Y' and all(ns[k]['role']!='output' for k in e) for e,p in zip(edges,labels))
        assert ds=={m['node']:m['delta_pi_over_4']%8 for m in metadata}
        example[name]={'defects':ds,'scalar':ex%8,'labels':labels}
    assert example['x7']['labels']==audited['d3_005']['labels']
    for a,b,c in [('x7_cancellation','x0_cancellation','x0x7_cancellation'),('z0','z3','z0z3')]:
        for pa,pb,pc in zip(example[a]['labels'],example[b]['labels'],example[c]['labels']):
            assert tuple(x^y for x,y in zip(BITS[pa],BITS[pb]))==BITS[pc]
    assert not example['z0z3']['defects'] and example['z0z3']['scalar']==0
    assert example['x7']['scalar']==5
    result={'status':'pass','stages':reports,'exact_local_tensor_checks':checks,
            'distinct_exact_local_checks':local.cache_info().currsize,
            'Y_transpose_signs_checked':y_corrections,'defect_markers_checked':star_count,
            'all_354_page_boundary_labels_scalars_and_defects_match':True,
            'examples':{k:{a:b for a,b in v.items() if a!='labels'} for k,v in example.items()},
            'cpu_seconds':time.process_time()-start,'sha256':hashes}
    if output is not None:
        Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    print(json.dumps(main(),indent=2))
