"""Direct gate-level ZX circuit for the old four-fault pattern in both repairs.

The fault-free first stage is replaced by its ideal encoded output. Source
growth/final gates, accepted records and noiseless readout are used explicitly.
"""
import json
import stim
from zx.circuits import encoder
from calculation.css_boundary import nullspace
from .results import DATA

def xor(values):
    result=0
    for v in values:result^=v
    return result


def phase_gates(qubit,phase):
    phase%=8
    if phase==0:return []
    if phase==7:return [f'T_DAG {qubit}']
    if phase==1:return [f'T {qubit}']
    return [f'T {qubit}']*phase


def build(case, record_index, with_faults, logical_outcome):
    directory = DATA / "inputs" / case
    program=json.loads((directory/'model.json').read_text())
    first=json.loads((directory/'window0.json').read_text())
    last=json.loads((directory/'window1.json').read_text())
    candidate=json.loads((DATA/'certificates'/case/'fault_pattern.json').read_text())
    start=first['source_event_stop_exclusive'];stop=last['source_event_stop_exclusive']
    events=program['events']
    records=[e[-1] for e in events[:stop] if e[0]=='MEASURE']
    n=len(records);assert records==list(range(n))
    constraints=[xor(1<<i for i in d) for d in program['detectors'] if all(i<n for i in d)]
    free=nullspace(constraints,n);assert len(free)==4 and 0<=record_index<16
    outcomes_word=xor(v for j,v in enumerate(free) if record_index>>j&1)
    assert not outcomes_word&((1<<sum(e[0]=='MEASURE' for e in events[:start]))-1)
    # All terminal syndrome offsets and the logical observable's earlier-record
    # offset vanish on the accepted record space. Thus final positive-code and
    # logical projections can be implemented by decoding and single-site reads.
    for d in program['detectors']:
        if any(i>=n for i in d):
            earlier=xor(1<<i for i in d if i<n)
            assert all(not (earlier&v).bit_count()%2 for v in free)
    earlier=xor(1<<i for row in program['observables'] for i in row if i<n)
    assert all(not (earlier&v).bit_count()%2 for v in free)
    # The source's terminal logical measurement is D X_all D^dagger, with
    # precisely the same D as the final exit layer.
    suffix=events[stop:];size=len(last['data']);all_data=sum(1<<q for q in last['data'])
    assert suffix[:size]==last['specification']['entry']
    assert suffix[size][:3]==['MEASURE_PAULI',all_data,0]
    assert suffix[size+1:2*size+1]==last['specification']['exit']

    enc,_,tab3=encoder(first['data'],first['faces'])
    _,dec,tab5=encoder(last['data'],last['faces'])
    for tab,size in ((tab3,7),(tab5,19)):
        X=tab.inverse()(stim.PauliString('X'*size))
        Z=tab.inverse()(stim.PauliString('Z'*size))
        assert X.sign==1 and X[size-1]==1 and all(X[i] in (0,3) for i in range(size-1))
        assert Z==stim.PauliString('_'*(size-1)+'Z')
    lines=['R '+' '.join(map(str,first['data'])),f'H {first["data"][-1]}']
    lines+=phase_gates(first['data'][-1],first['target_logical_phase'])+enc
    selected={f['event_index']:f for f in candidate['growth_faults']+[candidate['final_fault']]}
    measurement_bits=[];seen_faults=[]
    for i in range(start,stop):
        e=events[i];name=e[0]
        if name=='NOISE':
            if with_faults and i in selected:
                f=selected[i];code=f['outcome']['code']
                assert f['source_kind']=='DEPOLARIZE2'
                for j,q in enumerate(f['support']):
                    p=('I','X','Z','Y')[(code>>(2*j))&3]
                    if p!='I':lines.append(f'{p} {q}')
                seen_faults.append(i)
        elif name=='RESET':
            q,axis,sign=e[1:];assert sign==1 and axis in ('X','Z')
            lines.append(('RX' if axis=='X' else 'R')+f' {q}')
        elif name=='MEASURE':
            q,axis,record=e[1:];assert axis in ('X','Z')
            lines.append(('MX' if axis=='X' else 'M')+f' {q}')
            measurement_bits.append((outcomes_word>>record)&1)
        elif name=='FEEDBACK':
            axis,q,record=e[1:]
            if outcomes_word>>record&1:lines.append(f'{axis} {q}')
        elif name in ('CX','H','T','T_DAG','S','S_DAG','X','Y','Z'):
            lines.append(name+' '+' '.join(map(str,e[1:])))
        else:raise ValueError(('Unsupported actual source operation',i,e))
    assert seen_faults==(sorted(selected) if with_faults else [])
    lines+=dec
    lines.append('M '+' '.join(map(str,last['data'][:-1])))
    measurement_bits += [0]*(len(last['data'])-1)
    lines+=phase_gates(last['data'][-1],-last['target_logical_phase'])
    lines.append(f'MX {last["data"][-1]}');measurement_bits.append(logical_outcome)
    metadata=dict(record_index=record_index,source_record_word=str(outcomes_word),
                  accepted_record_dimension=len(free),measurement_outcomes=measurement_bits,
                  source_sha256=program['source_sha256'],fault_event_indices=seen_faults,
                  source_event_range=[start,stop],first_stage_fault_free=True,
                  terminal_record_offsets_vanish=True,
                  scope='All growth and final double-checking source gates, ideal preceding encoded output, noiseless code/logical readout; no escape.')
    return '\n'.join(lines)+'\n',metadata

