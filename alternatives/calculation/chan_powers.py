"""Integer power sums of each physical growth channel."""
from collections import defaultdict
from calculation.character_surrogate import canonical_basis

def powers(model,K=5):
    result=defaultdict(lambda:[0]*K)
    for channel in model['channels']:
        span=[0]
        for v in canonical_basis(channel['generator_labels']):span += [x^v for x in span]
        nu=channel['outcome_denominator'];assert 15%nu==0
        for k in range(1,K+1):
            delta=(-15//nu)**k
            uniform,remainder=divmod(15**k-delta,len(span));assert not remainder
            for v in span:result[v][k-1]+=uniform
            result[0][k-1]+=delta
    for k in range(1,K+1):assert sum(v[k-1] for v in result.values())==len(model['channels'])*15**k
    return result

