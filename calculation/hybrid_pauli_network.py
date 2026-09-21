"""X-error probabilities and Z-error characters for the corrected check."""

import itertools
import numpy as np
from .frame_boundary import CodeBoundary
from .incoming_positive import verify_code
from .positive_fault_network import FaultNetwork
from .binary_factors import Network


class HybridFaultNetwork(FaultNetwork):
    def __init__(self,qubits):
        self.network = Network()
        self.wires = {}
        for q in qubits:
            pair = [self.network.variable(f'initial_hybrid_{axis}_{q}') for axis in ('x','z_character')]
            self.wires[q] = pair
            self.network.fixed(pair[0])
        self.pending = {}
        self.events = {'one_qubit':0,'two_qubit':0}
        self.index = 0

    def noise(self,qubits,probabilities):
        probabilities = np.asarray(probabilities)
        if np.any(probabilities < 0) or abs(float(np.sum(probabilities))-1) > 2e-13:
            raise ValueError('Invalid Pauli channel')
        if probabilities.flat[0] == 1 and np.count_nonzero(probabilities) == 1:
            return
        count = len(qubits)
        mixed = np.zeros((2,)*(2*count))
        for bits in itertools.product((0,1),repeat=2*count):
            ex,characters = bits[:count],bits[count:]
            value = 0.
            for ez in itertools.product((0,1),repeat=count):
                labels = tuple(ex[j]+2*ez[j] for j in range(count))
                value += probabilities[labels]*(-1)**sum(characters[j]*ez[j] for j in range(count))
            mixed[bits] = value
        self.mixed_noise(qubits,mixed)

    def mixed_noise(self,qubits,mixed):
        self.index += 1
        count = len(qubits)
        old = [self.wires[q][0] for q in qubits]
        characters = [self.wires[q][1] for q in qubits]
        new = [self.network.variable(f'hybrid_x_{q}_after_noise_{self.index}') for q in qubits]
        table = np.zeros((2,)*(3*count))
        for bits in itertools.product((0,1),repeat=3*count):
            ex = tuple(bits[j] ^ bits[count+j] for j in range(count))
            table[bits] = mixed[ex+bits[2*count:]]
        self.network.add(old+new+characters,table)
        for q,v in zip(qubits,new):
            self.wires[q][0] = v
        self.events['one_qubit' if count == 1 else 'two_qubit'] += 1

    def cnot(self,c,t):
        self.flush((c,t))
        self.index += 1
        xc,vc = self.wires[c]
        xt,vt = self.wires[t]
        ox = self.network.variable(f'hybrid_x_{t}_after_CX_{self.index}')
        ov = self.network.variable(f'hybrid_z_character_{t}_after_CX_{self.index}')
        self.network.xor((ox,xt,xc))
        self.network.xor((ov,vt,vc))
        self.wires[t] = [ox,ov]

    def character_insertion(self,q,variable,parity=0):
        self.flush((q,))
        old = self.wires[q][1]
        new = self.network.variable(f'character_shift_at_{q}')
        self.network.xor((old,new,variable),parity=parity)
        self.wires[q][1] = new


class SourceHybridNetwork(HybridFaultNetwork):
    """Analytical channel kernels preserve structural zeros without subtraction."""
    def source_channel(self,item,p):
        if item['kind'] == 'depol2':
            c,t = item['support']
            mixed = np.zeros((2,2,2,2))
            mixed[0,0,:,:] = 1-16*p/15
            mixed[:,:,0,0] = 4*p/15
            mixed[0,0,0,0] = 1-4*p/5
            for axis,q in enumerate((c,t)):
                if q not in self.pending:
                    continue
                local = self.pending.pop(q)
                updated = np.zeros_like(mixed)
                for x0,x1,v0,v1 in itertools.product((0,1),repeat=4):
                    for e in (0,1):
                        source = (x0 ^ e,x1,v0,v1) if axis == 0 else (x0,x1 ^ e,v0,v1)
                        updated[x0,x1,v0,v1] += mixed[source]*local[e,(v0,v1)[axis]]
                mixed = updated
            self.mixed_noise((c,t),mixed)
            return
        if item['kind'] == 'depol1':
            mixed = np.array([[1-2*p/3,1-4*p/3],[2*p/3,0.]])
        else:
            axis = 'Z' if item['source_kind'] == 'X_READOUT_FLIP' else item['source_kind'][0]
            if axis == 'X':
                mixed = np.array([[1-p,1-p],[p,p]])
            elif axis == 'Z':
                mixed = np.array([[1.,1-2*p],[0.,0.]])
            elif axis == 'Y':
                mixed = np.array([[1-p,1-p],[p,-p]])
            else:
                raise ValueError('Unknown source Pauli axis')
        q, = item['support']
        previous = self.pending.get(q,np.array([[1.,1.],[0.,0.]]))
        self.pending[q] = np.array([previous[0]*mixed[0]+previous[1]*mixed[1],
                                    previous[0]*mixed[1]+previous[1]*mixed[0]])

    def flush(self,qubits):
        for q in qubits:
            if q in self.pending:
                self.mixed_noise((q,),self.pending.pop(q))


def hybrid_network(model,probability,a=0,component=0):
    """Return G_a(y,t), retaining y in little-parity-coordinate order."""
    if component not in (0,1) or not 0 <= probability < .5:
        raise ValueError('Expected t in {0,1} and 0 <= p < 1/2')
    data,ancillas = model['data'],model['specification']['ancillas']
    n,r = len(data),len(model['checks'])
    if a < 0 or a >> n:
        raise ValueError('Invalid incoming X frame')
    verify_code(CodeBoundary(n,model['checks'],model['flips']))
    builder = SourceHybridNetwork(sorted(data+ancillas))
    net = builder.network
    ys = [net.variable(f'incoming_Z_character_{i}') for i in range(r+1)]
    insertions = {}
    for item in model['channels']:
        insertions.setdefault(item['mapped_block_cut'],[]).append(item)
    for cut in range(len(model['block'])+1):
        for item in insertions.get(cut,[]):
            builder.source_channel(item,probability)
        if cut == len(model['block']):
            break
        name,*args = model['block'][cut]
        if name == 'CX':
            builder.cnot(*args)
        elif name == 'POST':
            x,z,sign = args
            if z or sign != 1 or x != 1 << model['specification']['hub']:
                raise ValueError('Unexpected selected check')
            builder.character_insertion(model['specification']['hub'],ys[-1],component)
        elif name not in ('T','T_DAG','RESET'):
            raise ValueError('Unexpected check operation')
    builder.flush(tuple(builder.pending))
    xs = [net.variable(f'hybrid_X_face_{i}') for i in range(r)]
    ws = [net.variable(f'hybrid_kernel_face_{i}') for i in range(r)]
    lx = net.variable('hybrid_X_logical_coset')
    for i,q in enumerate(data):
        x,vz = builder.wires[q]
        incident = [j for j,face in enumerate(model['faces']) if q in face]
        s = net.variable(f'hybrid_code_character_at_{q}')
        product = net.variable(f'hybrid_kernel_product_at_{q}')
        net.xor([s]+[ys[j] for j in incident])
        net.xor((vz,s),parity=component)
        net.xor([x,lx]+[xs[j] for j in incident],parity=a >> i & 1)
        values = np.zeros((2,2,2))
        for bit_x,bit_s in itertools.product((0,1),repeat=2):
            values[bit_x & bit_s,bit_x,bit_s] = 1
        net.add((product,x,s),values)
        net.xor([product]+[ws[j] for j in incident]+([ys[-1]] if a >> i & 1 else []))
        if a >> i & 1:
            phase = np.empty((2,2,2))
            for bit_x,bit_s,bit_c in itertools.product((0,1),repeat=3):
                phase[bit_x,bit_s,bit_c] = (-1)**(bit_x & (bit_s ^ bit_c))
            net.add((x,s,ys[-1]),phase)
    # Fourier expansion of the final ancillary zero-Z constraints. Data
    # characters y remain open; no normalization 2^(-r-1) is applied yet.
    net.multiply_scalar(2.0**(-len(ancillas)))
    net.hybrid_outputs = ys
    net.hybrid_metadata = {'incoming_x':a,'component':component,'open_character_bits':r+1,
        'events':builder.events,'quantity':'G_a(y,t)',
        'inverse_transform':'A=Walsh(G0)/2^(r+1); B=Walsh((G0-G1)/2)/2^(r+1)',
        'scope':'Complete source-window noise for the arbitrary incoming X frame; no physical fault truncation'}
    return net


def tensor_to_words(tensor):
    """Axis zero is the least significant parity bit."""
    return np.transpose(tensor,tuple(reversed(range(tensor.ndim)))).reshape(-1).copy()
