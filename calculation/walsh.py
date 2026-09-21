"""In-place Walsh transforms on ordinary prime-field residues."""
import numpy as np

def walsh_low(values,bits,prime,inverse=False,chunk=1<<16):
    assert values.dtype==np.uint64 and 0<=bits<=len(values).bit_length()-1 and 2<prime<1<<62
    p=np.uint64(prime);one=np.uint64(1)
    def update(a,b):
        left=a.copy();plus=left+b;minus=left+p-b
        if inverse:
            a[:]=((plus+(plus&one)*p)>>one)%p;b[:]=((minus+(minus&one)*p)>>one)%p
        else:
            a[:]=plus%p;b[:]=minus%p
    for bit in range(bits):
        width=1<<bit
        if width<=chunk:
            view=values.reshape(-1,2*width);rows=max(1,chunk//width)
            for start in range(0,len(view),rows):update(view[start:start+rows,:width],view[start:start+rows,width:])
        else:
            for base in range(0,len(values),2*width):
                for off in range(0,width,chunk):
                    size=min(chunk,width-off);update(values[base+off:base+off+size],values[base+width+off:base+width+off+size])
