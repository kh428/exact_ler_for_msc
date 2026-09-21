"""Small ctypes wrapper for the exact quadratic Gaussian C++ kernel."""
import ctypes as ct
from pathlib import Path
import sys
import numpy as np
from calculation.cyclotomic import omega
from calculation.dyadic import make


def dyadic_descriptor(h,j):
    if j==-1:return omega(0)*0
    assert j>=0,(h,j)
    v=omega(j)*make(0,1,0) if h%2 else omega(j)
    k=h//2
    return v*(1<<k) if k>=0 else v/(1<<-k)


class Kernel:
    def __init__(self,plan,obs,library=None):
        self.words=max(1,(plan['label_bits']+63)//64)
        self.lib=ct.CDLL(str(library))
        u=ct.POINTER(ct.c_uint64);i=ct.POINTER(ct.c_int);s=ct.POINTER(ct.c_int16)
        self.lib.phase_make.argtypes=[ct.c_int]*5+[u,u,i,ct.c_int,u,i]
        self.lib.phase_make.restype=ct.c_void_p
        self.lib.phase_evaluate.argtypes=[ct.c_void_p,u,ct.c_uint64,s]
        self.lib.phase_delete.argtypes=[ct.c_void_p]
        row=plan['outputs'][obs];q=np.array([m for m,e,c in row['phase']],dtype=np.uint64)
        errors=self.encode([e for m,e,c in row['phase']]);coeff=np.array([c for m,e,c in row['phase']],dtype=np.int32)
        constraints=self.encode([e for e,b in row['constraints']]);offset=np.array([b for e,b in row['constraints']],dtype=np.int32)
        self.ptr=self.lib.phase_make(row['n'],self.words,row['normalization_power'],row['normalization_phase'],len(q),
            q.ctypes.data_as(u),errors.ctypes.data_as(u),coeff.ctypes.data_as(i),len(offset),constraints.ctypes.data_as(u),offset.ctypes.data_as(i))
        if not self.ptr:raise ValueError('Not a supported quadratic phase plan')
    def encode(self,labels):
        return np.array([[(int(v)>>(64*j))&((1<<64)-1) for j in range(self.words)] for v in labels],dtype=np.uint64).reshape((-1,self.words))
    def descriptors(self,labels):
        a=self.encode(labels);out=np.empty((len(a),2),dtype=np.int16)
        self.lib.phase_evaluate(self.ptr,a.ctypes.data_as(ct.POINTER(ct.c_uint64)),len(a),out.ctypes.data_as(ct.POINTER(ct.c_int16)))
        assert not np.any(out[:,1]<-1)
        return out
    def exact(self,labels):
        return [dyadic_descriptor(int(h),int(j)) for h,j in self.descriptors(labels)]
    def close(self):
        if self.ptr:self.lib.phase_delete(self.ptr);self.ptr=None
    def __del__(self):
        if hasattr(self,'ptr'):self.close()


def pair_sum(kernels,labels,weights,begin=0,end=None):
    a,x,y=kernels;array=a.encode(labels);weights=np.array(weights,dtype=np.uint64)
    count=len(labels);end=count if end is None else min(count,end)
    limbs=np.zeros(8192,dtype=np.uint64);stats=np.zeros(5,dtype=np.uint64)
    u=ct.POINTER(ct.c_uint64)
    a.lib.phase_pair_sum.argtypes=[ct.c_void_p]*3+[u,u,ct.c_uint64,ct.c_uint64,ct.c_uint64,u,u]
    a.lib.phase_pair_sum.restype=ct.c_int
    code=a.lib.phase_pair_sum(a.ptr,x.ptr,y.ptr,array.ctypes.data_as(u),weights.ctypes.data_as(u),
        count,begin,end,limbs.ctypes.data_as(u),stats.ctypes.data_as(u))
    if code:raise RuntimeError('Exact pair kernel guard '+str(code))
    result=omega(0)*0
    for i in np.nonzero(limbs.reshape((-1,2)).any(axis=1))[0]:
        value=int(limbs[2*i])+(int(limbs[2*i+1])<<64)
        if value>>127:value-=1<<128
        h=int(i)//8-256;j=int(i)%8
        result+=dyadic_descriptor(h,j)*value
    return result,dict(pairs=int(stats[0]),accepted=int(stats[1]),not_pure_good=int(stats[2]),
                      first_bad_pair=None if stats[3]==np.iinfo(np.uint64).max else [int(stats[3]),int(stats[4])])


class PreparedSource:
    def __init__(self,kernel,labels):
        self.kernel=kernel;self.array=kernel.encode(labels) if not isinstance(labels,np.ndarray) else np.ascontiguousarray(labels,dtype=np.uint64)
        u=ct.POINTER(ct.c_uint64);lib=kernel.lib
        lib.phase_source_make.argtypes=[ct.c_void_p,u,ct.c_uint64];lib.phase_source_make.restype=ct.c_void_p
        lib.phase_source_delete.argtypes=[ct.c_void_p]
        self.ptr=lib.phase_source_make(kernel.ptr,self.array.ctypes.data_as(u),len(self.array))
        if not self.ptr:raise ValueError('Source preparation requires the fast phase form')
        self.memory_estimate=len(self.array)*(1400+8*kernel.words)
    def close(self):
        if self.ptr:self.kernel.lib.phase_source_delete(self.ptr);self.ptr=None
    def __del__(self):
        if hasattr(self,'ptr'):self.close()


def shifted_cross(kernels,left,right,lw,rw,shifts,sw,begin=0,end=None,mode=0):
    a,x,y=kernels;lw=np.ascontiguousarray(lw,dtype=np.uint64);rw=np.ascontiguousarray(rw,dtype=np.uint64)
    shifts=a.encode(shifts) if not isinstance(shifts,np.ndarray) else np.ascontiguousarray(shifts,dtype=np.uint64)
    sw=np.ascontiguousarray(sw,dtype=np.uint64);end=len(left.array) if end is None else end
    limbs=np.zeros(8192,dtype=np.uint64);stats=np.zeros(3,dtype=np.uint64);u=ct.POINTER(ct.c_uint64)
    a.lib.phase_shifted_cross_sum.argtypes=[ct.c_void_p]*5+[u]*4+[ct.c_uint64]*3+[ct.c_int,u,u]
    a.lib.phase_shifted_cross_sum.restype=ct.c_int
    code=a.lib.phase_shifted_cross_sum(a.ptr,x.ptr,y.ptr,left.ptr,right.ptr,lw.ctypes.data_as(u),rw.ctypes.data_as(u),
        shifts.ctypes.data_as(u),sw.ctypes.data_as(u),len(shifts),begin,end,mode,limbs.ctypes.data_as(u),stats.ctypes.data_as(u))
    if code:raise RuntimeError('Shifted exact kernel guard '+str(code))
    result=omega(0)*0
    for i in np.nonzero(limbs.reshape((-1,2)).any(axis=1))[0]:
        value=int(limbs[2*i])+(int(limbs[2*i+1])<<64)
        if value>>127:value-=1<<128
        result+=dyadic_descriptor(int(i)//8-256,int(i)%8)*value
    return result,dict(pairs=int(stats[0]),accepted=int(stats[1]),not_pure_good=int(stats[2]))
