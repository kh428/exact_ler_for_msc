"""Exact growth products that accumulate one Cauchy coefficient at a time."""
import ctypes
from .native_build import build_library
import hashlib
from pathlib import Path
import numpy as np
from .native_modular_growth import GrowthField
ROOT=Path(__file__).resolve().parents[1]


class PolynomialGrowthField(GrowthField):
    def __init__(self,field,build_dir=None):
        path, self.evidence = build_library("modular_polynomial_growth.cpp", build_dir)
        self.field = field
        self.library = ctypes.CDLL(str(path))
        ptr=field.ptr;size=ctypes.c_size_t;uint=ctypes.c_uint;u64=ctypes.c_uint64
        self.library.msc_growth_permute.argtypes=[ptr,size,ptr,uint,ctypes.c_int]
        self.library.msc_growth_walsh.argtypes=[ptr,size,uint,u64,ctypes.c_int]
        self.library.msc_growth_accumulate_products.argtypes=[ptr,size,ptr,size,ptr,size,
                                                               ptr,ptr,ptr,ptr,uint,uint,u64,ctypes.c_int]
        self.library.msc_growth_project_polynomial.argtypes=[ptr,size,uint,
            ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_uint16),ptr,size,uint,u64]
        for name in ('msc_growth_permute','msc_growth_walsh','msc_growth_accumulate_products',
                     'msc_growth_project_polynomial'):
            getattr(self.library,name).restype=ctypes.c_int

    def join(self,*args,**kwargs):
        raise TypeError('Use the explicit coefficient-streamed join schedule')

    def products(self,left,right,output,shape,reset):
        g,q,d=shape['g'],shape['q'],shape['d']
        assert left.size==1<<len(shape['up']) and right.size==1<<len(shape['vp']) and output.size==1<<d
        assert all(len(shape[k])==q for k in ('ac','bc','cc','pc'))
        columns=[np.asarray(shape[k],dtype=np.uint64) for k in ('ac','bc','cc','pc')]
        status=self.library.msc_growth_accumulate_products(self.field.pointer(left),left.size,
            self.field.pointer(right),right.size,self.field.pointer(output),output.size,
            *[self.field.pointer(v) for v in columns],g,q,self.field.prime,int(reset))
        if status:raise ValueError('Coefficient product rejected: '+str(status))

    def project(self,values,destinations,tags,bits,character):
        assert values.ndim==2 and values.dtype==np.uint64 and values.flags.c_contiguous
        assert destinations.dtype==np.int32 and tags.dtype==np.uint16
        assert destinations.flags.c_contiguous and tags.flags.c_contiguous
        assert len(values)==len(destinations)==len(tags)
        output=np.empty((1<<bits,values.shape[1]),dtype=np.uint64)
        status=self.library.msc_growth_project_polynomial(self.field.pointer(values),len(values),
            values.shape[1],destinations.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            tags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            self.field.pointer(output),len(output),character,self.field.prime)
        if status:raise ValueError('Polynomial leaf projection rejected: '+str(status))
        return output
