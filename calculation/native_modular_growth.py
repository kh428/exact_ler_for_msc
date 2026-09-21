"""Destructive finite-field joins for the frozen binary-subspace growth plan."""
import ctypes
from .native_build import build_library
import hashlib
import json
from pathlib import Path
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


class GrowthField:
    def __init__(self,field,build_dir=None):
        self.field=field
        path, self.evidence = build_library("modular_growth.cpp", build_dir)
        self.library = ctypes.CDLL(str(path))
        ptr=field.ptr;size=ctypes.c_size_t;uint=ctypes.c_uint;u64=ctypes.c_uint64
        self.library.msc_growth_permute.argtypes=[ptr,size,ptr,uint,ctypes.c_int]
        self.library.msc_growth_walsh.argtypes=[ptr,size,uint,u64,ctypes.c_int]
        self.library.msc_growth_products.argtypes=[ptr,size,ptr,size,ptr,size,ptr,ptr,ptr,ptr,uint,uint,u64]
        for name in ('msc_growth_permute','msc_growth_walsh','msc_growth_products'):
            getattr(self.library,name).restype=ctypes.c_int

    def permute(self,values,columns,inverse=False):
        columns=np.asarray(columns,dtype=np.uint64)
        status=self.library.msc_growth_permute(self.field.pointer(values),values.size,self.field.pointer(columns),len(columns),int(inverse))
        if status:raise ValueError(f'Exact coordinate permutation refused: {status}')

    def walsh(self,values,bits,inverse=False):
        status=self.library.msc_growth_walsh(self.field.pointer(values),values.size,bits,self.field.prime,int(inverse))
        if status:raise ValueError(f'Exact Walsh transform refused: {status}')

    def join(self,left,right,shape,*,max_live_bytes=3072*2**20):
        g,q,d=shape['g'],shape['q'],shape['d']
        if max(len(shape['up']),len(shape['vp']),d)>26 or q>32:
            raise ValueError('Native growth pilot dimensions exceeded')
        if left.size!=1<<len(shape['up']) or right.size!=1<<len(shape['vp']):
            raise ValueError('Growth input dimensions disagree with shape')
        if max(left.size,right.size,1<<d)//8+left.nbytes+right.nbytes+8*(1<<d)>max_live_bytes:
            raise ValueError('Local growth join exceeds its declared live-array allowance')
        assert all(len(shape[k])==q for k in ('ac','bc','cc','pc')) and len(shape['cp'])==d
        began=time.perf_counter()
        self.permute(left,shape['up']);self.permute(right,shape['vp'])
        self.walsh(left,g);self.walsh(right,g)
        output=np.empty(1<<d,dtype=np.uint64)
        columns=[np.asarray(shape[k],dtype=np.uint64) for k in ('ac','bc','cc','pc')]
        status=self.library.msc_growth_products(self.field.pointer(left),left.size,self.field.pointer(right),right.size,
            self.field.pointer(output),output.size,*[self.field.pointer(x) for x in columns],g,q,self.field.prime)
        if status:raise ValueError(f'Exact restricted product refused: {status}')
        self.walsh(output,g,inverse=True);self.permute(output,shape['cp'],inverse=True)
        return output,{'seconds':time.perf_counter()-began,'products':1<<q,'walsh_additions':g*(left.size+right.size+output.size),
                       'output_bits':d,'arithmetic':'Exact 62-bit Montgomery residues; destructive child transforms'}
