// Extra endpoint operations; existing scalar and polynomial kernels unchanged.
#include "modular_polynomial.cpp"
#include <cstdint>

extern "C" int msc_endpoint_walsh(U* values,std::size_t entries,unsigned width,
                                    unsigned bits,U p,int inverse) {
    if (!domain(p) || !entries || (entries&(entries-1)) || entries>cap ||
        !width || width>11 || bits>26 || (std::size_t(1)<<bits)>entries ||
        (inverse!=0 && inverse!=1)) return -1;
    Field f(p);
    for (unsigned bit=0;bit<bits;bit++) {
        const std::size_t step=std::size_t(1)<<bit;
        for (std::size_t base=0;base<entries;base+=2*step)
            for (std::size_t j=0;j<step;j++)
                for (unsigned k=0;k<width;k++) {
                    U &a=values[(base+j)*width+k], &b=values[(base+j+step)*width+k];
                    U x=a,y=b;
                    a=f.add(x,y); b=x>=y ? x-y : x+p-y;
                }
    }
    if (inverse) {
        U factor=1;
        for (unsigned i=0;i<bits;i++) factor=(factor&1) ? (factor+p)/2 : factor/2;
        factor=f.encode(factor);
        for (std::size_t i=0;i<entries*width;i++) values[i]=f.mul(values[i],factor);
    }
    return 0;
}

extern "C" int msc_endpoint_first_mix(const U* prefix,const U* check,
        const std::int32_t* tags,const std::uint32_t* syndromes,U* output,
        unsigned first,unsigned count,unsigned width,U p) {
    if (!domain(p) || first+count>256 || !count || !width || width>11) return -1;
    Field f(p); U scales[9],half=1;
    for (unsigned power=0;power<=8;power++) {
        scales[power]=f.encode(half);
        half=(half&1) ? (half+p)/2 : half/2;
    }
    for (unsigned local=0;local<count;local++) {
        const unsigned frame=first+local;
        const U* a=prefix+frame*width;
        for (unsigned j=0;j<4096;j++) {
            if (syndromes[j]>=8) return -2;
            const U* b=check+j*width;
            U product[11]={0};
            for (unsigned k=0;k<width;k++)
                for (unsigned i=0;i<=k;i++) product[k]=f.add(product[k],f.mul(a[i],b[k-i]));
            for (unsigned sx=0;sx<8;sx++) {
                const std::int32_t tag=tags[(std::size_t(local)*4096+j)*8+sx];
                if (tag<0) continue;
                if (tag/4>8) return -3;
                const unsigned sz=syndromes[j]^(frame&7);
                U* out=output+((sx*8+sz)*4+unsigned(tag&3))*width;
                for (unsigned k=0;k<width;k++) out[k]=f.add(out[k],f.mul(product[k],scales[tag/4]));
            }
        }
    }
    return 0;
}
