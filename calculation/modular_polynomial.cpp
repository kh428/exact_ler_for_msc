// Reuse the independently validated scalar Montgomery implementation.
// Both this source and the included source are authenticated by the loader.
#include "modular_binary.cpp"

extern "C" int msc_poly_convert(U* data,std::size_t n,U p,int encode) {
    if (!domain(p) || n>11*cap || (encode!=0 && encode!=1)) return -1;
    Field f(p);
    for (std::size_t i=0;i<n;i++) {
        if (data[i]>=p) return -2;
        data[i]=encode ? f.encode(data[i]) : f.reduce(data[i]);
    }
    return 0;
}

extern "C" int msc_poly_add(U* dst,const U* src,std::size_t n,U p) {
    if (!domain(p) || n>11*cap) return -1;
    Field f(p);
    for (std::size_t i=0;i<n;i++) dst[i]=f.add(dst[i],src[i]);
    return 0;
}

extern "C" int msc_poly_broadcast(U* dst,const U* src,unsigned bits,
        std::size_t source_entries,const U* flips,U fixed,unsigned width,
        unsigned old_degree,unsigned source_degree,U p) {
    if (!domain(p) || bits>26 || !source_entries || source_entries>cap ||
        (source_entries&(source_entries-1)) || fixed>=source_entries ||
        !width || width>11 || old_degree>=width || source_degree>10) return -1;
    for (unsigned i=0;i<bits;i++) if (flips[i]>=source_entries) return -2;
    const unsigned new_degree = (old_degree+source_degree<width)
                               ? old_degree+source_degree : width-1;
    Field f(p);
    const std::size_t n=std::size_t(1)<<bits;
    U index=fixed;
    for (std::size_t i=0;i<n;i++) {
        if (i) index ^= flips[__builtin_ctzll(i)];
        U* a=dst+i*width;
        const U* b=src+index*(source_degree+1);
        // Descending degrees preserve all old a coefficients needed later.
        for (int k=int(new_degree);k>=0;k--) {
            const unsigned lo = unsigned(k)>old_degree ? unsigned(k)-old_degree : 0;
            const unsigned hi = unsigned(k)<source_degree ? unsigned(k) : source_degree;
            U value=0;
            for (unsigned j=lo;j<=hi;j++) value=f.add(value,f.mul(a[k-j],b[j]));
            a[k]=value;
        }
    }
    return 0;
}
