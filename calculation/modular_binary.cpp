// Bounded single-thread binary-factor arithmetic. All stored values use
// Montgomery form a*2^64 mod p, with 2<p<2^62. No floating-point operations.
#include <cstddef>
#include <cstdint>

using U = std::uint64_t;
using Wide = unsigned __int128;
constexpr std::size_t cap = std::size_t(1) << 26;

struct Field {
    U p, neg_inv;
    explicit Field(U q): p(q) {
        U inv = 1;
        for (int i=0; i<6; i++) inv *= U(2)-p*inv;
        neg_inv = U(0)-inv;
    }
    U reduce(Wide x) const {
        U m = U(x)*neg_inv;
        // x<p^2<2^124, m*p<2^126, so the addition cannot overflow u128.
        U y = U((x+Wide(m)*p) >> 64);
        return y>=p ? y-p : y;
    }
    U mul(U a,U b) const {return reduce(Wide(a)*b);}
    U add(U a,U b) const {U y=a+b; return y>=p ? y-p : y;}
    U encode(U a) const {return U((Wide(a)<<64)%p);}
};
bool domain(U p) {return p>2 && p<(U(1)<<62) && (p&1);}

extern "C" int msc_binary_convert(U* data,std::size_t n,U p,int encode) {
    if (!domain(p) || n>cap || (encode!=0 && encode!=1)) return -1;
    Field f(p);
    for (std::size_t i=0;i<n;i++) {
        if (data[i]>=p) return -2;
        data[i]=encode ? f.encode(data[i]) : f.reduce(data[i]);
    }
    return 0;
}

extern "C" int msc_binary_add(U* dst,const U* src,std::size_t n,U p) {
    if (!domain(p) || n>cap) return -1;
    Field f(p);
    for (std::size_t i=0;i<n;i++) dst[i]=f.add(dst[i],src[i]);
    return 0;
}

extern "C" int msc_binary_broadcast(U* dst,const U* src,unsigned bits,
                                       std::size_t source_entries,const U* flips,
                                       U fixed,U p) {
    if (!domain(p) || bits>26 || source_entries==0 || source_entries>cap ||
        (source_entries&(source_entries-1)) || fixed>=source_entries) return -1;
    for (unsigned i=0;i<bits;i++) if (flips[i]>=source_entries) return -2;
    // Wrapper supplies cumulative XORs of source strides for each low-bit
    // carry. Every index stays inside the source's power-of-two allocation.
    Field f(p);
    const std::size_t n=std::size_t(1)<<bits;
    U index=fixed;
    dst[0]=f.mul(dst[0],src[index]);
    for (std::size_t i=1;i<n;i++) {
        index ^= flips[__builtin_ctzll(i)];
        dst[i]=f.mul(dst[i],src[index]);
    }
    return 0;
}
