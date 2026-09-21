// Exact bounded subspace-convolution kernels for the audited growth plan.
// Inputs/outputs are Montgomery residues. No floating-point operations.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
using U=std::uint64_t;
using Wide=unsigned __int128;
constexpr std::size_t cap=std::size_t(1)<<26;
struct Field {
    U p,neg_inv;
    explicit Field(U q):p(q){U inv=1;for(int i=0;i<6;i++)inv*=U(2)-p*inv;neg_inv=U(0)-inv;}
    U mul(U a,U b)const{Wide x=Wide(a)*b;U m=U(x)*neg_inv;U y=U((x+Wide(m)*p)>>64);return y>=p?y-p:y;}
    U add(U a,U b)const{U x=a+b;return x>=p?x-p:x;}
    U sub(U a,U b)const{return a>=b?a-b:a+p-b;}
    U half(U a)const{return (a+((a&1)?p:0))>>1;}
};
bool domain(U p){return p>2 && p<(U(1)<<62) && (p&1);}
bool length(std::size_t n){return n && n<=cap && !(n&(n-1));}

struct Map {
    U table[4][256]{};
    explicit Map(const U* cols,unsigned bits){
        for(unsigned block=0;block<4;block++)for(unsigned x=1;x<256;x++){
            unsigned bit=__builtin_ctz(x);unsigned index=8*block+bit;
            table[block][x]=table[block][x^(1u<<bit)]^(index<bits?cols[index]:U(0));
        }
    }
    U at(U x)const{return table[0][x&255]^table[1][(x>>8)&255]^table[2][(x>>16)&255]^table[3][(x>>24)&255];}
};

extern "C" int msc_growth_permute(U* values,std::size_t n,const U* columns,unsigned bits,int inverse){
    if(!length(n)||bits>26||n!=(std::size_t(1)<<bits)||(inverse!=0 && inverse!=1))return -1;
    U pivots[26]{};unsigned rank=0;bool identity=true;
    for(unsigned i=0;i<bits;i++){
        U v=columns[i];if(v>=n)return -2;identity&=v==(U(1)<<i);
        for(int k=int(bits)-1;k>=0;k--)if(v&(U(1)<<k)){
            if(pivots[k])v^=pivots[k];else{pivots[k]=v;rank++;break;}
        }
    }
    if(rank!=bits)return -3;
    if(identity)return 0;
    try{
        Map map(columns,bits);std::vector<unsigned char> seen((n+7)/8,0);
        for(std::size_t start=0;start<n;start++){
            if(seen[start>>3]&(1u<<(start&7)))continue;
            U index=start,saved=values[start];
            do{
                seen[index>>3]|=1u<<(index&7);U next=map.at(index);
                if(inverse){std::swap(saved,values[next]);index=next;}
                else{values[index]=next==start?saved:values[next];index=next;}
            }while(index!=start);
        }
        return 0;
    }catch(...){return -4;}
}

extern "C" int msc_growth_walsh(U* values,std::size_t n,unsigned bits,U p,int inverse){
    if(!domain(p)||!length(n)||bits>26||(std::size_t(1)<<bits)>n||(inverse!=0 && inverse!=1))return -1;
    Field f(p);
    for(unsigned bit=0;bit<bits;bit++){
        std::size_t width=std::size_t(1)<<bit;
        for(std::size_t base=0;base<n;base+=2*width)for(std::size_t j=0;j<width;j++){
            U a=values[base+j],b=values[base+width+j];
            U s=f.add(a,b),d=f.sub(a,b);
            values[base+j]=inverse?f.half(s):s;
            values[base+width+j]=inverse?f.half(d):d;
        }
    }
    return 0;
}

extern "C" int msc_growth_products(const U* a,std::size_t na,const U* b,std::size_t nb,
                                      U* out,std::size_t nc,const U* ac,const U* bc,
                                      const U* cc,const U* pc,unsigned g,unsigned q,U p){
    if(!domain(p)||!length(na)||!length(nb)||!length(nc)||g>26||q<g||q>32)return -1;
    U block=U(1)<<g,mask=block-1;
    if(block>std::min({na,nb,nc}))return -2;
    for(unsigned j=0;j<q;j++){
        if(ac[j]>=na||bc[j]>=nb||cc[j]>=nc||pc[j]>mask)return -3;
        if(j<g){if(ac[j]!=(U(1)<<j)||bc[j]!=(U(1)<<j)||cc[j]!=(U(1)<<j)||pc[j])return -4;}
        else if((ac[j]&mask)||(bc[j]&mask)||(cc[j]&mask))return -5;
    }
    Field f(p);U ca[32]{},cb[32]{},ccarry[32]{},cp[32]{};
    for(unsigned j=g;j<q;j++){
        unsigned k=j-g;ca[k]=ac[j]^(k?ca[k-1]:0);cb[k]=bc[j]^(k?cb[k-1]:0);
        ccarry[k]=cc[j]^(k?ccarry[k-1]:0);cp[k]=pc[j]^(k?cp[k-1]:0);
    }
    std::fill(out,out+nc,U(0));U ai=0,bi=0,ci=0,sign=0;
    for(U residual=0;residual<(U(1)<<(q-g));residual++){
        if(residual){unsigned k=__builtin_ctzll(residual);ai^=ca[k];bi^=cb[k];ci^=ccarry[k];sign^=cp[k];}
        for(U mode=0;mode<block;mode++){
            U left=a[ai+mode],right=b[bi+mode];if(!left||!right)continue;
            U term=f.mul(left,right);
            if(__builtin_parityll(sign&mode) && term)term=p-term;
            out[ci+mode]=f.add(out[ci+mode],term);
        }
    }
    return 0;
}
