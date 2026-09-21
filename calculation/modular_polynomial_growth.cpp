// Coefficient-streamed growth uses the unchanged exact scalar maps/Walsh.
#include "modular_growth.cpp"

extern "C" int msc_growth_accumulate_products(const U* a,std::size_t na,
        const U* b,std::size_t nb,U* out,std::size_t nc,const U* ac,
        const U* bc,const U* cc,const U* pc,unsigned g,unsigned q,U p,int reset) {
    if(!domain(p)||!length(na)||!length(nb)||!length(nc)||g>26||q<g||q>32||
       (reset!=0 && reset!=1))return -1;
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
    if(reset)std::fill(out,out+nc,U(0));
    U ai=0,bi=0,ci=0,sign=0;
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

extern "C" int msc_growth_project_polynomial(const U* values,std::size_t n,
        unsigned width,const std::int32_t* dest,const std::uint16_t* tags,
        U* out,std::size_t m,unsigned character,U p){
    if(!domain(p)||!length(n)||!length(m)||n>(1u<<20)||m>(1u<<20)||
       !width||width>11||character>=512)return -1;
    Field f(p);std::fill(out,out+m*width,U(0));
    for(std::size_t i=0;i<n;i++){
        if(dest[i]<0)continue;
        if(std::size_t(dest[i])>=m || tags[i]>=512)return -2;
        const bool sign=__builtin_parity(tags[i]&character);
        for(unsigned k=0;k<width;k++){
            U v=values[i*width+k];if(v>=p)return -3;
            if(sign && v)v=p-v;
            U &r=out[std::size_t(dest[i])*width+k];r=f.add(r,v);
        }
    }
    return 0;
}
