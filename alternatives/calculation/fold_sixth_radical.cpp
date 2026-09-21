// General exact kernel-vector zero tests, plus bounded-size triple allocation.
// Original benchmark implementations are included unchanged for comparison.
#include "fold_sixth_twins.cpp"

static uint64_t kernel_key(const Prepared& p,uint64_t v) {
  uint64_t result=p.lo&v;
  while(v){int i=__builtin_ctzll(v);v&=v-1;result^=p.adj[i];}
  return result;
}
static unsigned phase_at(const Prepared& p,uint64_t v) {
  unsigned value=__builtin_popcountll(p.lo&v)+2*__builtin_popcountll(p.hi&v);
  while(v){int i=__builtin_ctzll(v);v&=v-1;value+=2*__builtin_popcountll(p.adj[i]&v);}
  return value&3;
}
static Prepared combined_phase(const Plan&p,const Prepared&l,const Prepared&r,
                               const Prepared&base) {
  Prepared result;
  result.lo=l.lo^r.lo^base.lo;
  result.hi=l.hi^r.hi^base.hi^(l.lo&r.lo)^(l.lo&base.lo)^(r.lo&base.lo);
  for(int i=0;i<p.n;++i) {
    result.adj[i]=l.adj[i]^r.adj[i]^base.adj[i];
    uint64_t odd=l.odd[i]&r.odd[i];
    for(int e=0;e<p.extra_words[i];++e) {
      int k=p.extra_offset[i]+e;odd^=l.extra[k]&r.extra[k];
    }
    if(__builtin_popcountll(odd)&1)result.hi^=uint64_t(1)<<i;
  }
  return result;
}
static void learn_kernels(const Plan&p,const Prepared&l,const Prepared&r,
                         const Prepared&base,std::unordered_map<uint64_t,uint32_t>&scores) {
  const auto combined=combined_phase(p,l,r,base);
  std::array<uint64_t,64> basis{},vectors{};
  for(int i=0;i<p.n;++i) {
    uint64_t row=combined.adj[i]^(combined.lo&(uint64_t(1)<<i));
    uint64_t v=uint64_t(1)<<i;
    while(row) {
      int pivot=63-__builtin_clzll(row);
      if(!basis[pivot]){basis[pivot]=row;vectors[pivot]=v;break;}
      row^=basis[pivot];v^=vectors[pivot];
    }
    if(!row && __builtin_popcountll(v)<=8 && phase_at(combined,v)==2) {
      if(kernel_key(combined,v))throw 31;
      ++scores[v];
    }
  }
}
struct KernelPlane {
  uint64_t v;
  Bits lo,hi;
  std::unordered_map<uint64_t,Bits> matches;
};
static std::vector<KernelPlane> kernel_planes(const std::vector<Message>&right,
                  const std::unordered_map<uint64_t,uint32_t>&scores,int count) {
  std::vector<uint64_t> choices;
  for(const auto& kv:scores)if(__builtin_popcountll(kv.first)>1)choices.push_back(kv.first);
  std::sort(choices.begin(),choices.end(),[&](uint64_t a,uint64_t b){
    if(scores.at(a)!=scores.at(b))return scores.at(a)>scores.at(b);
    return a<b;
  });
  if(choices.size()>size_t(count))choices.resize(count);
  const size_t nw=(right.size()+63)/64;
  std::vector<KernelPlane> out;
  for(uint64_t v:choices) {
    KernelPlane plane{v,Bits(nw,0),Bits(nw,0),{}};
    for(size_t j=0;j<right.size();++j) {
      const auto&r=right[j].p;uint64_t bit=uint64_t(1)<<(j%64);
      unsigned q=phase_at(r,v);
      if(q&1)plane.lo[j/64]|=bit;if(q&2)plane.hi[j/64]|=bit;
      auto it=plane.matches.try_emplace(kernel_key(r,v),nw,0);
      it.first->second[j/64]|=bit;
    }
    out.push_back(std::move(plane));
  }
  return out;
}
static bool kernel_rejection(const Plan&p,const std::vector<Planes>&planes,
           const KernelPlane&k,const Prepared&l,const Prepared&base,Bits&bad) {
  auto it=k.matches.find(kernel_key(l,k.v)^kernel_key(base,k.v));
  if(it==k.matches.end())return false;
  unsigned a=phase_at(l,k.v),b=phase_at(base,k.v);
  const bool high=((a+b)&2),low=((a^b)&1);
  for(size_t w=0;w<bad.size();++w)
    bad[w]=k.hi[w]^(high?~uint64_t(0):0)^(low?k.lo[w]:0);
  uint64_t v=k.v&p.odd_active;
  while(v) {
    int i=__builtin_ctzll(v);v&=v-1;const auto&z=planes[i];
    auto apply=[&](uint64_t word,int offset){while(word){
      int bit=__builtin_ctzll(word);word&=word-1;
      for(size_t w=0;w<bad.size();++w)bad[w]^=z.odd[offset+bit][w];
    }};
    apply(l.odd[i],0);
    for(int e=0;e<p.extra_words[i];++e)apply(l.extra[p.extra_offset[i]+e],64*(e+1));
  }
  for(size_t w=0;w<bad.size();++w)bad[w]&=it->second[w];
  return true;
}

static void radical_join(const Sixth&c,const uint64_t*ll,size_t nl,const uint64_t*rr,
                      size_t nr,int max_tests,Hist&hist,uint64_t*stats) {
  if(!nl||!nr)return;
  const auto&p=*c.p;
  for(size_t rb=0;rb<nr;rb+=2048) {
    if(c.stop)throw 9;
    const size_t re=std::min(nr,rb+2048);
    std::vector<Message> right;right.reserve(re-rb);
    for(size_t j=rb;j<re;++j)right.push_back(message(c,rr[j],3));
    const int max_min=right.back().min;const bool filtered=right.size()>=32;
    auto planes=filtered?make_planes(p,right):std::vector<Planes>();
    const size_t nw=(right.size()+63)/64;Bits keep(nw),bad(nw),second(nw);
    std::unordered_map<uint64_t,uint32_t> scores;int learned=0;
    std::vector<KernelPlane> kernels;
    for(size_t i=0;i<nl;++i) {
      if(first(ll[i],3)>=max_min)break;
      if(last(ll[i])>=max_min)continue;
      const Message left=message(c,ll[i],3);++stats[10];const auto&l=left.p;
      const size_t start=std::upper_bound(right.begin(),right.end(),left.max,
           [](int value,const Message&m){return value<m.min;})-right.begin();
      stats[3]+=right.size()-start;
      auto evaluate_one=[&](size_t j){
        if(filtered && max_tests && learned<128) {
          learn_kernels(p,l,right[j].p,c.base,scores);++learned;
          if(learned==128){kernels=kernel_planes(right,scores,max_tests);stats[14]+=kernels.size();}
        }
        ++stats[11];int16_t value[2];prepared_pair(p,l,right[j].p,c.base,value);
        if(value[1]==-1)return;
        ++stats[7];add_index(hist,3,value,20*left.w[0]*right[j].w[0]);
      };
      if(!filtered){for(size_t j=start;j<right.size();++j)evaluate_one(j);continue;}
      std::fill(keep.begin(),keep.end(),~uint64_t(0));
      if(right.size()%64)keep.back()=(uint64_t(1)<<(right.size()%64))-1;
      for(size_t w=0;w<start/64;++w)keep[w]=0;
      if(start%64)keep[start/64]&=~((uint64_t(1)<<(start%64))-1);
      bool any=true;
      for(int v=0;v<p.n;++v) {
        ++stats[12];if(!rejection_bits(p,planes[v],l,c.base,v,bad))continue;
        uint64_t remains=0;for(size_t w=0;w<nw;++w){keep[w]&=~bad[w];remains|=keep[w];}
        stats[13]+=nw;if(!remains){any=false;break;}
      }
      if(!any)continue;
      for(const auto&k:kernels) {
        if(!kernel_rejection(p,planes,k,l,c.base,bad))continue;
        uint64_t remains=0;
        for(size_t w=0;w<nw;++w){keep[w]&=~bad[w];remains|=keep[w];}
        if(!remains){any=false;break;}
      }
      if(!any)continue;
      for(size_t w=0;w<nw;++w){uint64_t bits=keep[w];while(bits){
        const int b=__builtin_ctzll(bits);bits&=bits-1;evaluate_one(64*w+b);
      }}
    }
  }
}

// The prior complete fifth-order traversal supplies the exact array size.
static std::vector<Node> sized_triples(const Sixth&c,uint32_t slice,size_t expected) {
  if(expected>25000000)throw 10;
  std::vector<Node> out;out.reserve(expected);
  for(int k=0;k<c.n;++k) {
    if(c.stop)throw 9;
    const auto part=slice^c.project[k];
    for(uint64_t at=c.offsets[part];at<c.offsets[part+1];++at) {
      const auto ij=c.pairs[at];int i=ij>>16,j=ij&65535;
      if(j>=k)continue;
      auto sig=sx(sx(c.sig[i],c.sig[j]),c.sig[k]);if(sig==Sig{})continue;
      if(out.size()>=expected)throw 32;
      out.push_back({sig,(uint64_t(i)<<32)|(uint64_t(j)<<16)|uint64_t(k)});
    }
  }
  if(out.size()!=expected)throw 33;
  std::sort(out.begin(),out.end(),node_less);return out;
}
extern "C" int fold_accelerated_six(void*ptr,int kind,uint32_t slice,uint64_t begin,
    uint64_t end,uint64_t expected,int mode,int max_tests,uint64_t*limbs,uint64_t*stats) {
  if(mode<0||mode>1||max_tests<0||max_tests>64)return 23;
  try {
    const auto&c=*static_cast<Sixth*>(ptr);Hist h{};std::fill(stats,stats+15,0);
    auto run=[&](const uint64_t*l,size_t nl,const uint64_t*r,size_t nr){
      if(mode)radical_join(c,l,nl,r,nr,max_tests,h,stats);
      else twin_join(c,l,nl,r,nr,max_tests,h,stats);
    };
    if(kind==1) {
      if(begin>end||end>c.zero_triples.size())return 15;
      run(c.zero_triples.data(),c.zero_triples.size(),c.zero_triples.data()+begin,end-begin);
    }else if(kind==0) {
      auto triples=sized_triples(c,slice,expected);stats[8]=triples.size();size_t at=0;
      while(at<triples.size()) {
        auto sig=triples[at].sig;std::vector<uint64_t> ids;
        while(at<triples.size()&&triples[at].sig==sig)ids.push_back(triples[at++].ids);
        run(ids.data(),ids.size(),ids.data(),ids.size());
      }
    }else return 16;
    for(size_t i=0;i<4096;++i){auto v=static_cast<unsigned __int128>(h[3][i]);
      limbs[2*i]=uint64_t(v);limbs[2*i+1]=uint64_t(v>>64);}
    return 0;
  }catch(int e){return e;}catch(...){return 99;}
}
extern "C" int fold_kernel_zero(int n,uint64_t lo,uint64_t hi,const uint64_t*adj,uint64_t v) {
  Prepared p;p.lo=lo;p.hi=hi;for(int i=0;i<n;++i)p.adj[i]=adj[i];
  return !kernel_key(p,v) && phase_at(p,v)==2;
}
