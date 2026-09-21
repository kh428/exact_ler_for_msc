// Exact sixth-class join experiments, with adaptive ordering of zero tests.
// The tests are unchanged identities; sampling chooses only their order.
#include "fold_sixth_fast.cpp"
#include <numeric>

static bool rejection_bits(const Plan&p,const Planes&z,const Prepared&l,
                           const Prepared&base,int v,Bits&bad) {
  auto it=z.adjacency.find(l.adj[v]^base.adj[v]);
  if(it==z.adjacency.end())return false;
  const bool lo=l.lo>>v&1,bl=base.lo>>v&1;
  const bool hc=((l.hi^base.hi)>>v&1)^(lo&&bl);
  for(size_t w=0;w<bad.size();++w)
    bad[w]=z.hi[w]^(hc?~uint64_t(0):0)^((lo^bl)?z.lo[w]:0);
  if(!z.odd.empty()) {
    auto apply=[&](uint64_t val,int offset){while(val){
      const int k=__builtin_ctzll(val);val&=val-1;
      for(size_t w=0;w<bad.size();++w)bad[w]^=z.odd[offset+k][w];
    }};
    apply(l.odd[v],0);
    for(int e=0;e<p.extra_words[v];++e)apply(l.extra[p.extra_offset[v]+e],64*(e+1));
  }
  for(size_t w=0;w<bad.size();++w) {
    const uint64_t even=(lo^bl)?z.lo[w]:~z.lo[w];
    bad[w]&=even&it->second[w];
  }
  return true;
}

static void filter_join(const Sixth&c,const uint64_t*ll,size_t nl,
                        const uint64_t*rr,size_t nr,int mode,
                        Hist&hist,uint64_t*stats) {
  if(!nl||!nr)return;
  const auto&p=*c.p;
  constexpr size_t block=2048;
  for(size_t rb=0;rb<nr;rb+=block) {
    if(c.stop)throw 9;
    const size_t re=std::min(nr,rb+block);
    std::vector<Message> right;right.reserve(re-rb);
    for(size_t j=rb;j<re;++j)right.push_back(message(c,rr[j],3));
    const int max_min=right.back().min;
    const bool filtered=right.size()>=((mode&2)?4:32);
    auto planes=filtered?make_planes(p,right):std::vector<Planes>();
    const size_t nw=(right.size()+63)/64;
    Bits keep(nw),bad(nw);
    std::vector<int> order(p.n);std::iota(order.begin(),order.end(),0);
    if(filtered && (mode&1)) {
      std::vector<uint64_t> score(p.n,0);
      const size_t available=std::lower_bound(ll,ll+nl,max_min,
          [](uint64_t id,int value){return first(id,3)<value;})-ll;
      for(int sample=0;sample<12 && available;++sample) {
        const size_t i=sample*(available-1)/11;
        if(last(ll[i])>=max_min)continue;
        const auto m=message(c,ll[i],3);
        for(int v=0;v<p.n;++v)if(rejection_bits(p,planes[v],m.p,c.base,v,bad))
          for(uint64_t bits:bad)score[v]+=__builtin_popcountll(bits);
      }
      std::stable_sort(order.begin(),order.end(),[&](int a,int b){return score[a]>score[b];});
    }
    for(size_t i=0;i<nl;++i) {
      if(first(ll[i],3)>=max_min)break;
      if(last(ll[i])>=max_min)continue;
      const Message left=message(c,ll[i],3);++stats[10];
      const Prepared&l=left.p;
      const size_t start=std::upper_bound(right.begin(),right.end(),left.max,
          [](int value,const Message&m){return value<m.min;})-right.begin();
      stats[3]+=right.size()-start;
      auto evaluate_one=[&](size_t j){
        ++stats[11];int16_t value[2];prepared_pair(p,l,right[j].p,c.base,value);
        if(value[1]==-1)return;
        ++stats[7];add_index(hist,3,value,20*left.w[0]*right[j].w[0]);
      };
      if(!filtered){for(size_t j=start;j<right.size();++j)evaluate_one(j);continue;}
      std::fill(keep.begin(),keep.end(),~uint64_t(0));
      if(right.size()%64)keep.back()=(uint64_t(1)<<(right.size()%64))-1;
      for(size_t w=0;w<start/64;++w)keep[w]=0;
      if(start%64)keep[start/64]&=~((uint64_t(1)<<(start%64))-1);
      for(int v:order) {
        ++stats[12];
        if(!rejection_bits(p,planes[v],l,c.base,v,bad))continue;
        uint64_t any=0;
        for(size_t w=0;w<nw;++w){keep[w]&=~bad[w];any|=keep[w];}
        stats[13]+=nw;
        if(!any)break;
      }
      for(size_t w=0;w<nw;++w){
        uint64_t bits=keep[w];while(bits){
          const int b=__builtin_ctzll(bits);bits&=bits-1;evaluate_one(64*w+b);
        }
      }
    }
  }
}

static std::vector<Node> nonzero_triples(const Sixth&c,uint32_t slice,uint64_t limit) {
  std::vector<Node> out;out.reserve(400000);
  for(int k=0;k<c.n;++k) {
    if(c.stop)throw 9;
    const auto part=slice^c.project[k];
    for(uint64_t at=c.offsets[part];at<c.offsets[part+1];++at) {
      const auto ij=c.pairs[at];const int i=ij>>16,j=ij&65535;
      if(j>=k)continue;
      const auto sig=sx(sx(c.sig[i],c.sig[j]),c.sig[k]);
      if(sig==Sig{})continue;
      out.push_back({sig,(uint64_t(i)<<32)|(uint64_t(j)<<16)|uint64_t(k)});
    }
    if(out.size()>limit)throw 10;
  }
  std::sort(out.begin(),out.end(),node_less);return out;
}

extern "C" int fold_filter_six(void*ptr,int kind,uint32_t slice,uint64_t begin,
                                uint64_t end,int mode,uint64_t*limbs,uint64_t*stats) {
  if(mode<0||mode>3)return 22;
  try {
    const auto&c=*static_cast<Sixth*>(ptr);Hist h{};std::fill(stats,stats+14,0);
    if(kind==1) {
      end=std::min(end,uint64_t(c.zero_triples.size()));if(begin>end)return 15;
      filter_join(c,c.zero_triples.data(),c.zero_triples.size(),
                  c.zero_triples.data()+begin,end-begin,mode,h,stats);
    }else if(kind==0) {
      auto triples=nonzero_triples(c,slice,25000000);stats[8]=triples.size();
      size_t at=0;
      while(at<triples.size()) {
        const auto sig=triples[at].sig;std::vector<uint64_t> ids;
        while(at<triples.size()&&triples[at].sig==sig)ids.push_back(triples[at++].ids);
        filter_join(c,ids.data(),ids.size(),ids.data(),ids.size(),mode,h,stats);
      }
    }else return 16;
    for(size_t i=0;i<4096;++i) {
      const auto v=static_cast<unsigned __int128>(h[3][i]);
      limbs[2*i]=uint64_t(v);limbs[2*i+1]=uint64_t(v>>64);
    }
    return 0;
  } catch(int e) {return e;} catch(...) {return 99;}
}
