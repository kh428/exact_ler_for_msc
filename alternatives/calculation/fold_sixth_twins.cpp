// Exact two-variable zero tests. Learned samples select tests, never outcomes.
#include "fold_sixth_filter.cpp"

struct TwinPlane {
  int i,j;
  std::unordered_map<uint64_t,Bits> matches;
};

static uint64_t twin_row(const Prepared&v,int i,int j) {
  const uint64_t mask=(uint64_t(1)<<i)|(uint64_t(1)<<j);
  return v.adj[i]^v.adj[j]^(v.lo&mask);
}

// For q(z)=sum_i lin_i z_i+2 sum_{i<j} adj_ij z_i z_j (mod 4),
// M=adj+diag(lin mod 2). Equal rows i,j give M(e_i+e_j)=0.
// The sum is zero when hi_i != hi_j, by translation by e_i+e_j.
static void learn_twins(const Plan&p,const Prepared&l,const Prepared&r,
                       const Prepared&base,std::array<uint32_t,4096>&scores) {
  const uint64_t lo=l.lo^r.lo^base.lo;
  uint64_t hi=l.hi^r.hi^base.hi^(l.lo&r.lo)^(l.lo&base.lo)^(r.lo&base.lo);
  std::array<uint64_t,64> rows{};
  for(int i=0;i<p.n;++i) {
    uint64_t cross=l.odd[i]&r.odd[i];
    for(int e=0;e<p.extra_words[i];++e) {
      const int k=p.extra_offset[i]+e;cross^=l.extra[k]&r.extra[k];
    }
    if(__builtin_popcountll(cross)&1)hi^=uint64_t(1)<<i;
    rows[i]=l.adj[i]^r.adj[i]^base.adj[i]^(lo&(uint64_t(1)<<i));
  }
  for(int i=0;i<p.n;++i)if(!rows[i] && (hi>>i&1))return;
  for(int i=0;i<p.n;++i)for(int j=i+1;j<p.n;++j)
    if(rows[i]==rows[j] && ((hi>>i)^(hi>>j))&1)++scores[64*i+j];
}

static std::vector<TwinPlane> twin_planes(const Plan&p,const std::vector<Message>&right,
                                        const std::array<uint32_t,4096>&scores,int count) {
  std::vector<int> choices;
  for(int i=0;i<p.n;++i)for(int j=i+1;j<p.n;++j)if(scores[64*i+j])choices.push_back(64*i+j);
  std::stable_sort(choices.begin(),choices.end(),[&](int a,int b){return scores[a]>scores[b];});
  if(choices.size()>size_t(count))choices.resize(count);
  std::vector<TwinPlane> out;
  const size_t nw=(right.size()+63)/64;
  for(int key:choices) {
    TwinPlane plane{key/64,key%64,{}};
    for(size_t k=0;k<right.size();++k) {
      const uint64_t row=twin_row(right[k].p,plane.i,plane.j);
      auto it=plane.matches.try_emplace(row,nw,0);
      it.first->second[k/64]|=uint64_t(1)<<(k%64);
    }
    out.push_back(std::move(plane));
  }
  return out;
}

static void high_bits(const Plan&p,const Planes&z,const Prepared&l,const Prepared&base,
                      int v,Bits&bad) {
  const bool lo=l.lo>>v&1,bl=base.lo>>v&1;
  const bool hc=((l.hi^base.hi)>>v&1)^(lo&&bl);
  for(size_t w=0;w<bad.size();++w)bad[w]=z.hi[w]^(hc?~uint64_t(0):0)^((lo^bl)?z.lo[w]:0);
  if(!z.odd.empty()) {
    auto apply=[&](uint64_t val,int offset){while(val){
      const int k=__builtin_ctzll(val);val&=val-1;
      for(size_t w=0;w<bad.size();++w)bad[w]^=z.odd[offset+k][w];
    }};
    apply(l.odd[v],0);
    for(int e=0;e<p.extra_words[v];++e)apply(l.extra[p.extra_offset[v]+e],64*(e+1));
  }
}

static void twin_join(const Sixth&c,const uint64_t*ll,size_t nl,const uint64_t*rr,
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
    std::array<uint32_t,4096> scores{};int learned=0;
    std::vector<TwinPlane> twins;
    for(size_t i=0;i<nl;++i) {
      if(first(ll[i],3)>=max_min)break;
      if(last(ll[i])>=max_min)continue;
      const Message left=message(c,ll[i],3);++stats[10];const auto&l=left.p;
      const size_t start=std::upper_bound(right.begin(),right.end(),left.max,
           [](int value,const Message&m){return value<m.min;})-right.begin();
      stats[3]+=right.size()-start;
      auto evaluate_one=[&](size_t j){
        if(filtered && max_tests && learned<128) {
          learn_twins(p,l,right[j].p,c.base,scores);++learned;
          if(learned==128){twins=twin_planes(p,right,scores,max_tests);stats[14]+=twins.size();}
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
      for(const auto&t:twins) {
        auto it=t.matches.find(twin_row(l,t.i,t.j)^twin_row(c.base,t.i,t.j));
        if(it==t.matches.end())continue;
        high_bits(p,planes[t.i],l,c.base,t.i,bad);
        high_bits(p,planes[t.j],l,c.base,t.j,second);
        uint64_t remains=0;
        for(size_t w=0;w<nw;++w){
          keep[w]&=~((bad[w]^second[w])&it->second[w]);remains|=keep[w];
        }
        if(!remains){any=false;break;}
      }
      if(!any)continue;
      for(size_t w=0;w<nw;++w){uint64_t bits=keep[w];while(bits){
        const int b=__builtin_ctzll(bits);bits&=bits-1;evaluate_one(64*w+b);
      }}
    }
  }
}

extern "C" int fold_twin_six(void*ptr,int kind,uint32_t slice,uint64_t begin,
                              uint64_t end,int max_tests,uint64_t*limbs,uint64_t*stats) {
  if(max_tests<0||max_tests>64)return 23;
  try {
    const auto&c=*static_cast<Sixth*>(ptr);Hist h{};std::fill(stats,stats+15,0);
    if(kind==1) {
      end=std::min(end,uint64_t(c.zero_triples.size()));if(begin>end)return 15;
      twin_join(c,c.zero_triples.data(),c.zero_triples.size(),
                c.zero_triples.data()+begin,end-begin,max_tests,h,stats);
    }else if(kind==0) {
      auto triples=nonzero_triples(c,slice,25000000);stats[8]=triples.size();size_t at=0;
      while(at<triples.size()) {
        const auto sig=triples[at].sig;std::vector<uint64_t> ids;
        while(at<triples.size()&&triples[at].sig==sig)ids.push_back(triples[at++].ids);
        twin_join(c,ids.data(),ids.size(),ids.data(),ids.size(),max_tests,h,stats);
      }
    }else return 16;
    for(size_t i=0;i<4096;++i){const auto v=static_cast<unsigned __int128>(h[3][i]);
      limbs[2*i]=uint64_t(v);limbs[2*i+1]=uint64_t(v>>64);}
    return 0;
  }catch(int e){return e;}catch(...){return 99;}
}
