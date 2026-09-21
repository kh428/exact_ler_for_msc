// Exact sixth-order coefficient of the distinct-class tanh product.
// Physical-location exclusion is restored by the formal log/cosh identity.
#include "phase_bulk_filter.cpp"
#include <atomic>
#include <stdexcept>

using I128=__int128;
struct Sig {
  uint64_t a=0,b=0;
  bool operator==(const Sig&o)const{return a==o.a&&b==o.b;}
  bool operator<(const Sig&o)const{return b!=o.b?b<o.b:a<o.a;}
};
static Sig sx(Sig a,Sig b){return {a.a^b.a,a.b^b.b};}
struct Node {Sig sig;uint64_t ids;};
static bool node_less(const Node&a,const Node&b){return a.sig==b.sig?a.ids<b.ids:a.sig<b.sig;}
struct Sixth {
  Plan *p;Source *s;Prepared base;int bits,n;
  std::vector<Sig> sig;
  std::vector<uint32_t> project,pairs;
  std::vector<uint64_t> offsets,zero_triples,zero_pairs;
  std::vector<std::array<uint64_t,4>> powers;
  std::atomic<bool> stop{false};
};
static int first(uint64_t ids,int count){return count==3?int(ids>>32):int(ids>>16);}
static int last(uint64_t ids){return int(ids&65535);}
static std::vector<Node> triples(const Sixth&c,uint32_t slice,uint64_t limit){
  std::vector<Node> out;out.reserve(400000);
  for(int k=0;k<c.n;k++){
    if(c.stop)throw 9;
    const uint32_t part=slice^c.project[k];
    for(uint64_t at=c.offsets[part];at<c.offsets[part+1];at++){
      uint32_t ij=c.pairs[at];int i=ij>>16,j=ij&65535;if(j>=k)continue;
      out.push_back({sx(sx(c.sig[i],c.sig[j]),c.sig[k]),(uint64_t(i)<<32)|(uint64_t(j)<<16)|uint64_t(k)});
    }
    if(out.size()>limit)throw 10;
  }
  std::sort(out.begin(),out.end(),node_less);return out;
}
static std::vector<Node> pairs(const Sixth&c,uint32_t slice){
  std::vector<Node> out;out.reserve(c.offsets[slice+1]-c.offsets[slice]);
  for(uint64_t at=c.offsets[slice];at<c.offsets[slice+1];at++){
    uint32_t ij=c.pairs[at];out.push_back({sx(c.sig[ij>>16],c.sig[ij&65535]),ij});
  }
  std::sort(out.begin(),out.end(),node_less);return out;
}
struct Message {Prepared p;I128 w[3]{};int min,max;};
static Message message(const Sixth&c,uint64_t ids,int count){
  int i=count==3?int(ids>>32):int(ids>>16),j=count==3?int((ids>>16)&65535):int(ids&65535);
  Message r;r.min=i;r.max=last(ids);
  r.p=shifted(*c.p,c.s->values[i],c.s->values[j],c.base);
  const auto &a=c.powers[i],&b=c.powers[j];
  if(count==2){
    r.w[0]=2*I128(a[0])*b[0];
    r.w[1]=-3*(I128(a[0])*b[1]+I128(a[1])*b[0]);
    r.w[2]=8*((I128(a[2])-I128(a[0])*a[0]*a[0])*b[0]+
              (I128(b[2])-I128(b[0])*b[0]*b[0])*a[0])+6*I128(a[1])*b[1];
  }else{
    int k=last(ids);const auto &z=c.powers[k];
    r.p=shifted(*c.p,r.p,c.s->values[k],c.base);
    r.w[0]=6*I128(a[0])*b[0]*z[0];
    r.w[1]=-12*(I128(a[1])*b[0]*z[0]+I128(a[0])*b[1]*z[0]+I128(a[0])*b[0]*z[1]);
  }
  return r;
}
static I128 triple_degree6(const Sixth&c,uint64_t ids){
  std::array<int,3> ii{int(ids>>32),int((ids>>16)&65535),int(ids&65535)};
  I128 sum=0;
  for(int a=0;a<3;a++){
    const auto &x=c.powers[ii[a]],&y=c.powers[ii[(a+1)%3]],&z=c.powers[ii[(a+2)%3]];
    sum+=(-180*I128(x[3])+360*I128(x[0])*x[0]*x[1])*y[0]*z[0];
    sum-=120*(I128(x[2])-I128(x[0])*x[0]*x[0])*(I128(y[1])*z[0]+I128(y[0])*z[1]);
  }
  return sum-90*I128(c.powers[ii[0]][1])*c.powers[ii[1]][1]*c.powers[ii[2]][1];
}
// First four histograms: degree-six terms with 3,4,5,6 distinct classes.
// Last three: complete EGF coefficients U3,U4,U5, for a lower-order regression.
using Hist=std::array<std::array<I128,4096>,7>;
static void add_index(Hist&h,int index,const int16_t *v,I128 weight){
  if(v[1]==-1||!weight)return;
  if(v[1]!=0||v[0]<-256||v[0]>=256)throw 11;
  auto &entry=h[index][(v[0]+256)*8];
  I128 next;if(__builtin_add_overflow(entry,weight,&next))throw 12;entry=next;
}
static std::vector<Planes> make_planes(const Plan&p,const std::vector<Message>&right){
  const size_t nw=(right.size()+63)/64;std::vector<Planes> z(p.n);
  for(int v=0;v<p.n;v++){
    auto &plane=z[v];plane.lo.assign(nw,0);plane.hi.assign(nw,0);
    if(p.odd_active>>v&1)plane.odd.assign(64*(1+p.extra_words[v]),Bits(nw,0));
    for(size_t j=0;j<right.size();j++){
      const auto&r=right[j].p;size_t w=j/64;uint64_t bit=uint64_t(1)<<(j%64);
      if(r.lo>>v&1)plane.lo[w]|=bit;if(r.hi>>v&1)plane.hi[w]|=bit;
      auto it=plane.adjacency.try_emplace(r.adj[v],nw,0);it.first->second[w]|=bit;
      if(!plane.odd.empty()){
        auto put=[&](uint64_t val,int offset){while(val){int k=__builtin_ctzll(val);val&=val-1;plane.odd[offset+k][w]|=bit;}};
        put(r.odd[v],0);for(int e=0;e<p.extra_words[v];e++)put(r.extra[p.extra_offset[v]+e],64*(e+1));
      }
    }
  }
  return z;
}
// Both lists are sorted by their first class. Canonical separation makes each
// increasing k-class tuple appear once; repeated classes are never admitted.
static void join(const Sixth&c,const uint64_t*ll,size_t nl,int lc,
                 const uint64_t*rr,size_t nr,int rc,Hist&hist,uint64_t*stats){
  if(!nl||!nr)return;const auto&p=*c.p;const int degree=lc+rc;
  constexpr size_t block=2048;
  for(size_t rb=0;rb<nr;rb+=block){
    if(c.stop)throw 9;
    size_t re=std::min(nr,rb+block);std::vector<Message> right;right.reserve(re-rb);
    for(size_t j=rb;j<re;j++)right.push_back(message(c,rr[j],rc));
    const int max_min=right.back().min;bool filtered=right.size()>=32;
    auto planes=filtered?make_planes(p,right):std::vector<Planes>();
    const size_t nw=(right.size()+63)/64;Bits keep(nw),bad(nw);
    for(size_t i=0;i<nl;i++){
      if(first(ll[i],lc)>=max_min)break;
      if(last(ll[i])>=max_min)continue;
      const Message left=message(c,ll[i],lc);const Prepared &l=left.p;
      size_t start=std::upper_bound(right.begin(),right.end(),left.max,
          [](int v,const Message&m){return v<m.min;})-right.begin();
      stats[degree-3]+=right.size()-start;
      auto evaluate_one=[&](size_t j){
        int16_t v[2];prepared_pair(p,l,right[j].p,c.base,v);if(v[1]==-1)return;
        stats[degree+1]++;
        const auto &r=right[j];I128 weight;
        if(degree==6)weight=20*left.w[0]*r.w[0];
        else if(degree==5)weight=15*left.w[0]*r.w[1]+20*left.w[1]*r.w[0];
        else weight=15*left.w[0]*r.w[2]+20*left.w[1]*r.w[1]+15*left.w[2]*r.w[0];
        add_index(hist,degree-3,v,weight);
        if(degree==4){
          add_index(hist,5,v,6*left.w[0]*r.w[0]);
          add_index(hist,6,v,10*(left.w[0]*r.w[1]+left.w[1]*r.w[0]));
        }else if(degree==5)add_index(hist,6,v,10*left.w[0]*r.w[0]);
      };
      if(!filtered){for(size_t j=start;j<right.size();j++)evaluate_one(j);continue;}
      std::fill(keep.begin(),keep.end(),~uint64_t(0));
      if(right.size()%64)keep.back()=(uint64_t(1)<<(right.size()%64))-1;
      for(size_t w=0;w<start/64;w++)keep[w]=0;
      if(start%64)keep[start/64]&=~((uint64_t(1)<<(start%64))-1);
      for(int v=0;v<p.n;v++){
        const auto&z=planes[v];auto it=z.adjacency.find(l.adj[v]^c.base.adj[v]);
        if(it==z.adjacency.end())continue;
        bool lo=l.lo>>v&1,bl=c.base.lo>>v&1,hc=((l.hi^c.base.hi)>>v&1)^(lo&&bl);
        for(size_t w=0;w<nw;w++)bad[w]=z.hi[w]^(hc?~uint64_t(0):0)^((lo^bl)?z.lo[w]:0);
        if(!z.odd.empty()){
          auto apply=[&](uint64_t val,int offset){while(val){int k=__builtin_ctzll(val);val&=val-1;
            for(size_t w=0;w<nw;w++)bad[w]^=z.odd[offset+k][w];}};
          apply(l.odd[v],0);for(int e=0;e<p.extra_words[v];e++)apply(l.extra[p.extra_offset[v]+e],64*(e+1));
        }
        uint64_t any=0;for(size_t w=0;w<nw;w++){
          uint64_t even=(lo^bl)?z.lo[w]:~z.lo[w];keep[w]&=~(bad[w]&even&it->second[w]);any|=keep[w];
        }
        if(!any)break;
      }
      for(size_t w=0;w<nw;w++){uint64_t bits=keep[w];while(bits){int b=__builtin_ctzll(bits);bits&=bits-1;evaluate_one(64*w+b);}}
    }
  }
}
static void triples_only(const Sixth&c,const std::vector<uint64_t>&t,Hist&h,uint64_t*stats){
  for(auto ids:t){auto m=message(c,ids,3);int16_t v[2];prepared_pair(*c.p,m.p,c.base,c.base,v);
    stats[0]++;if(v[1]==-1)continue;stats[4]++;
    add_index(h,0,v,triple_degree6(c,ids));
    add_index(h,4,v,m.w[0]);add_index(h,5,v,m.w[1]);
    std::array<int,3> ii{int(ids>>32),int((ids>>16)&65535),int(ids&65535)};
    I128 w5=0;
    for(int a=0;a<3;a++){
      const auto &x=c.powers[ii[a]],&y=c.powers[ii[(a+1)%3]],&z=c.powers[ii[(a+2)%3]];
      w5+=40*(I128(x[2])-I128(x[0])*x[0]*x[0])*y[0]*z[0];
      w5+=30*I128(x[0])*y[1]*z[1];
    }
    add_index(h,6,v,w5);
  }
}
extern "C" {
void* sixth_make(void*plan,void*source,const uint64_t*signatures,const uint32_t*projection,
                 const uint64_t*powers,int n,int bits){
  if(n<3||n>=65536||bits<1||bits>18)return nullptr;
  auto*c=new Sixth;c->p=static_cast<Plan*>(plan);c->s=static_cast<Source*>(source);c->n=n;c->bits=bits;
  if(!c->p->fast||c->s->values.size()!=size_t(n)){delete c;return nullptr;}
  uint64_t zero[16]{};c->base=prepare(*c->p,zero);c->sig.resize(n);c->powers.resize(n);
  c->project.assign(projection,projection+n);c->offsets.assign((1<<bits)+1,0);
  for(int i=0;i<n;i++){c->sig[i]={signatures[2*i],signatures[2*i+1]};for(int k=0;k<4;k++)c->powers[i][k]=powers[4*i+k];}
  for(int i=0;i<n;i++)for(int j=i+1;j<n;j++)c->offsets[(projection[i]^projection[j])+1]++;
  for(size_t i=1;i<c->offsets.size();i++)c->offsets[i]+=c->offsets[i-1];
  c->pairs.resize(c->offsets.back());auto cursor=c->offsets;
  for(int i=0;i<n;i++)for(int j=i+1;j<n;j++)c->pairs[cursor[projection[i]^projection[j]]++]=(uint32_t(i)<<16)|j;
  return c;
}
int sixth_zero(void*ptr,uint64_t limit,uint64_t*out){
  try{auto&c=*static_cast<Sixth*>(ptr);auto t=triples(c,0,limit);auto p=pairs(c,0);
    for(const auto&r:t)if(r.sig==Sig{})c.zero_triples.push_back(r.ids);
    for(const auto&r:p)if(r.sig==Sig{})c.zero_pairs.push_back(r.ids);
    out[0]=c.zero_triples.size();out[1]=c.zero_pairs.size();return 0;
  }catch(int e){return e;}catch(...){return 99;}
}
void sixth_stop(void*ptr){static_cast<Sixth*>(ptr)->stop=true;}
void sixth_delete(void*ptr){delete static_cast<Sixth*>(ptr);}
int sixth_run(void*ptr,int kind,uint32_t slice,uint64_t begin,uint64_t end,uint64_t limit,
              uint64_t*limbs,uint64_t*stats){
  try{
    const auto&c=*static_cast<Sixth*>(ptr);Hist h{};std::fill(stats,stats+10,0);
    if(kind==1){
      end=std::min(end,uint64_t(c.zero_triples.size()));if(begin>end)return 15;
      const auto*r=c.zero_triples.data()+begin;size_t nr=end-begin;
      join(c,c.zero_pairs.data(),c.zero_pairs.size(),2,r,nr,3,h,stats);
      join(c,c.zero_triples.data(),c.zero_triples.size(),3,r,nr,3,h,stats);
    }else if(kind==2){
      triples_only(c,c.zero_triples,h,stats);
      join(c,c.zero_pairs.data(),c.zero_pairs.size(),2,c.zero_pairs.data(),c.zero_pairs.size(),2,h,stats);
    }else if(kind==0){
      auto tt=triples(c,slice,limit);auto pp=pairs(c,slice);stats[8]=tt.size();stats[9]=pp.size();
      size_t it=0,ip=0;
      while(it<tt.size()||ip<pp.size()){
        Sig s=ip==pp.size()?tt[it].sig:(it==tt.size()?pp[ip].sig:(tt[it].sig<pp[ip].sig?tt[it].sig:pp[ip].sig));
        std::vector<uint64_t> t,p;
        while(it<tt.size()&&tt[it].sig==s)t.push_back(tt[it++].ids);
        while(ip<pp.size()&&pp[ip].sig==s)p.push_back(pp[ip++].ids);
        if(s==Sig{})continue;
        join(c,p.data(),p.size(),2,p.data(),p.size(),2,h,stats);
        join(c,p.data(),p.size(),2,t.data(),t.size(),3,h,stats);
        join(c,t.data(),t.size(),3,t.data(),t.size(),3,h,stats);
      }
    }else return 16;
    for(size_t k=0;k<7;k++)for(size_t i=0;i<4096;i++){
      auto v=static_cast<unsigned __int128>(h[k][i]);size_t o=k*8192+2*i;limbs[o]=uint64_t(v);limbs[o+1]=uint64_t(v>>64);
    }
    return 0;
  }catch(int e){return e;}catch(...){return 99;}
}
}
