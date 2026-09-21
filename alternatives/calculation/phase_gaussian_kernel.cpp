// Exact quadratic phase sums. No floating point or modular reconstruction.
// The returned pair (h,j) represents 2^(h/2) exp(i*pi*j/4).
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

struct Term { uint64_t quantum; int coefficient; int odd_index=-1; };
struct Group { std::vector<uint64_t> error; std::vector<Term> terms; };
struct Constraint { std::vector<uint64_t> error; int offset; };
struct Plan {
  int n, words, norm, phase;
  std::vector<Group> groups;
  std::vector<Constraint> constraints;
  bool fast=true;
  std::array<int,64> extra_offset{},extra_words{};
  uint64_t odd_active=0;
};
struct Prepared {
  std::array<uint64_t,64> adj{},odd{};
  std::array<uint64_t,16> extra{};
  uint64_t lo=0,hi=0,used=0,syn0=0,syn1=0;
};
struct Source {int words;std::vector<uint64_t> labels;std::vector<Prepared> values;};
static int parity(const uint64_t *label,const std::vector<uint64_t>&mask) {
  uint64_t x=0;for(size_t j=0;j<mask.size();j++)x^=label[j]&mask[j];
  return __builtin_popcountll(x)&1;
}
static void finish(const Plan&p,std::array<int,64> lin,std::array<uint64_t,64> adj,int constant,int16_t*out);
static void evaluate(const Plan &p,const uint64_t *label,int16_t *out) {
  out[0]=0;out[1]=-1; // -1 denotes the zero sum.
  for(const auto &c:p.constraints)if(parity(label,c.error)!=c.offset)return;
  std::array<int,64> lin{};
  std::array<uint64_t,64> adj{};
  int constant=0;
  for(const auto &g:p.groups) {
    bool zero=true;for(uint64_t w:g.error)zero &= w==0;
    if(!zero && !parity(label,g.error))continue;
    for(const auto &t:g.terms) {
      if(!t.quantum) {constant+=t.coefficient;continue;}
      int i=__builtin_ctzll(t.quantum);
      uint64_t other=t.quantum & (t.quantum-1);
      if(!other)lin[i]=(lin[i]+t.coefficient/2)&3;
      else {int j=__builtin_ctzll(other);adj[i]^=uint64_t(1)<<j;adj[j]^=uint64_t(1)<<i;}
    }
  }
  finish(p,lin,adj,constant,out);
}
static void finish(const Plan&p,std::array<int,64> lin,std::array<uint64_t,64> adj,int constant,int16_t*out) {
  uint64_t active=0;
  for(int i=0;i<p.n;i++){active|=adj[i];if(lin[i])active|=uint64_t(1)<<i;}
  int h=2*(p.n-__builtin_popcountll(active)),phase=constant+p.phase;
  while(active) {
    int v=-1,best=1000;uint64_t todo=active;
    while(todo) {
      int i=__builtin_ctzll(todo);todo&=todo-1;
      int score=((lin[i]&1)?0:100)+__builtin_popcountll(adj[i]&active);
      if(score<best){best=score;v=i;}
    }
    uint64_t bit=uint64_t(1)<<v,near=adj[v]&active;
    int a=lin[v];active^=bit;
    todo=near;while(todo){int u=__builtin_ctzll(todo);todo&=todo-1;adj[u]&=~bit;}
    if(a&1) {
      h++;phase+=(a==1?1:-1);
      todo=near;
      while(todo) {
        int u=__builtin_ctzll(todo);todo&=todo-1;
        lin[u]=(lin[u]-a)&3;
        adj[u]^=near^(uint64_t(1)<<u);
      }
    } else if(!near) {
      if(a==2)return;
      h+=2;
    } else {
      int w=-1,degree=100;todo=near;
      while(todo) {
        int u=__builtin_ctzll(todo);todo&=todo-1;
        int d=__builtin_popcountll(adj[u]&active);
        if(d<degree){degree=d;w=u;}
      }
      uint64_t wb=uint64_t(1)<<w,rest=near^wb,wn=adj[w]&active;
      int offset=a/2,lw=lin[w];active^=wb;h+=2;
      // No odd linear pivot remained, so lw is even.
      if(lw&1){out[1]=-2;return;}
      phase+=2*lw*offset;
      todo=rest;
      while(todo){int u=__builtin_ctzll(todo);todo&=todo-1;lin[u]=(lin[u]+(offset?-lw:lw))&3;}
      todo=wn;
      while(todo) {
        int u=__builtin_ctzll(todo);todo&=todo-1;
        adj[u]&=~wb;
        if(offset)lin[u]^=2;
        if(rest&(uint64_t(1)<<u))lin[u]^=2;
        uint64_t others=rest&~(uint64_t(1)<<u);
        adj[u]^=others;
        while(others){int r=__builtin_ctzll(others);others&=others-1;adj[r]^=uint64_t(1)<<u;}
      }
    }
  }
  out[0]=h-2*p.norm;out[1]=(phase%8+8)%8;
}

static Prepared prepare(const Plan&p,const uint64_t*label) {
  Prepared v;std::array<int,64> lin{};
  for(size_t i=0;i<p.constraints.size();i++)if(parity(label,p.constraints[i].error)) {
    if(i<64)v.syn0|=uint64_t(1)<<i;else v.syn1|=uint64_t(1)<<(i-64);
  }
  for(const auto&g:p.groups) {
    bool zero=true;for(auto w:g.error)zero&=w==0;
    if(!zero&&!parity(label,g.error))continue;
    for(const auto&t:g.terms) {
      int i=__builtin_ctzll(t.quantum);uint64_t other=t.quantum&(t.quantum-1);
      if(!other) {
        lin[i]=(lin[i]+t.coefficient/2)&3;
        if(t.odd_index>=0) {
          if(t.odd_index<64)v.odd[i]|=uint64_t(1)<<t.odd_index;
          else v.extra[p.extra_offset[i]+(t.odd_index-64)/64]|=uint64_t(1)<<((t.odd_index-64)%64);
        }
      } else {int j=__builtin_ctzll(other);v.adj[i]^=uint64_t(1)<<j;v.adj[j]^=uint64_t(1)<<i;}
    }
  }
  for(int i=0;i<p.n;i++) {
    if(lin[i]&1)v.lo|=uint64_t(1)<<i;
    if(lin[i]&2)v.hi|=uint64_t(1)<<i;
    v.used|=v.adj[i];
  }
  return v;
}
static void prepared_pair(const Plan&p,const Prepared&a,const Prepared&b,const Prepared&base,int16_t*out) {
  out[0]=0;out[1]=-1;
  if(a.syn0!=b.syn0 || a.syn1!=b.syn1)return;
  uint64_t lo=a.lo^b.lo^base.lo;
  uint64_t hi=a.hi^b.hi^base.hi^(a.lo&b.lo)^(a.lo&base.lo)^(b.lo&base.lo);
  uint64_t possible=a.used|b.used|base.used;
  uint64_t isolated=(p.n==64?~uint64_t(0):((uint64_t(1)<<p.n)-1))&~possible&~lo;
  uint64_t done=isolated&~p.odd_active;
  if(hi&done)return;
  isolated&=p.odd_active;
  auto cross=[&](int i){uint64_t v=a.odd[i]&b.odd[i];
    for(int j=0;j<p.extra_words[i];j++){int k=p.extra_offset[i]+j;v^=a.extra[k]&b.extra[k];}
    return __builtin_popcountll(v)&1;};
  while(isolated) {
    int i=__builtin_ctzll(isolated);uint64_t bit=uint64_t(1)<<i;isolated&=isolated-1;done|=bit;
    if(cross(i))hi^=bit;
    if(hi&bit)return; // An isolated -1 phase sums to zero.
  }
  std::array<int,64> lin{};std::array<uint64_t,64> adj{};
  for(int i=0;i<p.n;i++) {
    uint64_t bit=uint64_t(1)<<i;
    if(!(done&bit) && cross(i))hi^=bit;
    lin[i]=((lo>>i)&1)+2*((hi>>i)&1);adj[i]=a.adj[i]^b.adj[i]^base.adj[i];
  }
  finish(p,lin,adj,0,out);
}
static Prepared shifted(const Plan&p,const Prepared&a,const Prepared&b,const Prepared&base) {
  Prepared v;v.lo=a.lo^b.lo^base.lo;
  v.hi=a.hi^b.hi^base.hi^(a.lo&b.lo)^(a.lo&base.lo)^(b.lo&base.lo);
  v.syn0=a.syn0^b.syn0;v.syn1=a.syn1^b.syn1;
  for(int i=0;i<p.n;i++) {
    uint64_t cross=a.odd[i]&b.odd[i];
    for(int j=0;j<p.extra_words[i];j++){int k=p.extra_offset[i]+j;cross^=a.extra[k]&b.extra[k];}
    if(__builtin_popcountll(cross)&1)v.hi^=uint64_t(1)<<i;
    v.adj[i]=a.adj[i]^b.adj[i]^base.adj[i];v.used|=v.adj[i];
    v.odd[i]=a.odd[i]^b.odd[i]^base.odd[i];
  }
  for(int i=0;i<16;i++)v.extra[i]=a.extra[i]^b.extra[i]^base.extra[i];
  return v;
}

extern "C" {
void *phase_make(int n,int words,int norm,int phase,int nt,
  const uint64_t *quantum,const uint64_t *errors,const int *coeff,
  int nc,const uint64_t *constraints,const int *offsets) {
  if(n<0 || n>64 || words<1)return nullptr;
  auto *p=new Plan{n,words,norm,phase,{},{}};
  for(int i=0;i<nt;i++) {
    int d=__builtin_popcountll(quantum[i]);
    if(d>2 || (d==1 && (coeff[i]&1)) || (d==2 && coeff[i]!=4)){delete p;return nullptr;}
    std::vector<uint64_t> mask(errors+i*words,errors+(i+1)*words);
    auto it=std::find_if(p->groups.begin(),p->groups.end(),[&](const Group&g){return g.error==mask;});
    if(it==p->groups.end()){p->groups.push_back(Group{mask,{}});it=p->groups.end()-1;}
    it->terms.push_back(Term{quantum[i],coeff[i]});
  }
  for(int i=0;i<nc;i++)p->constraints.push_back(Constraint{
    std::vector<uint64_t>(constraints+i*words,constraints+(i+1)*words),offsets[i]});
  std::array<int,64> odds{};
  for(auto&g:p->groups)for(auto&t:g.terms) {
    if(!t.quantum){p->fast=false;continue;}
    if(__builtin_popcountll(t.quantum)==1 && ((t.coefficient/2)&1)) {
      int i=__builtin_ctzll(t.quantum);t.odd_index=odds[i]++;
    }
  }
  int extras=0;
  for(int i=0;i<n;i++) {
    if(odds[i])p->odd_active|=uint64_t(1)<<i;
    p->extra_offset[i]=extras;p->extra_words[i]=std::max(0,(odds[i]+63)/64-1);extras+=p->extra_words[i];
  }
  if(extras>16)p->fast=false;
  if(nc>128)p->fast=false;
  for(int i=0;i<nc;i++)if(offsets[i])p->fast=false;
  return p;
}
void phase_delete(void *p){delete static_cast<Plan*>(p);}
void *phase_source_make(void *pa,const uint64_t*labels,uint64_t count) {
  const auto &p=*static_cast<Plan*>(pa);if(!p.fast)return nullptr;
  auto *s=new Source;s->words=p.words;s->labels.assign(labels,labels+count*p.words);s->values.reserve(count);
  for(uint64_t i=0;i<count;i++)s->values.push_back(prepare(p,labels+i*p.words));return s;
}
void phase_source_delete(void *s){delete static_cast<Source*>(s);}
void phase_evaluate(void *ptr,const uint64_t *labels,uint64_t count,int16_t *out) {
  const auto &p=*static_cast<Plan*>(ptr);
  for(uint64_t i=0;i<count;i++)evaluate(p,labels+i*p.words,out+2*i);
}
// Histogram of exact B responses for a weighted XOR product of two identical
// lists. Each histogram entry is a signed 128-bit integer, returned as two
// uint64 limbs. h in [-256,255], j in [0,7]. Zero acceptance is a safe shortcut
// only because these input labels are physical Pauli patterns.
int phase_pair_sum(void *pa,void *px,void *py,const uint64_t *labels,
  const uint64_t *weights,uint64_t count,uint64_t begin,uint64_t end,uint64_t *limbs,
  uint64_t *stats) {
  const auto &a=*static_cast<Plan*>(pa),&x=*static_cast<Plan*>(px),&y=*static_cast<Plan*>(py);
  if(a.words!=x.words || a.words!=y.words || a.words>16 || end>count)return 1;
  std::array<__int128,4096> histogram{};uint64_t label[16];
  std::vector<Prepared> prepared;Prepared base;
  if(a.fast) {
    uint64_t zero[16]{};base=prepare(a,zero);prepared.reserve(count);
    for(uint64_t i=0;i<count;i++)prepared.push_back(prepare(a,labels+i*a.words));
  }
  stats[0]=stats[1]=stats[2]=0;stats[3]=stats[4]=~uint64_t(0);
  // The omitted diagonal has label zero and B=0 for the verified ideal target.
  for(uint64_t i=begin;i<end;i++)for(uint64_t j=i+1;j<count;j++) {
    stats[0]++;
    int16_t av[2],xv[2],yv[2];
    if(a.fast)prepared_pair(a,prepared[i],prepared[j],base,av);
    else {for(int w=0;w<a.words;w++)label[w]=labels[i*a.words+w]^labels[j*a.words+w];evaluate(a,label,av);}
    if(av[1]==-1)continue;
    if(av[1]!=0)return 2; // Physical acceptance is positive real.
    for(int w=0;w<a.words;w++)label[w]=labels[i*a.words+w]^labels[j*a.words+w];
    stats[1]++;evaluate(x,label,xv);evaluate(y,label,yv);
    if(xv[1]<-1 || yv[1]<-1)return 3;
    if(xv[1]==0 && yv[1]==0 && xv[0]+1==av[0] && yv[0]+1==av[0])continue;
    stats[2]++;
    if(stats[3]==~uint64_t(0)){stats[3]=i;stats[4]=j;}
    __int128 weight=static_cast<__int128>(weights[i])*weights[j]*(i==j?1:2);
    auto insert=[&](int h,int phase,__int128 value) {
      if(h < -256 || h >= 256)return false;
      histogram[(h+256)*8+(phase&7)]+=value;return true;
    };
    if(!insert(av[0]-2,av[1],weight))return 4;
    if(xv[1]!=-1 && !insert(xv[0]-3,xv[1]+4,weight))return 4;
    if(yv[1]!=-1 && !insert(yv[0]-3,yv[1]+4,weight))return 4;
  }
  for(size_t i=0;i<histogram.size();i++) {
    unsigned __int128 v=static_cast<unsigned __int128>(histogram[i]);
    limbs[2*i]=static_cast<uint64_t>(v);limbs[2*i+1]=static_cast<uint64_t>(v>>64);
  }
  return 0;
}
// Modes: 0 = different signature buckets, full rectangle with factor two;
// 1 = same bucket, upper triangle with factor two; 2 = mixed S1^2/S1*S2
// upper triangle with the two ordered weights added. Mode 2 uses shift zero.
int phase_shifted_cross_sum(void *pa,void *px,void *py,void *sl,void *sr,
  const uint64_t*lw,const uint64_t*rw,const uint64_t*shifts,const uint64_t*sw,
  uint64_t nt,uint64_t begin,uint64_t end,int mode,uint64_t *limbs,uint64_t *stats) {
  const auto &a=*static_cast<Plan*>(pa),&x=*static_cast<Plan*>(px),&y=*static_cast<Plan*>(py);
  const auto &left=*static_cast<Source*>(sl),&right=*static_cast<Source*>(sr);
  if(!a.fast || a.words>16 || end>left.values.size() || mode<0 || mode>2)return 1;
  if(mode && left.values.size()!=right.values.size())return 2;
  uint64_t zero[16]{},label[16];Prepared base=prepare(a,zero);
  std::array<__int128,4096> histogram{};stats[0]=stats[1]=stats[2]=0;
  for(uint64_t t=0;t<nt;t++) {
    Prepared transform=prepare(a,shifts+t*a.words);
    for(uint64_t i=begin;i<end;i++) {
      Prepared v=shifted(a,left.values[i],transform,base);
      for(uint64_t j=mode?i+1:0;j<right.values.size();j++) {
        stats[0]++;int16_t av[2],xv[2],yv[2];prepared_pair(a,v,right.values[j],base,av);
        if(av[1]==-1)continue;
        if(av[1]!=0)return 3;
        stats[1]++;
        for(int w=0;w<a.words;w++)label[w]=left.labels[i*a.words+w]^right.labels[j*a.words+w]^shifts[t*a.words+w];
        evaluate(x,label,xv);evaluate(y,label,yv);
        if(xv[1]<-1 || yv[1]<-1)return 4;
        if(xv[1]==0 && yv[1]==0 && xv[0]+1==av[0] && yv[0]+1==av[0])continue;
        stats[2]++;
        __int128 weight=mode==2?(static_cast<__int128>(lw[i])*rw[j]+static_cast<__int128>(lw[j])*rw[i]):
          2*static_cast<__int128>(lw[i])*rw[j];weight*=sw[t];
        auto insert=[&](int h,int phase,__int128 value){if(h < -256 || h>=256)return false;histogram[(h+256)*8+(phase&7)]+=value;return true;};
        if(!insert(av[0]-2,av[1],weight))return 5;
        if(xv[1]!=-1 && !insert(xv[0]-3,xv[1]+4,weight))return 5;
        if(yv[1]!=-1 && !insert(yv[0]-3,yv[1]+4,weight))return 5;
      }
    }
  }
  for(size_t i=0;i<histogram.size();i++) {
    unsigned __int128 v=static_cast<unsigned __int128>(histogram[i]);
    limbs[2*i]=static_cast<uint64_t>(v);limbs[2*i+1]=static_cast<uint64_t>(v>>64);
  }
  return 0;
}
}
