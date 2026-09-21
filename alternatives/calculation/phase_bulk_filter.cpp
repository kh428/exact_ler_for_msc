// Exact batched elimination of isolated quantum-phase variables.
// Benchmark sums over supplied labels, not a complete physical coefficient.
#include "phase_gaussian_kernel.cpp"
#include <chrono>
#include <unordered_map>

using Bits = std::vector<uint64_t>;
struct Planes {
  Bits used, lo, hi;
  std::vector<Bits> odd;
  std::unordered_map<uint64_t,Bits> adjacency;
};

struct Tagged {
  Prepared value;
  int constant=0;
  std::array<uint64_t,4> odd_constants{};
};
static Plan without_constants(const Plan &p) {
  Plan r=p;
  for(auto &g:r.groups)g.terms.erase(std::remove_if(g.terms.begin(),g.terms.end(),
      [](const Term&t){return !t.quantum;}),g.terms.end());
  r.fast=true;
  return r;
}
static Tagged tagged(const Plan &p,const Plan &reduced,const uint64_t *label) {
  Tagged t;t.value=prepare(reduced,label);int index=0;
  for(const auto &g:p.groups) {
    bool zero=true;for(auto w:g.error)zero &= w==0;
    const bool active=zero || parity(label,g.error);
    for(const auto &term:g.terms)if(!term.quantum) {
      if(active)t.constant=(t.constant+term.coefficient)&7;
      if(!zero && term.coefficient%4==2) {
        if(index>=256)throw 1;
        if(active)t.odd_constants[index/64]|=uint64_t(1)<<(index%64);
        index++;
      } else if(!zero && term.coefficient%2)throw 2;
    }
  }
  return t;
}
static void tagged_pair(const Plan &p,const Tagged &l,const Tagged&r,const Tagged&base,int16_t*out) {
  prepared_pair(p,l.value,r.value,base.value,out);
  if(out[1]<0)return;
  uint64_t cross=0;for(int w=0;w<4;w++)cross^=l.odd_constants[w]&r.odd_constants[w];
  const int c=l.constant+r.constant-base.constant-4*(__builtin_popcountll(cross)&1);
  out[1]=(out[1]+c+32)&7;
}

extern "C" int phase_bulk_sum(void *pa,void *px,void *py,void *pl,void *pr,
                              int mode,uint64_t *limbs,uint64_t *stats) {
  const auto &a=*static_cast<Plan*>(pa),&x=*static_cast<Plan*>(px),&y=*static_cast<Plan*>(py);
  const auto &left=*static_cast<Source*>(pl),&right=*static_cast<Source*>(pr);
  if(!a.fast || a.words>16 || right.values.size()>8192 || left.values.size()>8192 || mode<0 || mode>7)return 1;
  const size_t nr=right.values.size(), nw=(nr+63)/64;
  if(!nr || left.values.empty())return 2;
  uint64_t zero[16]{},label[16];const Prepared base=prepare(a,zero);
  std::vector<Planes> planes(a.n);
  const bool direct=mode>=5;
  const int filter=mode==7?3:(mode==6?2:(mode==5?0:(mode==3?2:(mode==4?0:mode))));
  auto start=std::chrono::steady_clock::now();
  if(filter)for(int i=0;i<a.n;i++) {
    auto &z=planes[i];z.used.assign(nw,0);z.lo.assign(nw,0);z.hi.assign(nw,0);
    if(filter>=2 && (a.odd_active>>i&1))z.odd.assign(64*(1+a.extra_words[i]),Bits(nw,0));
    for(size_t j=0;j<nr;j++) {
      const auto &r=right.values[j];const uint64_t b=uint64_t(1)<<(j%64);const size_t w=j/64;
      if(r.used>>i&1)z.used[w]|=b;
      if(r.lo>>i&1)z.lo[w]|=b;
      if(r.hi>>i&1)z.hi[w]|=b;
      if(filter==3) {
        auto found=z.adjacency.try_emplace(r.adj[i],nw,0);
        found.first->second[w]|=b;
      }
      if(!z.odd.empty()) {
        uint64_t v=r.odd[i];while(v){int k=__builtin_ctzll(v);v&=v-1;z.odd[k][w]|=b;}
        for(int ew=0;ew<a.extra_words[i];ew++) {
          v=r.extra[a.extra_offset[i]+ew];
          while(v){int k=__builtin_ctzll(v);v&=v-1;z.odd[64*(ew+1)+k][w]|=b;}
        }
      }
    }
  }
  const Plan reduced_x=without_constants(x),reduced_y=without_constants(y);
  std::vector<Tagged> lx,rx,ly,ry;Tagged bx,by;
  if(mode==3 || mode==4) {
    try {
      bx=tagged(x,reduced_x,zero);by=tagged(y,reduced_y,zero);
      for(size_t i=0;i<left.values.size();i++) {
        lx.push_back(tagged(x,reduced_x,&left.labels[i*a.words]));
        ly.push_back(tagged(y,reduced_y,&left.labels[i*a.words]));
      }
      for(size_t j=0;j<right.values.size();j++) {
        rx.push_back(tagged(x,reduced_x,&right.labels[j*a.words]));
        ry.push_back(tagged(y,reduced_y,&right.labels[j*a.words]));
      }
    }catch(...){return 6;}
  }
  stats[4]=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
  std::array<__int128,4096> hist{};stats[0]=left.values.size()*nr;stats[1]=stats[2]=stats[3]=0;
  auto evaluate_pair=[&](size_t i,size_t j)->int {
    stats[1]++;int16_t av[2],xv[2],yv[2];prepared_pair(a,left.values[i],right.values[j],base,av);
    if(av[1]==-1)return 0;if(av[1]!=0)return 3;stats[2]++;
    if(direct) {
      if(av[0] < -256 || av[0]>=256)return 5;
      stats[3]++;hist[(av[0]+256)*8+av[1]]++;return 0;
    }
    if(mode>=3){tagged_pair(reduced_x,lx[i],rx[j],bx,xv);tagged_pair(reduced_y,ly[i],ry[j],by,yv);}
    else {
      for(int w=0;w<a.words;w++)label[w]=left.labels[i*a.words+w]^right.labels[j*a.words+w];
      evaluate(x,label,xv);evaluate(y,label,yv);
    }
    if(xv[1]<-1 || yv[1]<-1)return 4;
    if(xv[1]==0 && yv[1]==0 && xv[0]+1==av[0] && yv[0]+1==av[0])return 0;
    stats[3]++;
    auto add=[&](int h,int j){if(h < -256 || h>=256)return false;hist[(h+256)*8+(j&7)]++;return true;};
    if(!add(av[0]-2,av[1]))return 5;
    if(xv[1]!=-1 && !add(xv[0]-3,xv[1]+4))return 5;
    if(yv[1]!=-1 && !add(yv[0]-3,yv[1]+4))return 5;
    return 0;
  };
  Bits keep(nw), bad(nw);
  for(size_t i=0;i<left.values.size();i++) {
    if(!filter){for(size_t j=0;j<nr;j++){int e=evaluate_pair(i,j);if(e)return e;}continue;}
    std::fill(keep.begin(),keep.end(),~uint64_t(0));
    if(nr%64)keep.back()=(uint64_t(1)<<(nr%64))-1;
    const auto &l=left.values[i];
    uint64_t possible=(a.n==64?~uint64_t(0):((uint64_t(1)<<a.n)-1));
    if(filter!=3)possible&=~(l.used|base.used);
    if(filter==1)possible&=~a.odd_active;
    while(possible) {
      const int v=__builtin_ctzll(possible);possible&=possible-1;const auto &z=planes[v];
      const Bits *matches=nullptr;
      if(filter==3) {
        auto found=z.adjacency.find(l.adj[v]^base.adj[v]);
        if(found==z.adjacency.end())continue;
        matches=&found->second;
      }
      const bool ll=l.lo>>v&1, bl=base.lo>>v&1;
      const bool hc=((l.hi^base.hi)>>v&1)^(ll&&bl);
      for(size_t w=0;w<nw;w++)bad[w]=z.hi[w]^(hc?~uint64_t(0):0)^((ll^bl)?z.lo[w]:0);
      if(filter>=2 && !z.odd.empty()) {
        uint64_t b=l.odd[v];
        while(b){int k=__builtin_ctzll(b);b&=b-1;for(size_t w=0;w<nw;w++)bad[w]^=z.odd[k][w];}
        for(int ew=0;ew<a.extra_words[v];ew++) {
          b=l.extra[a.extra_offset[v]+ew];
          while(b){int k=__builtin_ctzll(b);b&=b-1;for(size_t w=0;w<nw;w++)bad[w]^=z.odd[64*(ew+1)+k][w];}
        }
      }
      uint64_t any=0;
      for(size_t w=0;w<nw;w++) {
        const uint64_t even=(ll^bl)?z.lo[w]:~z.lo[w];
        keep[w]&=~(bad[w]&even&(matches?(*matches)[w]:~z.used[w]));any|=keep[w];
      }
      if(!any)break;
    }
    for(size_t w=0;w<nw;w++) {
      uint64_t bits=keep[w];while(bits){int b=__builtin_ctzll(bits);bits&=bits-1;int e=evaluate_pair(i,64*w+b);if(e)return e;}
    }
  }
  for(size_t i=0;i<hist.size();i++){auto v=static_cast<unsigned __int128>(hist[i]);limbs[2*i]=uint64_t(v);limbs[2*i+1]=uint64_t(v>>64);}
  return 0;
}
