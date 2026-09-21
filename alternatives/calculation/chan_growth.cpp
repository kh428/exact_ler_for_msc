// Exact contracted growth coefficients using Newton identities in the XOR
// group algebra. All integers are checked against a signed 128-bit bound.
// Endpoint h1 has zero incoming first-stage syndrome; h2 is unrestricted.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>
using U=unsigned __int128;using I=__int128;
struct Hash {size_t operator()(U x)const{return uint64_t(x)^(uint64_t(x>>64)*0x9e3779b97f4a7c15ULL);}};
struct S {U label;uint64_t w[5];};
struct P {U label;uint64_t aa,ab,bb,ac,bc,ad;};
struct G {U key;uint32_t begin,end;};
struct C {U label;uint64_t w;};
std::string decimal(I x){if(!x)return "0";bool neg=x<0;U y=neg?-x:x;std::string z;while(y){z+=char('0'+y%10);y/=10;}if(neg)z+='-';std::reverse(z.begin(),z.end());return z;}
struct Lookup {
 std::vector<uint64_t> h1;std::unordered_map<uint32_t,uint64_t> h2;
 std::vector<std::pair<uint32_t,uint64_t>> nonzero;uint64_t maxh=0;
 Lookup(const char*a,const char*b):h1(1<<20){
  for(int which=0;which<2;which++){
   std::ifstream in(which?b:a);size_t n;in>>n;
   for(size_t i=0;i<n;i++){uint32_t key;uint64_t w;in>>key>>w;if(key>=(1u<<26))throw std::runtime_error("endpoint label");maxh=std::max(maxh,w);
    if(which)h2[key]=w;else{if(key&63)throw std::runtime_error("h1 initial syndrome");h1[key>>6]=w;nonzero.push_back({key,w});}}
   if(!in)throw std::runtime_error("endpoint input");
  }
 }
 uint64_t one(uint32_t w)const{return w&63?0:h1[w>>6];}
 uint64_t two(uint32_t w)const{auto it=h2.find(w);return it==h2.end()?0:it->second;}
};
std::vector<G> groups(const std::vector<P>&p,std::vector<uint32_t>&idx,U mask){
 idx.resize(p.size());std::iota(idx.begin(),idx.end(),0);
 std::sort(idx.begin(),idx.end(),[&](uint32_t a,uint32_t b){U x=p[a].label&mask,y=p[b].label&mask;return x!=y?x<y:p[a].label<p[b].label;});
 std::vector<G> g;g.reserve(p.size());
 for(uint32_t i=0;i<idx.size();){uint32_t j=i+1;U key=p[idx[i]].label&mask;while(j<idx.size()&&(p[idx[j]].label&mask)==key)j++;g.push_back({key,i,j});i=j;}
 return g;
}
int main(int argc,char**argv){try{
 if(argc!=9)throw std::runtime_error("powers h1 h2 bits threads mode limit output");
 auto began=std::chrono::steady_clock::now();
 int bits=std::stoi(argv[4]),threads=std::stoi(argv[5]),mode=std::stoi(argv[6]);size_t limit=std::stoull(argv[7]);
 if(bits<1||bits>90||threads<1||threads>14)throw std::runtime_error("arguments");
 U mask=(U(1)<<bits)-1;Lookup h(argv[2],argv[3]);
 std::ifstream in(argv[1]);size_t n;in>>n;std::vector<S>s(n);uint64_t total1=0,total5=0;
 for(auto &r:s){uint64_t lo,hi;in>>lo>>hi;for(auto &v:r.w)in>>v;r.label=U(lo)|(U(hi)<<64);total1+=r.w[0];total5+=r.w[4];}
 if(!in||n>5000)throw std::runtime_error("power input/size");
 // Sum absolute Newton terms is bounded by 120 * max(S1,Sj)^degree * max h.
 long double guard=120.0L*h.maxh;for(int j=0;j<5;j++)guard*=std::max(uint64_t(1),total1);
 if(guard>1e36L)throw std::runtime_error("128-bit height guard");
 std::vector<P> p;p.reserve(n*(n+1)/2);
 for(size_t i=0;i<n;i++)for(size_t j=i;j<n;j++){
  auto&a=s[i];auto&b=s[j];uint64_t m=i==j?1:2;
  auto mixed=[&](int u,int v){return a.w[u]*b.w[v]+(i==j?0:a.w[v]*b.w[u]);};
  p.push_back({a.label^b.label,m*a.w[0]*b.w[0],mixed(0,1),m*a.w[1]*b.w[1],mixed(0,2),mixed(1,2),mixed(0,3)});
 }
 std::sort(p.begin(),p.end(),[](auto a,auto b){return a.label<b.label;});
 size_t z=0;for(size_t i=0;i<p.size();i++){if(z&&p[z-1].label==p[i].label){auto&a=p[z-1];auto&b=p[i];a.aa+=b.aa;a.ab+=b.ab;a.bb+=b.bb;a.ac+=b.ac;a.bc+=b.bc;a.ad+=b.ad;}else p[z++]=p[i];}p.resize(z);
 std::vector<uint32_t> index;auto g=groups(p,index,mask);
 // Fourth powers and the A*B fifth-order correction share this pair-pair join.
 std::vector<I> aa1(threads),aa2(threads),ab1(threads);std::atomic<size_t> job{0};
 std::vector<std::thread> pool;
 for(int t=0;t<threads;t++)pool.emplace_back([&,t]{
  for(;;){size_t start=job.fetch_add(128);if(start>=g.size())break;
   for(size_t gi=start;gi<std::min(start+128,g.size());gi++)for(uint32_t i=g[gi].begin;i<g[gi].end;i++)for(uint32_t j=i;j<g[gi].end;j++){
    auto&a=p[index[i]];auto&b=p[index[j]];uint32_t w=(a.label^b.label)>>bits;
    uint64_t v1=h.one(w),v2=h.two(w);if(!v1&&!v2)continue;
    I value=I(a.aa)*b.aa*(i==j?1:2);aa1[t]+=value*v1;aa2[t]+=value*v2;
    ab1[t]+=(I(a.aa)*b.ab+(i==j?0:I(a.ab)*b.aa))*v1;
   }
  }
 });for(auto&t:pool)t.join();pool.clear();
 I v4[2]={},aab=std::accumulate(ab1.begin(),ab1.end(),I(0));
 v4[0]=std::accumulate(aa1.begin(),aa1.end(),I(0));v4[1]=std::accumulate(aa2.begin(),aa2.end(),I(0));
 I v5corr=-10*aab;
 for(auto &r:s){
  auto found=std::lower_bound(g.begin(),g.end(),r.label&mask,[](G a,U b){return a.key<b;});
  if(found!=g.end()&&found->key==(r.label&mask))for(uint32_t i=found->begin;i<found->end;i++){
   auto&a=p[index[i]];uint32_t w=(a.label^r.label)>>bits;uint64_t v1=h.one(w),v2=h.two(w);
   v4[0]-=6*I(a.aa)*r.w[1]*v1;v4[1]-=6*I(a.aa)*r.w[1]*v2;
   v5corr+=(15*I(a.bb)*r.w[0]+20*I(a.aa)*r.w[2])*v1;
  }
  if(!(r.label&mask)){uint32_t w=r.label>>bits;v4[0]-=6*I(r.w[3])*h.one(w);v4[1]-=6*I(r.w[3])*h.two(w);v5corr+=24*I(r.w[4])*h.one(w);}
 }
 for(auto &r:p)if(!(r.label&mask)){
  uint32_t w=r.label>>bits;I coeff=3*I(r.bb)+8*I(r.ac);
  v4[0]+=coeff*h.one(w);v4[1]+=coeff*h.two(w);
  v5corr-= (20*I(r.bc)+30*I(r.ad))*h.one(w);
 }
 for(int j=0;j<2;j++){if(v4[j]<0||v4[j]%24)throw std::runtime_error("G4 Newton positivity/integrality");v4[j]/=24;}
 double fourth_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
 std::cerr<<"G4 contractions complete in "<<fourth_seconds<<" s\n";
 I fifth=0;size_t used=0,all=0;uint64_t matched=0;
 if(mode>=5){
  // Absorb the sparse final first-order response into S1 before doing the
  // fifth-order join: C=S1*h1. This avoids a separate loop for each single.
  U cmask=(U(1)<<(bits+6))-1;
  std::vector<C> c;c.reserve(s.size()*h.nonzero.size());
  for(auto &r:s)if(r.w[0])for(auto [label,w]:h.nonzero)c.push_back({r.label^(U(label)<<bits),r.w[0]*w});
  std::sort(c.begin(),c.end(),[](C a,C b){return a.label<b.label;});
  z=0;for(size_t i=0;i<c.size();i++){if(z&&c[z-1].label==c[i].label)c[z-1].w+=c[i].w;else c[z++]=c[i];}c.resize(z);
  std::sort(c.begin(),c.end(),[&](C a,C b){U x=a.label&cmask,y=b.label&cmask;return x!=y?x<y:a.label<b.label;});
  std::vector<G> cg;for(uint32_t i=0;i<c.size();){uint32_t j=i+1;U key=c[i].label&cmask;while(j<c.size()&&(c[j].label&cmask)==key)j++;cg.push_back({key,i,j});i=j;}
  all=cg.size();used=limit?std::min(limit,all):all;
  g=groups(p,index,cmask);
  std::unordered_map<U,uint32_t,Hash> lookup;lookup.reserve(g.size());for(uint32_t i=0;i<g.size();i++)lookup[g[i].key]=i;
  job=0;std::vector<I> totals(threads);std::vector<uint64_t> matches(threads);
  for(int t=0;t<threads;t++)pool.emplace_back([&,t]{
   std::vector<uint64_t> values(1<<20),active((1<<20)/64);
   for(;;){size_t ci=job.fetch_add(1);if(ci>=used)break;
    // Sample evenly for a limited pilot, including the identity group.
    size_t actual=used==all?ci:ci*all/used;auto &target=cg[actual];
    std::fill(active.begin(),active.end(),0);
    for(uint32_t k=target.begin;k<target.end;k++){uint32_t r=c[k].label>>(bits+6);values[r]=c[k].w;active[r>>6]|=uint64_t(1)<<(r&63);}
    for(auto &left:g){U right_key=left.key^target.key;if(right_key<left.key)continue;auto it=lookup.find(right_key);if(it==lookup.end())continue;auto &right=g[it->second];
     for(uint32_t i=left.begin;i<left.end;i++)for(uint32_t j=left.key==right.key?i:right.begin;j<right.end;j++){
      auto&a=p[index[i]];auto&b=p[index[j]];if(!a.aa||!b.aa)continue;matches[t]++;
      uint32_t r=(a.label^b.label)>>(bits+6);if(!(active[r>>6]>>(r&63)&1))continue;
      totals[t]+=I(a.aa)*b.aa*(i==j?1:2)*values[r];
     }
    }
   }
  });for(auto&t:pool)t.join();pool.clear();
  fifth=std::accumulate(totals.begin(),totals.end(),I(0))+v5corr;
  matched=std::accumulate(matches.begin(),matches.end(),uint64_t(0));
  if(used==all){if(fifth<0||fifth%120)throw std::runtime_error("G5 Newton positivity/integrality");fifth/=120;}
 }
 std::ofstream out(argv[8]);
 out<<"{\"complete\":"<<(mode<5||used==all?"true":"false")<<",\"G4_h1\":\""<<decimal(v4[0])<<"\",\"G4_h2\":\""<<decimal(v4[1])<<"\",\"G5_h1\":\""<<decimal(fifth)<<"\",\"growth_five_computed\":"<<(mode>=5&&used==all?"true":"false")<<",\"pairs\":"<<p.size()<<",\"C_groups_used\":"<<used<<",\"C_groups_all\":"<<all<<",\"AA_C_matches\":"<<matched<<",\"fourth_seconds\":"<<fourth_seconds<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()<<",\"threads\":"<<threads<<"}\n";
 if(!out)throw std::runtime_error("output");
}catch(std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
