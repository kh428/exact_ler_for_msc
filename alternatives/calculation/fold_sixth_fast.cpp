// Versioned experiments. The archived exact coefficient engine is unchanged.
#include "phase_sixth_direct.cpp"

// Retain only actual zero signatures, before allocating/sorting large Nodes.
extern "C" int fold_fast_zero(void* ptr, uint64_t limit, uint64_t* out) {
  try {
    auto& c = *static_cast<Sixth*>(ptr);
    if (!c.zero_triples.empty() || !c.zero_pairs.empty()) return 20;
    for (int k=0; k<c.n; ++k) {
      if (c.stop) throw 9;
      const auto part=c.project[k];
      for (uint64_t at=c.offsets[part]; at<c.offsets[part+1]; ++at) {
        const uint32_t ij=c.pairs[at];
        const int i=ij>>16, j=ij&65535;
        if (j>=k || !(sx(sx(c.sig[i],c.sig[j]),c.sig[k])==Sig{})) continue;
        c.zero_triples.push_back((uint64_t(i)<<32)|(uint64_t(j)<<16)|uint64_t(k));
      }
      if (c.zero_triples.size()>limit) throw 10;
    }
    std::sort(c.zero_triples.begin(),c.zero_triples.end());
    for (uint64_t at=c.offsets[0]; at<c.offsets[1]; ++at) {
      const auto ij=c.pairs[at];
      if (c.sig[ij>>16]==c.sig[ij&65535]) c.zero_pairs.push_back(ij);
    }
    std::sort(c.zero_pairs.begin(),c.zero_pairs.end());
    out[0]=c.zero_triples.size(); out[1]=c.zero_pairs.size();
    return 0;
  } catch(int e) {return e;} catch(...) {return 99;}
}

struct LabelTriple {
  std::array<uint64_t,5> label{};
  uint16_t first=0,last=0;
};

// Measure exact merging, preserving the endpoint needed to exclude repeated
// classes across the canonical left/right split. No coefficient is computed.
extern "C" int fold_merge_census(void* ptr, int kind, uint32_t slice,
                                  uint64_t limit, uint64_t* out) {
  try {
    const auto& c=*static_cast<Sixth*>(ptr);
    if(c.s->words>5) return 21;
    std::vector<uint64_t> ids;
    if(kind==1) ids=c.zero_triples;
    else {
      auto source=triples(c,slice,limit);
      ids.reserve(source.size());
      for(const auto& row:source) if(!(row.sig==Sig{})) ids.push_back(row.ids);
    }
    std::vector<LabelTriple> rows;rows.reserve(ids.size());
    for(const auto id:ids) {
      LabelTriple v;v.first=id>>32;v.last=id&65535;
      const int middle=(id>>16)&65535;
      for(int w=0;w<c.s->words;++w)
        v.label[w]=c.s->labels[v.first*c.s->words+w]
                  ^c.s->labels[middle*c.s->words+w]
                  ^c.s->labels[v.last*c.s->words+w];
      rows.push_back(v);
    }
    std::sort(rows.begin(),rows.end(),[](const LabelTriple&a,const LabelTriple&b){
      if(a.label!=b.label)return a.label<b.label;
      if(a.first!=b.first)return a.first<b.first;
      return a.last<b.last;
    });
    std::vector<uint64_t> seen(c.n,0);uint64_t groups=0,by_first=0,by_last=0;
    int first_previous=-1;std::array<uint64_t,5> previous{};
    for(const auto& row:rows) {
      if(!groups || row.label!=previous) {
        ++groups;previous=row.label;first_previous=-1;
      }
      if(first_previous!=row.first) {++by_first;first_previous=row.first;}
      if(seen[row.last]!=groups) {++by_last;seen[row.last]=groups;}
    }
    out[0]=rows.size();out[1]=groups;out[2]=by_first;out[3]=by_last;
    return 0;
  } catch(int e) {return e;} catch(...) {return 99;}
}
