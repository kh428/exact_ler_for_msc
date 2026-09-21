"""Paired A/B traversal of the selected binary contraction tree."""
import time

class TreeTraversal:
    def _measure(self,i):
        node = self.nodes[i]
        if not node["children"]:
            self.has_final[i] = node["factor"] == 568
            size = 8*(1<<node["d"])*(2 if self.has_final[i] else 1)
            return size,size
        a,b = node["children"]
        pa,sa = self._measure(a); pb,sb = self._measure(b)
        self.has_final[i] = self.has_final[a] or self.has_final[b]
        size = 8*(1<<node["d"])*(2 if self.has_final[i] else 1)
        copy = (sb if self.has_final[a] else sa) if self.has_final[i] else 0
        local = sa+sb+size+copy+max(sa,sb,size)//64
        forward = max(pa,sa+pb,local); reverse = max(pb,sb+pa,local)
        self.left_first[i] = forward <= reverse
        return min(forward,reverse),size

    def evaluate(self,character):
        assert 0 <= character < self.characters
        began = time.perf_counter(); cpu = time.process_time()
        seen = set(); joins = products = adds = 0
        def leaf(f):
            destinations,tags,bits = self.maps[f]
            out = []
            for v in self.physical[f]:
                p = self.projector.project(v,destinations,tags,bits,character)[:,0]
                self.field.convert(p,encode=True)
                out.append(p)
            return tuple(out) if f == 568 else out[0]
        def visit(i):
            nonlocal joins,products,adds
            node = self.nodes[i]
            if not node["children"]:
                f = node["factor"]; assert f not in seen; seen.add(f)
                return leaf(f)
            a,b = node["children"]
            if self.left_first[i]: left = visit(a); right = visit(b)
            else: right = visit(b); left = visit(a)
            if self.has_final[i]:
                if self.has_final[a]:
                    one,p1 = self.native.join(left[0],right.copy(),node["shape"])
                    two,p2 = self.native.join(left[1],right,node["shape"])
                else:
                    one,p1 = self.native.join(left.copy(),right[0],node["shape"])
                    two,p2 = self.native.join(left,right[1],node["shape"])
                result = (one,two)
                products += p1["products"]+p2["products"]
                adds += p1["walsh_additions"]+p2["walsh_additions"]
            else:
                result,p1 = self.native.join(left,right,node["shape"])
                products += p1["products"]; adds += p1["walsh_additions"]
            joins += 1
            return result
        result = visit(self.root)
        for v in result: self.field.convert(v,encode=False)
        assert joins == 568 and seen == set(range(569))
        assert products == self.candidate["costs"]["paired_products"]
        assert adds == self.candidate["costs"]["paired_walsh_additions"]
        return dict(character=character,prime=str(self.prime),
                    residues={name:int(result[o][0]) for o,name in enumerate(("A","B"))},
                    seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu,
                    physical_factors=569,complete_joins=568,products=products,walsh_additions=adds)
