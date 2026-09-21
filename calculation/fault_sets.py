"""Complete small-weight selection by necessary XOR constraints.

Each item is one nonidentity outcome at a physical location. An item has
integer fields location and label. Results contain increasing item indices,
distinct locations, and zero XOR label. No zero-label or cancelling-pair
shortcut is used. These conditions do not themselves certify a logical error.
"""

from bisect import bisect_right
from collections import defaultdict


def gf2_rank(labels):
    rows = {}
    for label in labels:
        while label:
            pivot = label.bit_length()-1
            if pivot in rows:
                label ^= rows[pivot]
            else:
                rows[pivot] = label
                break
    return len(rows)


def candidates(items, weight):
    if weight not in (0, 1, 2, 3):
        raise ValueError("Only complete weights zero through three are implemented")
    if weight == 0:
        yield ()
        return
    by_label = defaultdict(list)
    for i, item in enumerate(items):
        by_label[item["label"]].append(i)
    if weight == 1:
        for i in by_label[0]:
            yield (i,)
        return
    for i, a in enumerate(items):
        if weight == 2:
            matches = by_label.get(a["label"], ())
            for j in matches[bisect_right(matches, i):]:
                if a["location"] != items[j]["location"]:
                    yield (i, j)
        else:
            for j in range(i+1, len(items)):
                b = items[j]
                if a["location"] == b["location"]:
                    continue
                matches = by_label.get(a["label"] ^ b["label"], ())
                for k in matches[bisect_right(matches, j):]:
                    if items[k]["location"] not in (a["location"], b["location"]):
                        yield (i, j, k)
