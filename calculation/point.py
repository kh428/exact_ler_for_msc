"""Final Pauli responses and the complete noisy-growth average at one prime.

The first-stage distribution is supplied as an exact rational table. The final
response is rebuilt from the circuit-derived model. All eight growth characters
are retained, so the calculation includes every physical fault order.
"""
from fractions import Fraction
import json
from pathlib import Path
import numpy as np

from .character_surrogate import canonical_basis
from .exact_open_hybrid import exact_open_hybrid
from .modular_binary_contraction import ModularNetwork, ModularWorkspace, contract_modular
from .native_modular_binary import BinaryField
from .native_modular_growth import GrowthField
from .native_polynomial_growth import PolynomialGrowthField
from .rational_binary_network import contract_reference, residue
from .tree_traversal import TreeTraversal
from .walsh import walsh_low

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT/relative).read_text())


def first_weights(point_name):
    table = load(f"data/endpoints/{point_name}_first_stage.json")
    numerators, denominator = table["numerators"], int(table["denominator"])
    # Stored axes are X syndrome, Z syndrome, I/X/Z/Y logical tag.
    return [Fraction(int(numerators[i&7][(i>>3)&7][((i>>7)&1)|(((i>>6)&1)<<1)]), denominator)
            for i in range(256)]


def final_response(p, prime, build_dir, progress=print):
    model = load("data/inputs/d5_soft_source_noise_20260905.json")
    field = BinaryField(prime, build_dir)
    tables, checks = {}, []
    for central in (0, 1):
        saved = load(f"data/models/final_response_plan_c{central}.json")
        for component in (0, 1):
            network = exact_open_hybrid(model, Fraction(p), component)
            keep = sorted(network.boundary_y[:-1]+network.boundary_a)
            network = network.condition({network.boundary_y[-1]: central})
            network.simplify(protected=keep)
            if (keep != saved["output_variables"] or sorted(network.variables) != saved["variables"]
                    or [list(f.scope) for f in network.factors] != saved["scopes"]):
                raise ValueError("Response network differs from its saved contraction plan")
            variables = saved["conditioning"]["conditioned_variables"]
            if len(variables) != 7:
                raise ValueError("Expected seven conditioned response indices")
            compiled = ModularNetwork(network, field)
            workspace = ModularWorkspace()
            total = np.zeros((2,)*len(keep), dtype=np.uint64)
            for index in range(128):
                assignment = {v: index>>j&1 for j, v in enumerate(variables)}
                value, _ = contract_modular(compiled.condition(assignment), saved["conditioning"]["plan"],
                    output_variables=keep, workspace=workspace, max_live_bytes=1700*2**20)
                field.add(total, value)
                if index == 0:
                    nonzero = np.flatnonzero(value)
                    for flat in (int(nonzero[0]), int(nonzero[-1])):
                        bits = np.unravel_index(flat, value.shape)
                        reference = network.condition(assignment | dict(zip(keep, map(int, bits))))
                        reference.simplify()
                        wanted, _ = contract_reference(reference, reference.plan(), prime, max_entries=1<<20)
                        if int(wanted) != int(value.flat[flat]):
                            raise ValueError("Response slice disagrees with Python integer contraction")
                        checks.append({"central": central, "component": component,
                                       "slice": index, "entry": flat, "residue": int(wanted)})
            values = total.transpose(tuple(reversed(range(total.ndim)))).reshape(1024, 512).copy()
            for a, y in ((0, 0), (1023, 511)):
                word = y | (a<<9)
                reference = network.condition({v: word>>j&1 for j, v in enumerate(keep)})
                reference.simplify()
                wanted, _ = contract_reference(reference, reference.plan(), prime, max_entries=1<<20)
                if int(wanted) != int(values[a, y]):
                    raise ValueError("Response slice sum disagrees with unsliced Python contraction")
                checks.append({"central": central, "component": component,
                               "incoming_X": a, "face_character": y, "residue": int(wanted)})
            tables[central, component] = values
            progress(f"Final response component ({central}, {component}): 128/128 slices")
    transformed = []
    for component in (0, 1):
        values = np.concatenate((tables[0, component], tables[1, component]), axis=1)
        walsh_low(values.reshape(-1), 10, prime, inverse=True)
        transformed.append(values)
    acceptance, signed = transformed
    difference = (acceptance+np.uint64(prime)-signed) % np.uint64(prime)
    bad = (difference+(difference & np.uint64(1))*np.uint64(prime)) >> np.uint64(1)
    return np.stack((acceptance, bad), axis=1), checks


def growth_groups():
    model = load("data/models/growth_frame_model.json")["model"]
    merged = {}
    for channel in model["channels"]:
        basis = canonical_basis(channel["generator_labels"])
        merged.setdefault(basis, {"basis": basis, "channels": []})["channels"].append(channel)
    groups = sorted(merged.values(), key=lambda g: min(c["source_location"] for c in g["channels"]))
    return ([{"basis": tuple(1<<i for i in list(range(49,55))+[73,74]), "channels": []}]
            + groups + [{"basis": tuple(1<<i for i in range(55,75)), "channels": []}])


class PointExecutor(TreeTraversal):
    def __init__(self, p, prime, weights, response, build_dir):
        self.prime, self.p = prime, Fraction(p)
        self.candidate = load("data/plan/candidate.json")
        self.nodes, self.root = self.candidate["plan"]["nodes"], self.candidate["plan"]["root"]
        self.characters = self.candidate["all_slices"]
        groups = growth_groups()
        if len(groups) != 569 or self.characters != 8:
            raise ValueError("Growth factor/character count differs from plan")
        self.field = BinaryField(prime, build_dir)
        self.native = GrowthField(self.field, build_dir)
        self.projector = PolynomialGrowthField(self.field, build_dir)
        self.has_final, self.left_first = {}, {}
        self.peak, _ = self._measure(self.root)
        with np.load(ROOT/"data/plan/maps.npz", allow_pickle=False) as data:
            self.maps = {f: (data[f"d{f}"], data[f"t{f}"],
                            self.candidate["leaves"][str(f)]["output_bits"]) for f in range(569)}
        self.physical = {}
        slopes = {"flip": Fraction(2), "depol1": Fraction(4,3), "depol2": Fraction(16,15)}
        for f, group in enumerate(groups):
            n = 1<<len(group["basis"])
            if f == 0:
                values = (np.asarray([residue(v,prime) for v in weights], dtype=np.uint64),)
            elif f == 568:
                i = np.arange(n, dtype=np.uint32)
                x = ((i>>9)&511)|((i>>10)&512)
                z = (i&511)|((i>>9)&512)
                values = tuple(response[x, o, z].copy() for o in (0,1))
            else:
                damping = 1
                for channel in group["channels"]:
                    damping = damping*residue(1-slopes[channel["kind"]]*self.p,prime)%prime
                uniform = (1-damping)*pow(n,-1,prime)%prime
                vector = np.full(n, uniform, dtype=np.uint64)
                vector[0] = (int(vector[0])+damping)%prime
                values = (vector,)
            self.physical[f] = tuple(np.ascontiguousarray(v[:,None]) for v in values)
