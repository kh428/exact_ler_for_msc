"""Plot the saved alternative-circuit series and comparison data. No simulation."""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path

from alternatives.results import DATA, evaluate, latex_series, load_case, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory for four PDF/PNG figures and the equations")
    parser.add_argument("--dpi", type=int, default=500)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new directory")
    if not 72 <= args.dpi <= 1200:
        parser.error("Choose a DPI between 72 and 1200")
    output.mkdir(parents=True)
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    import numpy as np

    plt.rcParams.update({"font.size": 9.3, "axes.linewidth": .8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    red, green, blue, orange = "#DB3333", "#33A640", "#3A50BE", "#E38B1F"
    checks = {}

    def curve(case, p, order=None):
        record = load_case(case)
        k = record["degree"] if order is None else order
        x = np.asarray(p) / (1 - np.asarray(p))
        return np.polynomial.polynomial.polyval(x, [float(Fraction(v)) for v in record["L"][:k+1]])

    def style(ax, title, limits, ylabel=r"$P_L$"):
        ax.set(xscale="log", yscale="log", xlim=limits,
               xlabel=r"Circuit-level noise strength ($p$)", ylabel=ylabel)
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=.3, which="both", linewidth=.5)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.legend(fontsize=8, frameon=False)

    def save(fig, stem, *, layout=True):
        if layout:
            fig.tight_layout(pad=.8, w_pad=1.8)
        fig.savefig(output / f"{stem}.pdf", metadata={"Creator": "Matplotlib", "Title": stem,
                                                     "CreationDate": None, "ModDate": None})
        fig.savefig(output / f"{stem}.png", dpi=args.dpi)
        plt.close(fig)

    proxy = read_json(DATA / "comparisons/rp2.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.55))
    for ax, d in zip(axes, (3, 5)):
        case, sample = f"rp2-d{d}", proxy[f"d{d}"]
        p = np.geomspace(3e-4, .003, 400)
        ax.plot(p, curve(case, p), color=red, lw=1.9, label=r"$T$")
        ax.errorbar([sample["p"]], [sample["LER"]],
                    yerr=[[sample["LER"]-sample["lower"]], [sample["upper"]-sample["LER"]]],
                    color=green, fmt="s", ms=4, capsize=2, label="Chen et al.: S proxy")
        value = float(evaluate(load_case(case), "1/1000"))
        ratio = value / sample["LER"]
        ax.text(.97, .06, rf"$P_L^{{({d})}}/(\mathrm{{proxy\ prob}})={ratio:.2f}$ at $p=10^{{-3}}$",
                transform=ax.transAxes, ha="right", fontsize=8.4)
        style(ax, rf"$\mathbb{{RP}}^2$, $d={d}$", (3e-4, .003))
        checks[case] = {"series_order": load_case(case)["degree"], "p001_series": value,
                        "p001_proxy": sample["LER"], "ratio": ratio,
                        "meaning": "Finite actual-T series compared with S-proxy samples"}
    save(fig, "rp2_comparison")

    fold = read_json(DATA / "comparisons/fold.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7))
    for ax, key, title in zip(axes, ("proxy", "native"), ("Fold d=3: Y proxy", "Fold d=3: T")):
        rows = fold[key]
        p = [r["p"] for r in rows]
        y = [r["paper_ordinate"] for r in rows]
        err = [[r["paper_ordinate"]-r["paper_lower"] for r in rows],
               [r["paper_upper"]-r["paper_ordinate"] for r in rows]]
        ax.errorbar(p, y, yerr=err, color="#333333", fmt="o", ms=4, capsize=2,
                    label="Sahay et al. (digitised)")
        suffix = " (sampled)" if key == "proxy" else " (floating)"
        ax.plot(p, [r["B"] for r in rows], color=red, marker="s", ms=3.4,
                label=r"$B$: per attempt" + suffix)
        ax.plot(p, [r["PL"] for r in rows], color=blue, marker="^", ms=3.4,
                label=r"$P_L=B/A$: conditional" + suffix)
        style(ax, title, (.00085, .008), "Plotted error quantity")
        checks["fold-"+key] = {
            "B_inside_published_bar": [r["paper_lower"] <= r["B"] <= r["paper_upper"] for r in rows],
            "PL_inside_published_bar": [r["paper_lower"] <= r["PL"] <= r["paper_upper"] for r in rows],
            "convention": fold["conventions"][key]}
    save(fig, "fold_comparisons")

    def chan_points(d):
        data = read_json(DATA / f"comparisons/chan_v1_d{d}.json")
        return data["flagged"]["points"] if d == 3 else data["points"]

    def sampling(ax, rows):
        p = np.array([r["p"] for r in rows])
        y = np.array([r["LER"] for r in rows])
        lo = np.array([r["lower"] for r in rows])
        hi = np.array([r["upper"] for r in rows])
        ax.errorbar(p, y, yerr=[y-lo, hi-y], color=orange, fmt="o", ms=4,
                    elinewidth=.8, capsize=2, label="Chan et al.: T sampling (arXiv v1)")
        ax.fill_between(p, lo, hi, color=orange, alpha=.14, linewidth=0)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.65))
    for ax, d in zip(axes, (3, 5)):
        rows, case = chan_points(d), f"chan-v1-d{d}"
        p = np.geomspace(min(r["p"] for r in rows), max(r["p"] for r in rows), 400)
        k = load_case(case)["degree"]
        ax.plot(p, curve(case, p), color=red, lw=1.9, label=rf"Through $x^{{{k}}}$")
        sampling(ax, rows)
        style(ax, rf"Chan et al. arXiv v1: repaired $d={d}$", (p[0]*.85, p[-1]*1.15))
    save(fig, "chan_v1_comparison")

    rows = chan_points(5)
    p = np.geomspace(.0005, .002, 500)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.plot(p, curve("chan-v1-d5", p), color=red, lw=1.8, label=r"arXiv v1 repair, through $x^7$")
    ax.plot(p, curve("chan-four-round-d5", p), color="#277D46", ls="--", lw=1.8,
            label=r"Four-round repair, through $x^6$")
    sampling(ax, rows)
    style(ax, "Chan et al.: repaired d=5 cultivation", (.00045, .0022))
    inset = ax.inset_axes([.59, .17, .36, .22])
    inset.plot(p, curve("chan-four-round-d5", p, 6)/curve("chan-v1-d5", p, 6), color="#277D46")
    inset.set(xscale="log", ylim=(.988, 1.001), title=r"Same-order ratio: $S_6^{\rm four}/S_6^{\rm v1}$")
    inset.tick_params(labelsize=7)
    inset.title.set_fontsize(8)
    inset.xaxis.set_minor_formatter(NullFormatter())
    inset.grid(alpha=.25)
    fig.subplots_adjust(left=.11, right=.98, top=.93, bottom=.35)
    for y, name, title in ((.23, "chan-v1-d5", "arXiv v1 repair"), (.10, "chan-four-round-d5", "Four-round repair")):
        fig.text(.12, y+.043, title, fontsize=9)
        fig.text(.12, y, r"$P_L^{(5)}(x)=" + latex_series(load_case(name)) + "$", fontsize=9)
    fig.text(.12, .025, r"$x=p/(1-p)$. No escape. Finite series; all sampling points belong to arXiv v1.", fontsize=8.5)
    save(fig, "chan_four_round_comparison", layout=False)
    a = evaluate(load_case("chan-four-round-d5"), "1/1000", 6)
    b = evaluate(load_case("chan-v1-d5"), "1/1000", 6)
    checks["chan-four-round"] = {"p001_new_S6": float(a), "p001_old_S6": float(b),
                                 "same_order_relative_difference": float(a/b-1), "new_sampling": False}
    (output / "plot_checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    from alternatives.results import cases
    equations = ["% Exact coefficients of finite series; no numerical remainder bound.\n"]
    for case in cases():
        row = load_case(case)
        equations += ["% " + case + "\n", "\\[P_L^{("+str(row["distance"])+")}(x)="+latex_series(row)+"\\]\n"]
    (output / "series.tex").write_text("\n".join(equations))
    print(f"Saved four PDF/PNG figures, equations and numerical plot checks in {output}")


if __name__ == "__main__":
    main()
