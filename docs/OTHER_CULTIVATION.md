# Additional circuit models and comparison data

For every model, `A` is the probability of acceptance per initial attempt and
`B` is the probability of acceptance **and** a wrong logical readout. The
conditional logical error rate is `B/A`. No decoder or escape stage is included.

The files in `data/alternatives/series/` store rational coefficients as strings.
For the location convention stated in each file,

```text
A(p) = (1-p)^N Σ a_k x^k
B(p) = (1-p)^N Σ e_k x^k
P_L(p) = B(p)/A(p),       x = p/(1-p).
```

The common factor cancels in `P_L`. Some d=5 records contain only the acceptance
coefficients needed for the displayed quotient: if the first nonzero error
coefficient is at degree d, `L` through degree K needs `a` only through K-d.
An absent acceptance coefficient is unknown, not zero. The evaluator checks
this when it divides the series.

## RP²

The inputs are the supplied cultivation-only circuits from
[Chen et al.](https://arxiv.org/abs/2503.18657) and
[Cultiv_T_RP2](https://github.com/Zihan-Chen-PhMA/Cultiv_T_RP2).
The supplied Stim files are S proxies despite the `T` in their original names.
The frozen models convert the audited injection and double-checking S layers
to T layers, retaining gate signs, noise placement, measurement records,
feed-forward, detector reference parities and terminal code projection.

There are 678 physical noise locations at d=3 and 2,405 at d=5. The supplied
CSV files give one noise strength, `p=0.001`, for each distance. They contain
S-proxy samples, so agreement with them would not validate the actual-T
calculation. The comparison uses
`errors / (shots - discards)` and the likelihood intervals recorded in
`comparisons/rp2.json`; they are not relabelled as confidence intervals.

## Fold-transversal

The model follows [Sahay et al.](https://arxiv.org/abs/2509.05212).
At d=3 it is the 374-location terminated native-T core with unitary injection.
It excludes the noisy pre-escape syndrome round. Terminal code and logical
projections are noiseless.

The d=5 model is the updated 2,116-location native-T reconstruction. It uses
hook injection, `(r1,r2)=(2,0)` and controlled-layer counts `3,3,7,7`. Native
two- and three-qubit gates have depolarising noise with total probability p;
unused sites have idle noise per native layer. Other source noise is retained.
Three-qubit depolarisation introduces the denominator 63, so the coefficient
accounting uses 315 rather than 15. The input model records the detailed
measurement and readout convention. This reconstruction has not been matched
to an author-supplied noisy T circuit.

The comparison figure is a diagnostic of the d=3 plotting convention. Its Y
proxy uses independent sampled counts. Its T entries use an all-orders
binary64 evaluation, not an exact fraction or a rigorous numerical enclosure.
The original no-escape plotting script and its raw counts were unavailable.

All four Y-proxy per-attempt estimates lie inside the digitised published
bars. Three of the four T per-attempt values do; the value at `p=0.007` lies
below the lower bar. The conditional values miss the three larger-p bars in
both panels. This suggests a difference in normalisation or error definition;
it does not identify a confirmed bug in the authors' code.

## Repaired colour-code cultivation

The arXiv v1 inputs are the flagged `d3a6f2` and `d5a19f13` circuits accompanying
[Chan et al.](https://arxiv.org/abs/2609.17706). They include injection and the
full cultivation sequence, with actual T gates and noiseless final readout.
The d=5 coefficients count 2,308 relevant locations out of 5,071 physical
locations. Noise with identity response is summed exactly; changing this
convention changes raw `a_k,e_k`, but not the conditional `L_k`.

The separate `chan-four-round-d5` case uses the author-supplied
`d5a19r4f13` source. These are the four-round files called arXiv v2 in the
appendix. The package identifies the files by checksum rather than assuming
that an online version has a particular circuit. There are 2,576 relevant
locations. The first nonzero term is degree five and the saved calculation
extends through degree six.

Every plotted Chan sample belongs to arXiv v1. The four-round curve is a finite
series without new sampling points. At `p=0.001`, the two degree-six sums differ
by about 0.57%; that comparison has no bound on the omitted terms and does not
establish an all-orders performance difference.

The ZX records check an explicit four-fault pattern. It has `A=B=1/4` in the
v1 source and is rejected by the four-round source. These checks use rational
ZX phases with complex floating-point scalar sums, tested to `1e-12`. They are
separate from the exact arithmetic certificate of the coefficients.

## Provenance

`provenance.json` records the original checksums of copied inputs and results.
`manifest.json` at the package root checks the files distributed here. These
serve different purposes: historical implementation hashes identify the code
used for the original run; the package hash identifies the refactored copy.
Neither is a signature or a proof that the circuit compiler is correct.

The original RP² MIT notice is retained in `licenses/RP2-MIT.txt`. Author-supplied
circuits and numerical data retain their source attribution; the package's
licence does not replace an upstream licence. Correspondence and draft papers
are not part of this package.
