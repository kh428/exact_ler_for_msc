# Calculation guide

## Two forms of the result

`verify_results.py` is the shortest entry point. It uses integer and rational
arithmetic to reconstruct the archived probabilities and coefficients. Each
validation step is a separate named function; the tests include altered prime
witnesses, duplicated characters and inconsistent held-out residues.

`recompute_prime.py` executes one complete modular calculation. It reads the
exact first-stage weights, rebuilds the final response, then joins every growth
factor. It is useful for checking the contraction independently of the archived
character values. Repeating it over the saved prime pool supplies the residues
used by the fraction reconstruction. It does not regenerate the first-stage
weights from the original circuit.

## Order of operations

1. `exact_open_hybrid.py` constructs rational gate, noise, parity and projection
   factors for a double-checking response. Its incoming indices remain open.
2. `modular_binary_contraction.py` reduces those factors modulo a certified prime
   and eliminates indices in the supplied order. Seven fixed indices give 128
   slices for each of four endpoint components.
3. `point.final_response` sums those slices and applies the inverse Walsh
   transform, giving the acceptance and accepted-error response tables.
4. `point.growth_groups` groups channels with equal binary label spans.
   Together with the two endpoints there are 569 factors.
5. `point.PointExecutor` projects the factors onto each of eight characters.
   `tree_traversal.py` evaluates the same exact tree for both observables,
   sharing subtrees wherever possible.
6. `verify_results.crt` reconstructs integer numerators after all required primes
   have been evaluated. The acceptance numerator is never divided out modulo
   a prime; the final ratio is formed over the rationals.

## Arithmetic modules

| Module | Role |
|---|---|
| `rational_binary_network.py` | Rational factors and Python-integer reference contraction |
| `character_surrogate.py`, `frame_boundary.py` | Binary labels and stabiliser-equivalent Pauli classes |
| `native_modular_binary.py`, `modular_binary.cpp` | Montgomery arithmetic and binary broadcasts |
| `native_modular_growth.py`, `modular_growth.cpp` | Restricted XOR joins, coordinate changes and Walsh transforms |
| `truncated_polynomial.py` | Exact scalars in `F_q[x]/(x^(K+1))`, through degree ten |
| `polynomial_open_hybrid.py` | Response factors with polynomial noise weights |
| `polynomial_binary_contraction.py` | Polynomial-factor elimination |
| `polynomial_channel_law.py` | Exact physical-weight endpoint distributions |
| `native_polynomial_growth.py` | Coefficient-streamed growth products |

The polynomial source modules implement the arithmetic used for the series.
The supplied public command reconstructs the complete series from saved modular
coefficients; it does not launch a new full degree-ten campaign.

The native library loader compiles only the C++ files in this package with an
installed C++17 compiler. Compilation and numerical outputs are written beneath
the requested output directory. No package installation is performed by the
calculation commands.
