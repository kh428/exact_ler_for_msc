# Probabilities, series and exact arithmetic

For the specified circuit and noiseless final code/logical readout,

```text
A(p) = Pr(accepted),
B(p) = Pr(accepted and logical error),
P_L(p) = B(p)/A(p).
```

The names `d=3` and `d=5` refer to the underlying colour codes. The leading
physical fault orders for these noisy circuits are respectively two and three.

## Fixed noise strength

The exact point files store fractions as numerator/denominator strings. For each
prime `q`, eight signed character contributions are averaged modulo `q`.
If `D` is the supplied sufficient denominator, `N_A=D A` and `N_B=D B` are
integers with `0 <= N_B <= N_A <= D`. The verifier reconstructs those integers
separately and checks that the product of primes exceeds `D`, before forming
`N_B/N_A`. One additional prime checks every grid point. All primes have Proth
primality witnesses and exceed `2^61`.

The stored denominator accounting includes the exact first-stage denominator,
final-response factor denominators, growth-channel denominators and Fourier
normalisation. The `p=1/1000` certificate is the repeat using cached exact
endpoint values; the documented one-prime command rebuilds its final endpoint.

## Series coefficients

The raw positive fault-weight sums use

```text
A(p) = (1-p)^N sum_k a_k x^k,
B(p) = (1-p)^N sum_k e_k x^k,       x=p/(1-p).
```

| Circuit | Physical source locations | Retained N | Identity-response locations removed |
|---|---:|---:|---:|
| Standalone d=3 | 518 | 402 | 116 |
| Complete d=5 | 3564 | 1996 | 1568 |

The common factor cancels in `P_L`. With `a_0=1`, the quotient recursion is

```text
L_k = e_k - sum_(j=1)^k a_j L_(k-j).
```

Thus the stored result is `P_L(x)=sum_(k=0)^10 L_k x^k + O(x^11)`.
The first nonzero terms are `(32/75)x^2` and `(574/375)x^3`. The `L_k` include
acceptance corrections and need not be positive. To count all physical
locations instead, multiply both raw polynomials by `(1+x)^(removed locations)`.
The quotient is unchanged.

The conservative coefficient bound is

```text
Q_3=2^11, Q_5=2^31,
Q_d 15^k a_k and Q_d 15^k e_k are nonnegative integers,
0 <= e_k <= a_k <= binomial(N,k).
```

The product of reconstruction primes must therefore exceed
`Q_d 15^k binomial(N,k)`. At order ten this bound has 115 bits for `d=3` and
158 bits for `d=5`. Two primes reconstruct the `d=3` coefficients and a third
checks them. All three supplied primes reconstruct the `d=5` coefficients.

## Omitted accepted mass

Let `A_K,E_K` include physical fault weights through `K`, with `(1-p)^N`
restored. With exact `A`, the omitted accepted mass is `delta=A-A_K` and

```text
E_K/A <= P_L <= (E_K+delta)/A.
```

This follows from `0 <= B-E_K <= delta`. It bounds the complete LER and is
different from the residual of `sum_(k=0)^K L_k x^k`. The latter can be measured
directly at the stored exact points. No radius of convergence is inferred from
agreement on a finite grid. All CSV decimals are rounded displays; JSON fraction
strings and modular residues are the authoritative arithmetic data.
