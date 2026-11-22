# Solver optimisation study (2026-11-21, GPT-6.2 Sol; prototypes only, nothing applied)

Prototypes: `opt/sparse_core.py`, `opt/bench.py`, `opt/bench_scale.py`. No existing file was modified.
Measured on the real N = 2,000 data on the 4-vCPU VPS, under concurrent load (timings are approximate).

## Headline

Keep the 1e-6 floor of the k-NN rule but stop storing the walk densely. With A the binary union k-NN adjacency,
eta = 1e-6 and d the Sinkhorn scaling, the lazy walk is exactly sparse plus rank one:

    M = I/2 + (1 - eta)/2 * D A D - eta/2 * D^2 + eta/2 * d d^T .

Slice centring does not remove the outer product unless d is constant; dropping the floor instead changes the
first 16 eigenvalues by up to 0.25%.

| Phase (N = 2,000) | Current | Prototype |
|---|---|---|
| Walk storage, 12 sites | 384 MB | 7.5 MB (agrees to 2.3e-13 entrywise) |
| Sinkhorn, all sites | 49 s | 0.19 s |
| Site spectra for t* | 197 s (12 full eigendecompositions) | 2.7-6.6 s (spectrum ends only) |
| t* search | 38 s | 5-8 s (bracketed root; t* matches to 4e-14) |
| 16-mode solve, t = t* | 13.8 s inverses + 9.6 s modes | 2.0 s, direct sparse Lanczos (3.3-11.8x over three couplings) |
| 256-mode solve | 195 s (ncv = 384) | 145 s with polynomial filter (I - L/b)^16; eigenvalues to 1.7e-14, subspace 4.2e-11 |

## Ranked recommendations

1. Compact walk (CSR + one outer product per site) and sparse Sinkhorn; keep Newton as a fallback.
2. Partial site spectra and a bracketed root search for t* instead of full eigendecompositions.
3. Direct sparse Lanczos for tens of modes (slice-constant sector shifted above the target spectrum); the
   polynomial filter for hundreds (checked at one coupling only).
4. For dense (Gaussian-type) kernels: SPD `potrf`/`potri` instead of general inverses (setup 1.9-3.7x faster),
   and a flattened matmul for the Ritz contraction (0.28 s vs 1.25 s at K = 256).

Not worth it: sparse LU of the block system (fill-in grows fast), nested shift-invert with CG, LOBPCG with the
tested preconditioner, float32 factors (subspace error 4.7e-5), reusing factors across couplings.

## Projections (one coupling)

| N | Modes | Current dense: time / RAM | Sparse: time / RAM |
|---|---|---|---|
| 10,000 | 16 | 30-75 min / 22-27 GB | 12-30 s / 0.25-0.4 GB |
| 10,000 | 256 | 1.3-2.5 h / 23-28 GB | 10-30 min / 0.8-1.2 GB |
| 50,000 | 16 | 2.5-6 days / 550-675 GB | 1-4 min / 0.9-1.3 GB |
| 50,000 | 256 | 3-7 days / 555-680 GB | 1-4 h / 3.5-4.8 GB |

Scalings assumed: dense setup N^3, inverse applications N^2; sparse throughput from synthetic fixed-degree graphs,
not iteration counts on real larger graphs.

## Smooth (dense) kernels

Compact storage and sparse Sinkhorn do not carry over exactly. Partial spectra, bounded Krylov memory, the matmul
and the SPD path do. Truncating to a few hundred neighbours or landmark Nystrom changes the operator: rebalance
and measure the drift in t*, eigenvalues and subspaces before accepting either.

## Proposed changes (not applied)

- `coherent.py`: a compact walk type with vector and block products, preserving the floor and diagonal convention.
- `spacetime.py`: accept a walk operator; dispatch compact walks to direct Lanczos, filter opt-in for large K;
  SPD inversion on the dense path; matmul Ritz contraction; guard modes at the cutoff.
- `spacetime_stage.py`: partial site spectra, Frobenius statistic without the dense non-lazy stack, one bracketed
  t* per scale, streamed mode output.
- `test_spacetime.py`: compact-vs-dense walk and t* checks, subspace comparison over a degenerate group.

## Limits

Dense multi-variant sweeps hit their time caps; the 256-mode comparison covers one coupling; no GPU, no Gaussian
kernel and no full large-N stage was benchmarked.
