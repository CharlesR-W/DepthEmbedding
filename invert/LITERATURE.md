# Inverting a spectral (supra-Laplacian) embedding of activations: what the literature offers

Written 2026-11-21. The sources were read (arXiv source, author PDFs, publisher pages) by me and by five research subagents working in parallel; every paper that could not be opened is listed in §11. A GPT-6.2 reviewer then checked the derivations and recommendations independently, and its corrections are folded in. **[paper]** marks a statement a paper makes; **[inference]** marks our own reasoning or derivation. Unmarked formulas are standard.

---

## 0. Summary and ranked recommendation

**The problem has two halves that the literature treats separately.** The *forward* out-of-sample map (activation → coordinates) is a solved problem for a smooth Sinkhorn kernel. The answer is a half Sinkhorn step with the probe's scalings held fixed, followed, for the joint operator, by one L×L tridiagonal solve per mode (§2). The *inverse* (coordinates → activation) is not solved in your regime. Every comparison we reviewed uses 2–4 coordinates that fully parametrise a low-dimensional manifold. Yours are 16 coordinates on a cloud of intrinsic dimension 40–77, so any decoder can at best estimate the conditional mean E[x | y], and its round trip cannot be exact without carrying information from somewhere else (§1).

Ranked on being principled and low-tuning first, accuracy second:

1. **Conditional-mean decoder by kernel ridge / Gaussian-process regression on the coordinates, with hyperparameters chosen by leave-one-out (LOO) error, evaluated by held-out images** (§3.6). x̂(y) = trend(y) + k(y)ᵀ(K + σ²I)⁻¹(X − trend). Climb a ladder and stop when held-out error stops improving: the mean, then the affine decoder (`dec_linear`), then isotropic KRR with a small bandwidth × ridge search (`dec_krr`), and per-coordinate (ARD) length scales only if they beat isotropic KRR on held-out images. The fit and its LOO score need only the probe's cosine Gram. Far from the data it reverts to the trend, or to the mean without one. Geometric harmonics is the same kind of estimator with a spectral cut-off in place of the ridge (§3.1), so it is a variant, not a competitor.
2. **Auto-adaptive Laplacian pyramid (ALP) with the vector LOO stopping rule** (§3.4). The only parameter is a starting width, and it has a published rule. No matrix is inverted, it costs O(LN²), and early stopping suits a conditional-mean target. Only its first level is confined to the convex hull of the probe activations; later levels add signed corrections. Far from the data it carries a fixed combination of values at the extreme probe point outward unchanged.
3. **Forward-consistent pre-image by "matching" around a reference** (§6.2). Use it only if an exact round trip is a requirement, because it optimises consistency, not accuracy. Solve x* = argmin_{‖x‖=1} ‖x − x₀‖² subject to F_l(x) = y by tangent-space SQP, with the closed-form Jacobian of the smooth forward map. Here x₀ is design 1's mean, a retrieved real probe activation, or the residual-carrying guess x₀ = g(y) + (x_ref − g(y_ref)). It is the only design with an exact round trip (when it converges), and it is the closest principled form of your "residual" idea: Debrabant–Samaey L2 matching from equation-free computing, and Thorstensen/Étyngier's Nyström-constrained Karcher mean from shape analysis. It needs a smooth forward kernel, and it is infeasible for targets outside the forward map's range (see the next paragraph).
4. **The affine decoder as the extrapolation control** (`dec_linear`: least squares on span{1, coordinates}, §5.1). Use it to show what linear extrapolation in coordinates produces. It is the cleanest version: the LLE barycentric decoder of Saul & Roweis is local but discontinuous, and with its trace-scaled regulariser it reverts towards the neighbour mean far away instead of extrapolating.

**The obstacle no method removes [inference].** A Nyström-type forward map built from a row-stochastic kernel row sends every activation, including out-of-distribution and adversarial ones, into a bounded set: a positive weighting of probe coordinates passed through a fixed linear map (1/λ, or the chain solve of §2.3). Coordinates outside that set are the image of no activation. For them no decoder can have an exact round trip, and "the activation at that point" has no meaning that the literature (or the geometry) supplies; each decoder's off-data behaviour is a convention (§8). The hull-type bound is only an outer bound: not every point inside it is attained, because the kernel weights of a single unit vector are jointly constrained. Real out-of-distribution images never hit this problem, because the forward map is defined for them.

**For transfer between networks** (§7): with shared probe images, the functional map between the two networks' spectral bases is a K×K least-squares matrix. To predict network B's coordinate row from network A's at site l, use y_B ≈ y_A W_l with W_l = Φ_{A,l}⁺Φ_{B,l}. The principal angles between the two spectral subspaces (the similarity measure) come from the singular values of Q_BᵀQ_A, with Q orthonormal bases of each Φ's column space. Decoding into network B from A's coordinates is design 1 with B's activations as targets.

---

## 1. Setting, notation, and three facts that shape everything

At site l: X_l ∈ ℝ^{N×D} holds the probe activations, centred and with unit-norm rows, and G_l = X_l X_lᵀ is their cosine Gram. P_l is the doubly stochastic walk; the joint operator is L_t = blockdiag((I − M_l)/c_l) + t(P_path ⊗ I_N), with eigenpairs (μ_k, f_k) and f_k = (f_{k,1}, …, f_{k,L}), f_{k,l} ∈ ℝᴺ. A query has coordinates y ∈ ℝᴷ at site l.

**Fact 1: the decoder can only target a conditional mean, and its norm measures how much it explains [inference].** Since ‖x‖ = 1 for every activation, for the true conditional mean g(y) = E[x | y]:

  E[‖x‖² | y] = 1 = ‖g(y)‖² + E[‖x − g(y)‖² | y].

So ‖g(y)‖² is the fraction of an activation's "energy" that the coordinates determine at y. With neighbours at cosine ≈ 0.3 and intrinsic dimension 40–77, expect ‖g‖ well below 1. That is suggestive, not proven: the dimension estimate does not by itself rule out the coordinates being injective on a finite probe set.

The identity holds for the *true* conditional mean only, so the norm of a *fitted* decoder ĥ is not a confidence score. Its expected captured share at y is ‖g(y)‖² − ‖ĥ − g(y)‖², which can be negative even when ‖ĥ‖ = 1. Measure the captured share 2⟨ĥ,x⟩ − ‖ĥ‖² (and the cosine) on held-out images instead. Renormalising g is a legitimate choice if the target is a direction: among unit vectors, g/‖g‖ maximises the expected cosine.

For raw activations a = m_l + ρx, the conditional mean is m_l + E[ρx | y], which is not E[ρ | y]·g(y) in general. If raw squared error is the objective, regress the raw (centred) activations directly.

**Fact 2: interpolation is the wrong target [inference, consistent with every paper below].** Exact interpolators (untruncated GH, kriging without nugget, local RBF, Laplacian pyramid run to convergence, Bermanis' multiscale extension with err = 0) reproduce the part of each probe activation that the coordinates do not explain, as if it were signal. Every published success of these methods is in a regime where coordinates determine the state (v = v(u) in equation-free language); yours is not.

**Fact 3: some of the K joint modes carry no information at a single site [inference from Froyland, Kalia & Koltai].** P_l is doubly stochastic, so (I − M_l)1 = 0. Every vector of the form e_path ⊗ 1_N, with e_path an eigenvector of the path Laplacian, is therefore an exact eigenvector of L_t with eigenvalue t·ν_path. These are Froyland–Kalia–Koltai's "temporal" eigenvectors. Restricted to one site they are constant over images. They then add nothing to an isotropic distance, so they are redundant; they are harmful only if per-coordinate standardisation divides by their near-zero spread and amplifies numerical noise. FKK identify them by the variance of their fibre means (arXiv:2409.11984). If a temporal and a spatial eigenvalue collide, a solver can return mixtures, so project the temporal subspace away rather than dropping columns by a variance threshold. `spacetime.modes` (called by `dm.joint_modes`) already solves on the subspace with zero mean in every site, so the current code returns the lowest *spatial* modes, not the lowest modes of the unrestricted operator. Similarly, some "spatial" modes may be harmonics (functions) of earlier ones. Dsilva, Talmon, Coifman & Kevrekidis (ACHA 2018) give a LOO local-linear-regression test: regress φ_k on φ₁…φ_{k−1}, with bandwidth = median pairwise distance/3; a small normalised residual r_k means φ_k is a harmonic. Conformal autoencoders (G. A. Kevrekidis et al., arXiv:2408.16138) make the same point: "eigenvectors can be (and often are) functionally related".

---

## 2. Forward map and kernel choice (your item 5)

### 2.1 Kernel options

| Kernel | Formula | Parameters and automatic rule | Out-of-sample row | Under distance concentration and hubness |
|---|---|---|---|---|
| Self-tuning (Zelnik-Manor & Perona 2004) | exp(−d²_ij/(σ_iσ_j)), σ_i = distance to 7th NN | k = 7 "in all our experiments" [paper] | Well defined: σ_x from x's kNN among the probe | Dense version does not localise at your numbers [inference]: chord² ≈ 1.4 to the nearest neighbour vs 2.0 to a typical point, so σ_iσ_j ≈ 1.45 gives weights 0.37 vs 0.25; the ~2,000 far points carry ~500 units of row mass against ~6 from the 15 neighbours. That points towards the "degenerates to PCA" regime (K ≈ const + affine in G) for this toy row. Whether the spectrum is actually PCA-like should be checked on the measured row masses and eigenspaces before truncating. Schnitzer et al. call it "local scaling" and show it reduces hubs. |
| Variable bandwidth (Berry & Harlim 2016) | exp(−‖x−y‖²/(4ερ(x)ρ(y))), ρ = q^β, then α-normalisation by q_ε^α and row normalisation; limit Δf + c₁∇f·∇q/q with c₁ = 2 − 2α + dβ + 2β [paper] | β = −½; α = ½ − d/4 gives Laplace–Beltrami; ε by the maximal slope of log Σ_ij K_ε against log ε (Coifman, Shkolnisky, Sigworth & Singer 2008), slope ≈ d/2 [paper] | Every quantity is a training-set sum evaluated at x, so it is well defined (Bengio-style) | With d ≈ 40–77, ρ^d and α ≈ −10 to −19 make the normalisations numerically brittle [inference]. The ε slope rule is still useful on its own. |
| CkNN (Berry & Sauer 2019) | edge iff ‖x−y‖ < δ√(ρ(x)ρ(y)), ρ = k-NN distance | k (10 in their example); δ from a persistence diagram [paper] | Well defined and consistent with the training graph [inference]. A frozen OR-symmetrised kNN graph can also be extended consistently if each probe's k-th-neighbour radius is stored and both directions are tested (with an explicit self-mask and tie rule); the current `knn_oos_rows` does not do this | Unweighted, hence not smooth. Within its class of constructions (and under its assumptions), the unique unweighted one whose Laplacian converges spectrally to the Laplace–Beltrami operator of a conformally changed metric [paper]. |
| Doubly stochastic Gaussian with a global ε (Marshall & Coifman 2019; Wormell & Reich 2021; Landa, Coifman & Kluger 2021; Landa & Cheng 2022; Cheng & Landa 2022) | P = diag(v) K diag(v), K_ij = exp(−‖x_i−x_j‖²/ε), zero diagonal, v_i(Kv)_i = 1 | One global ε. No published automatic rule in these papers; Landa & Cheng use a fixed ε = 0.1. Calibrate by row perplexity (what invert/dm.py does) or by the Coifman log-log slope | Closed-form half step (§2.2) | On the sphere exp(−‖x−y‖²/ε) = e^{−2/ε}e^{2cos/ε}, so P_ij = exp((2cos_ij − r_i − r_j)/ε) with r_i = −ε log v_i + 1 (the constant e^{−2/ε} absorbed into the offsets). That is CSLS's form (Conneau et al. 2018), with offsets chosen to equalise degrees instead of taken from kNN means: a soft CSLS [inference]. It leaves the result unchanged under any per-point offset of the squared distances, which is the mechanism behind hub suppression. Equal column sums bound each point's total incoming *mass*, not its top-k occurrence count. Landa–Coifman–Kluger prove asymptotic robustness to heteroskedastic noise under a noise model. Within a row, P_ij/P_ik = (K_ij/K_ik)(v_j/v_k), so contrast changes only through the column factors; the spread of distances relative to ε still governs localisation, so concentration must be handled by a small enough ε. | Limit operator: −q⁻¹div(q∇f), the α = ½ diffusion-maps operator [Marshall–Coifman Thm. 1; Cheng–Landa]. |
| Symmetric entropic affinities (SNEkhorn; Van Assel, Vayer, Flamary & Courty, NeurIPS 2023) | P = exp((λ⊕λ − 2C) ⊘ (γ⊕γ)), dual ascent on (γ, λ); doubly stochastic; at least n−1 rows have perplexity exactly ξ [paper] | Perplexity ξ | Not given in the paper [paper, source grepped]. Natural extension: hold the training duals fixed and solve two scalar equations (unit row sum, entropy = log ξ + 1) for (γ_x, λ_x) [inference; existence and uniqueness not shown] | Handles concentration by construction: the per-point temperature is set by the local gap in distances, not their absolute size. Toy numbers [inference]: with 15 points at chord² 1.4 and 1,985 at 2.0, a global ε = 0.12 gives the two groups equal total mass (perplexity ≈ 2√(15·1985) ≈ 345), so perplexity ≈ 15 needs ε well below 0.12. SNEkhorn's pair temperatures are not a global ε, so this is only indicative. The published formulation allows diagonal terms. Dense O(n²) per iteration. |
| Hubness rescalings (Schnitzer, Flexer, Schedl & Widmer 2012; Feldbauer & Flexer 2019) | NICDM d_xy/√(μ_xμ_y); mutual proximity MP(d_xy) = P(X > d_xy ∩ Y > d_yx), with a Gaussian independence approximation | k for μ; MP-Gauss-I has no parameter. Feldbauer & Flexer recommend "any of the four methods MP, LS, NICDM, and DSL" [paper] | Well defined (stored per-point statistics) | MP works on per-point z-scores, so it removes concentration directly, but it is a similarity, not a PSD kernel with a diffusion limit [inference]. Good as a diagnostic or a pre-transform. Radovanović et al. 2010: hubs are points near the data mean, and the effect grows with intrinsic dimension [paper]. |

Two attributions to correct. SNEkhorn is by Van Assel, **Vayer, Flamary & Courty**. Van Assel, Espinasse, Chiquet & Picard wrote the related "A probabilistic graph coupling view of dimension reduction" (NeurIPS 2022). Zass & Shashua (NIPS 2006) were read only as an abstract. It argues that Ncut and Ratio-cut implicitly look for the nearest doubly stochastic matrix (under relative entropy and L1) and proposes the Frobenius-nearest one; it has no out-of-sample formula.

**Judgement.** For cosine geometry, use the doubly stochastic Gaussian-on-cosine kernel with zero diagonal and a single ε calibrated by row perplexity (or by the log-log slope). It is smooth, has a limit theory, removes hubs through its scalings, and has a closed-form local out-of-sample row. This is what `invert/dm.py: ds_kernel` already does. If effective neighbourhood sizes still vary a lot across rows, step up to SNEkhorn's per-row entropy control. Its extension costs a two-scalar solve per new point. Avoid dense Zelnik-Manor–Perona and plain Gaussians, which are linear-kernel-like at these distances. Berry–Harlim's α/β normalisations are fragile at d ≈ 50.

### 2.2 The correct Nyström row for a Sinkhorn-scaled kernel

The rule, with the probe scalings v_j held fixed:

  v_x = 1 / Σ_j k(x, x_j) v_j,  p(x, j) = v_x k(x, x_j) v_j = v_j k(x, x_j) / Σ_{j′} v_{j′} k(x, x_{j′}).

This is stated (in different notation) by several sources:
- **Wormell & Reich (SIAM J. Numer. Anal. 2021, §2)** [paper]. They extend the discrete Sinkhorn weights to a function U by U(x)·(D^M U)(x) ≡ 1, with (D^M φ)(x) = (1/M)Σ_i g_ε(x − x^i)φ(x^i), so "ϖ(U) = u". The operator P^M = U D^M U has the matrix's spectrum and eigenvectors. Note that their kernel matrix includes the diagonal.
- **Marshall & Coifman (2019)** [paper]. d(x) = ∫ k(x,y)/d(y) dμ̂(y) is "an immediate consequence of the bi-stochasticity". They give φ_k(x) = λ_k⁻¹∫ b(x,y)φ_k(y)dμ̂(y) and its gradient, ∇φ_k(x) = λ_k⁻¹∫ (y − x̄)/ε · b(x,y)φ_k(y) dμ̂(y) with x̄ = ∫ y b(x,y) dμ̂. Their d is the reciprocal of v.
- **Landa & Cheng (arXiv:2209.08004)** [paper]. The scalings converge to a function ρ_ε solving (πε)^{−d/2}∫ρ_ε(x)K_ε(x,y)ρ_ε(y)q(y)dy = 1, with ‖ρ_ε − q^{−1/2}F_ε‖ = O(ε²). So v ∝ 1/√density is a theorem, and the half step is the empirical version of that equation at a new x.
- **Pooladian & Niles-Weed (arXiv:2109.12004)** [paper] call the same half step for entropic-OT potentials "a canonical extension … to out-of-sample points".
- **Bengio, Paiement, Vincent et al. (NIPS 2003/04, §4.2)** [paper]. The general principle for data-dependent kernels: normalise with expectations over the training points only. Prop. 3: "we do not seek to approximate K̃(x,x)".
- **Fowlkes, Belongie, Chung & Malik (TPAMI 2004)** [paper] take the opposite, transductive route: they re-estimate all degrees including the new points. The Sinkhorn analogue would be re-running Sinkhorn on the augmented set, which moves every probe's coordinates.

**Self-loops [inference, corrected after review].** Train with a zero diagonal (Landa–Coifman–Kluger's choice, because k(0) dwarfs the off-diagonal values under high-dimensional noise). Then, *if probe i is explicitly masked out of its own row*, the half step reproduces v_i and P's row i up to the Sinkhorn tolerance, and the joint solve reproduces the probe's coordinates. This does not refit Sinkhorn without i. The smooth map as a function of x does **not** do this by itself: `oos_rows` gives anchor i positive weight even at x = x_i, and its limit as x → x_i is the *unmasked* row. So there are two forward maps:
- F_masked, which uses identity information and reproduces the probe embedding exactly;
- F_smooth, the continuous function of x that a pre-image solver differentiates.

They differ at the probes. Say which one defines the target, and measure the discrepancy at the probes. If training includes the diagonal, as in Wormell–Reich and Marshall–Coifman, the unmasked smooth map does reproduce the training values, at the price of the self-weight.

**kNN-truncated supports [inference].** If training uses a symmetrised kNN support ("i ∈ kNN(j) or j ∈ kNN(i)"), the out-of-sample row can only use x's own kNN, so probe rows are not reproduced exactly. That is the case for the current 0/1 15-NN rule (`knn_oos_rows`). Either use a dense kernel with small ε (truncating only for speed, at a k that holds more than 99% of the row mass), or use a support rule a new point can evaluate symmetrically (CkNN-style thresholds).

**A free kernel-support diagnostic [inference].** 1/v_x = Σ_j k(x,x_j)v_j is the unnormalised row mass. For probe images it equals 1/v_i, which Landa–Cheng relate asymptotically to √density. Comparing 1/v_x with the same quantity on held-out images (computed with the same self-inclusion convention) flags inputs far from the probe at no extra cost. This matters because the normalised row p(x,·) always sums to 1, so the coordinates alone cannot reveal that x is far from every probe image. It is a diagnostic, not a calibrated density estimate or an adversarial detector with any guarantee in this dimension.

### 2.3 Nyström extension of the joint (supra-Laplacian) modes [inference: our derivation; no paper found]

Froyland & Koltai (CPAM 2023), Froyland, Kalia & Koltai (arXiv:2409.11984), and Banisch & Koltai (Chaos 2017) contain no out-of-sample extension. Lindenbaum et al.'s multi-view diffusion maps (Information Fusion 2020, §5.5) have one, but for a product kernel, not a supra-Laplacian.

Take a new image with activations at all L sites, and its rows p_l(x,·) from §2.2. Write the (l, x) row of L_t f = μf with M_l = (I + P_l)/2, b_l = Σ_j p_l(x,j) f_l(j), and u_l = f_l(x):

  (u_l − b_l)/(2c_l) + t(deg_l u_l − u_{l−1} − u_{l+1}) = μ u_l,  l = 1..L,

i.e. with D_c = diag(1/(2c_l)) and A = D_c + tP_path,

  (A − μ_k I) u_k = D_c b_k.

This is one tridiagonal L×L solve per mode, after computing L kernel rows. Cost: O(L·N·D) for the cosines, O(L·N·K) for the row–mode products, and O(K·L) for the solves.
- **t = 0.** It reduces to ordinary Nyström u = b/(1 − 2c_lμ), where 1 − 2c_lμ is the non-lazy walk eigenvalue.
- **Probes.** With zero-diagonal training and the self-mask of §2.2, it reproduces every probe image's coordinates.
- **Invertibility.** The exact condition is μ_k ∉ σ(A). Being below λ_min(A) is sufficient but not guaranteed for every retained mode. Check the distance from μ_k to σ(A), not only the condition number. Near resonance the Nyström value is ill-determined.

Two other agents derived the same system independently, and `invert/dm.py: joint_forward` implements it (with dense solves).

**When only one site's activation changes** (e.g. inside a pre-image loop) [inference]:
- *Preferred.* Hold the other sites' *activations*, i.e. their b_{l′}, at a reference image's values, and solve the full system. Then u_l responds to b_l with coefficient [(A − μI)⁻¹]_{ll}/(2c_l). This is "change one site's activation, keep the rest of the network state", and it is the map to use in design 3.
- *A cruder surrogate.* Clamp the other sites' *coordinates* at the reference's: u_l = [b_l/(2c_l) + t Σ_{l′∼l} u_{l′}^{ref}] / (1/(2c_l) + t deg_l − μ). This has its own possible zero denominator, and it is not the joint map.
- Running the network forward from site l supplies only the later sites, so it does not remove the need for one of these choices.

---

## 3. The inverse as function extension in coordinate space (your items 1–3)

All of these regress the activation (each of D outputs) on the K coordinates. For fixed hyperparameters each is a linear smoother x̂ = SX, so any held-out error can be *scored* from the N×N Gram G = XXᵀ alone. The exact PRESS (closed-form LOO) identity holds for ridge-type smoothers like KRR, not for every linear smoother. The kernel lives in ℝᴷ, where distance concentration is mild. The coordinates must be put on comparable scales, or given separate length scales, because raw eigenvector entries are ≈ 1/√N with no meaningful relative scale. Each method gives x̂(y) = Σ_i w_i(y) x_i, a weighted combination of probe activations.

### 3.1 Geometric harmonics (Coifman & Lafon, ACHA 21:31–52, 2006)

**Formula [paper].** Restrict a symmetric PSD kernel k to the sample and diagonalise, λ_jψ_j = Kψ_j. Extend each eigenvector by Ψ_j(y) = λ_j⁻¹Σ_i k(y, y_i)ψ_j(y_i) (Def. 2). Truncate to S_δ = {j : λ_j ≥ δλ₀} (Def. 3). Then

  (Ef)(y) = Σ_{j∈S_δ} ⟨f, ψ_j⟩ Ψ_j(y) = k(y)ᵀ Ψ_S Λ_S⁻¹ Ψ_Sᵀ f,

"a truncated pseudo-inverse" of K, whose condition number is bounded by 1/δ.

- **Ill-conditioning.** "Numerically, this extension procedure is extremely ill-conditioned as one divides by the eigenvalues of a compact operator" [paper, §2.1].
- **Acceptance test.** f is (η, δ)-extendable if ‖P_δf‖² ≥ (1 − η)‖f‖² (Def. 7). A function that fails at one scale may pass at a finer one.
- **Relations [paper].** δ = 0 is zeroth-order kriging (simple kriging). The ridge F_γ = (KK* + γI)⁻¹Kf tends to Nyström as γ → 0. Among all RKHS extensions, Ef is the one most concentrated on the sample (Prop. 6).
- **Bandwidth.** A single scale "is essentially arbitrary as it is a priori unrelated to the function f" [paper].
- **The paper's multiscale scheme (Algorithm 2) is not a residual loop** [paper]. It precomputes, for each Laplace–Beltrami eigenfunction (or dyadic packet), the smallest band B_j at which it extends, and sums the separately extended pieces. The commonly cited "iterate on the residual with halving bandwidth" appears later: Lafon, Keller & Coifman 2006 (TPAMI) halve σ until the discarded energy is below τ, which selects one scale per function, and Chiavazzo et al. 2014 iterate on residuals with ε_l = 2^{1−l}ε₀. Bermanis et al. 2013 is in §3.2.
- **Off the data [paper].** Gaussian extensions "do not oscillate off" the set; they decay like the kernel and vanish beyond a distance ∝ σ (Lafon–Keller–Coifman: "beyond this distance, the extension numerically vanishes").
- **Interpolates?** No, unless δ → 0.
- **Round trip.** Not defined (one-directional).
- **Cost.** O(N³) for the eigendecomposition per scale; O(N + |S|) per output per query, or O(ND) per query for full activations.
- **Automatic rules.** None for δ in the paper. A common bandwidth heuristic is σ = 2 × median pairwise distance (Rabin & Fishelov 2017, citing Schclar). For fixed (ε, δ), GH is a linear smoother, and `dec_gh` scores it by the projection formula r_i/(1 − S_ii). That is *not* exact LOO for GH: omitting a point changes the eigenvectors, the leading eigenvalue and the cut-off. Treat it as an approximate score, or use K-fold held-out images. GH is a spectral-cut-off sibling of kernel ridge (§3.6).
- **Related.** Rabin & Fishelov (Appl. Math. Comput. 2017) build higher-order kernels from Gaussian combinations, which keep more energy in the leading modes.

**Lafon, Keller & Coifman, "Data fusion and multicue data matching by diffusion maps" (TPAMI 28(11), 2006) [paper].** The "lifting" there is an alignment step, not an inverse. Each graph is embedded, an affine map is fit on landmarks, and correspondence is the nearest neighbour g(x_i) = argmin_{y∈Ω₂}‖f(x̃_i) − ỹ‖. They argue the extension kernel should differ from the embedding kernel and be "as large as possible" for each function's complexity.

### 3.2 Multiscale extension with interpolative decomposition (Bermanis, Averbuch & Coifman, ACHA 34:15–29, 2013)

**Algorithm [paper].**
1. Set scales ε_s = T/2^s, with T = max{dist(x*, D), 2(diam D/2)²}. T depends on the query, so the coarsest scale always reaches it.
2. At each scale, estimate the numerical rank l(s) from a bound (δ = 0.1).
3. Take a randomized interpolative decomposition G^{(s)} ≈ B^{(s)}P^{(s)}. The selected columns form an "adaptive grid".
4. Fit the residual: c = B^{(s)†}(f − F^{(s−1)}). (The authors' corrections sheet fixes the printed version, which applied the fit to f.)
5. Extend by Σ_k g_{ε_s}(x*, x_{s_k})c_k.
6. Stop when ‖f − F^{(s−1)}‖ ≤ err. "If we take err = 0 then F = f", i.e. it becomes a multiscale interpolant.

**Theory [paper].** Rank bounds only, no error bounds: R_δ ≲ ((2/π)√(ln δ⁻¹))^d × (number of √ε-cubes covering the data).
**Example [paper].** Plain Gaussian kernels had condition numbers 10¹⁸–10²⁰ at coarse scales; the B^{(s)} had 3.6 to 5.9×10⁵.
**Off the data [paper].** Each level decays to 0 (Remark 4.2).

**For you [inference].** In d = 16 the rank bound exceeds N at any useful ε, so the scheme reaches full rank (interpolation) within a few levels. Its value is numerical stability, not a parameter rule. It would need an err > 0 chosen by LOO to target a conditional mean.

### 3.3 Laplacian pyramids (Rabin & Coifman, SDM 2012, pp. 189–199) — paper not accessed; formula as restated by four later papers

**Formula.**
- Normalised kernel at level ℓ: P_ℓ(y, y_j) = G_ℓ(y − y_j)/Σ_{j′}G_ℓ(y − y_{j′}), with σ_ℓ = σ₀/μ^ℓ, μ = 2. This is a Nadaraya–Watson weight, defined at any y.
- Levels: s₀ = P₀f; d_ℓ = f − f_{ℓ−1} on the sample; s_ℓ(y) = Σ_j P_ℓ(y, y_j)d_ℓ(y_j); f_L = Σ_{ℓ≤L} s_ℓ.
- Stop when the training residual is below a tolerance.
- Cost: O(LN²) to train, O(LN) per query per output.

**Analysis (Leeb, arXiv:1909.07974) [paper].**
- LP is Nadaraya–Watson with L2-boosting and a bandwidth that changes per level. The residual factorises: D_ℓ = (I − P_{ℓ−1})⋯(I − P₀).
- For decaying kernels, ‖I − P_ℓ‖ < ε once σ_ℓ < c·δ_min. So LP converges to exact interpolation, and the bandwidths can level off at σ* < cδ_min.
- Stability: ‖f_ℓ‖_∞ ≤ C(σ₀/δ_min)^{log_μ 2}‖f‖_∞.
- Very small bandwidths "introduce high-frequency artifacts"; Leeb suggests σ_ℓ = max(σ₀/μ^ℓ, σ*).

**Off the data [inference].** Every P_ℓ(y,·) is a convex weight vector. Along a ray with a unique extreme probe point x*, all levels lock onto x*, so f_L(y) → s₀(x*) + Σ_{ℓ≥1}d_ℓ(x*). That is a fixed combination of the level-0 smooth and the stored residual corrections at x*, not necessarily x*'s own activation. It is bounded, flat, and overconfident. Only level 0 is guaranteed to lie in the convex hull of the activations, since later levels add signed corrections.

**Interpolates?** In the limit (Leeb). Not with early stopping.

### 3.4 Auto-adaptive Laplacian pyramids (Fernández, Rabin, Fishelov & Dorronsoro: arXiv:1311.6594; ESANN 2016 pp. 59–64; Eng. Appl. AI 93:103682, 2020)

**Trick [paper].** Run the whole pyramid with the kernel diagonal zeroed (P̃). The training residual then approximates the LOO residual at no extra cost. Stop at L = argmin_ℓ err_ℓ, "just when the training error approximation to the LOOCV error starts growing". At test time use the ordinary P_ℓ on the stored residuals.
- The 2014 version normalises and then zeroes, so rows sum to less than 1.
- The 2020 version zeroes K's diagonal and then normalises, which makes each level an exact Nadaraya–Watson LOO.

**Vector-valued outputs (2020 §3.1) [paper].** One L is chosen by minimising Σ_p‖D̃_p‖² over all components. This is what you need for D outputs, and it is computable from the Gram alone.

**Claims [paper].** "Does not require parameterization" and "does not overfit".
- σ₀: "provided it is wide enough". The 2020 heuristic is σ₀ = 10·max distance, with maxits = log₂(σ₀/σ_min) and σ_min = min distance/5.
- The LOO is approximate: residuals at other points were fitted using x_p's own value. The evidence is that the stopping index matched exact LOOCV in four examples.

**Local ALP (2020) [paper].** Per-point stopping levels from errors averaged over ν neighbours. ν is chosen by 10-fold CV, so it is not tuning-free.

**Results [paper].**
- Extended a 3-D diffusion embedding of 10,800-dimensional weather inputs; k-means on extended vs recomputed embeddings agreed 97.5%.
- UCI missing-value imputation against k-NN only, with mixed results at a 30% test split.

**Implementation.** datafold's `LaplacianPyramidsInterpolator(auto_adaptive=True)` is a tested one; `invert/dm.py: dec_lp` follows the 2020 version.

**My reading [inference].** As σ → 0, P̃ tends to "average of the nearest *other* points" rather than I, so the zero-diagonal error cannot reach 0 and Leeb's convergence argument stops applying. That is why the stopping rule behaves like a conditional-mean selector. Zeroing each level's diagonal still does not make the multi-level error an exact LOO (earlier levels' residuals at other points used x_p's value), so confirm the chosen level on held-out images.

### 3.5 Double diffusion maps / latent harmonics (Evangelou, Dietrich, Chiavazzo, Lehmberg, Meila & Kevrekidis, J. Comput. Phys. 2023, 112072; arXiv:2204.12536)

**Method [paper].**
- Step 1: an ordinary diffusion map. Non-harmonic eigenvectors are chosen by eye and checked with Dsilva's test.
- Step 2: a *second*, unnormalised Gaussian kernel on the latent coordinates, K*(φ_i, φ_j) = exp(−‖φ_i − φ_j‖²/(2ε₂)), with its first d eigenpairs, and GH extension of each ambient coordinate: (Ef)(φ_new) = Σ_{j≤d}⟨f, ψ_j⟩σ_j⁻¹Σ_i K*(φ_new, φ_i)ψ_j(φ_i).
- Rationale: harmonics discarded in step 1 "are still important in extending a function", so a second diffusion map on Φ rebuilds that basis.
- Restriction: Nyström, differentiated analytically to push vector fields through the chain rule.

**Parameters [paper].** Reported, not derived. For example ε₂ = 4.46×10⁻⁵ with d = 300 (Chafee–Infante), and ε₂ = 4.11×10⁻⁵ with d = 400 (H₂ combustion). No rule for ε₂ or d appears in the text.

**Failure modes [paper].** Trajectories near the manifold boundary are "more prone to numerical errors". Stiff targets are "numerically challenging". Validity is claimed only "in the neighborhood of the training set".

**Test sizes [paper].** ≈ 2,800 points in ℝ¹⁰ and ≈ 3,800 in ℝ⁹, both intrinsic dimension 2.

**For you [inference].** A tiny ε₂ with 300–400 eigenvectors is close to exact Gaussian interpolation, which is wrong for a conditional mean (Fact 2). It decays to the zero vector off the data. It is GH (§3.1) applied to the coordinates; what is new is the argument for a separate lifting kernel.

**Related.** Koronaki et al. 2023 (arXiv:2301.11728) use the same lift with 38 eigenvectors, ambient dimension 58,100, intrinsic dimension 3, N = 720.

### 3.6 Kernel ridge / Gaussian-process (kriging) regression on the coordinates

**Formula (Rasmussen & Williams 2006, ch. 2 and 5; standard).**

  x̂(y) = m(y) + k(y)ᵀ(K + σ²I)⁻¹(X − M),  s²(y) = k(y,y) − k(y)ᵀ(K + σ²I)⁻¹k(y),

with an ARD kernel k(y,y′) = s_f² exp(−½Σ_k (y_k − y′_k)²/ℓ_k²) and an optional parametric trend m(y) = β₀ + Bᵀy, i.e. universal kriging (`dec_krr(linear=True)` is a version of this).

**Parameters.** The length scales ℓ_1..ℓ_K, the noise σ², and s_f.
- *Closed-form LOO (PRESS).* For fixed hyperparameters and the smoother H = K(K + σ²I)⁻¹, the LOO residual of probe i is r_i/(1 − H_ii). The total LOO error is Σ_i[(I−H)G(I−H)ᵀ]_ii/(1−H_ii)², which needs only the Gram G = XXᵀ. Each candidate costs O(N³) in general (the rotated Gram is dense); `dec_krr` does this directly. PRESS does not re-estimate the hyperparameters on each fold, so when many are tuned (ARD), confirm on held-out images.
- *Marginal likelihood.* −(D/2)log|K + σ²I| − ½tr[(K + σ²I)⁻¹XXᵀ] + const. It is invariant to orthogonal rotations of the outputs, so it also needs only G. You can replace X by its N×N PCA scores, provided the determinant term keeps the factor D (the discarded zero columns still count).
- *Caution [inference].* The likelihood treats D correlated output dimensions as independent with a shared kernel, which overweights the evidence. Prefer held-out or LOO error for selection, and use the likelihood's gradients only to fit ARD scales if ARD is used.
- *Caution about ARD [inference].* Diagonal ARD depends on the orientation chosen within nearly degenerate spectral subspaces, and standardising low-variance coordinates can amplify unstable modes. Automatic optimisation of 16+ hyperparameters is not the same as low tuning. Try it only after isotropic KRR, and keep it only if it wins on held-out images.

**Off the data.** k(y) → 0, so x̂ → the trend (linear extrapolation) or the mean (≈ 0 for centred data), and s² → s_f². The displayed s² is the known-mean variance: it omits the uncertainty of a fitted trend, and it is the GP's uncertainty about the mean, not the conditional spread of activations.
**Interpolates?** Only as σ² → 0.
**Round trip.** Not guaranteed.
**Cost.** O(N³) per hyperparameter evaluation (the Gram trick makes it independent of D); O(NK + ND) per query. At N = 50k, use sparse/inducing-point GPs or local GPs on kNN in ℝᴷ.
**Literature status.**
- Bakır, Weston & Schölkopf (NIPS 2004) is exactly this ("learning to find pre-images" by KRR from KPCA coordinates), but with hand-set σ and λ.
- Bermanis et al. 2013 note the equivalence of Nyström to the GP mean.
- Chiavazzo et al. 2014 found local kriging competitive (below) and global kriging bad.

### 3.7 What the comparison papers found

**Chiavazzo, Gear, Dsilva, Rabin & Kevrekidis, "Reduced models in chemical kinetics via nonlinear data-mining", Processes 2(1):112–140 (2014); read as arXiv:1307.6849.**
- Setup: H₂/air, 9 species, N = 3,810, a 2-D manifold, lifting from the 2-D reduced space. Restriction methods compared were Nyström, RBF and LP. Mean reduced-space error ‖δψ‖ (Table 1) [paper]:
  - Local RBF (p = 3, 50 neighbours) lift + Nyström restriction: **5.66×10⁻⁵**, the best.
  - Local kriging (8 neighbours, θ = 10⁻³): 7.31×10⁻⁵.
  - Local GH (15 neighbours): 1.27×10⁻⁴.
  - RBF lift + RBF restriction: 2.28×10⁻⁴.
  - LP lift (20 levels) + Nyström: 2.64×10⁻⁴.
  - LP used as the restriction: 8×10⁻⁴ to 2.4×10⁻².
  - **Global** kriging: 5.8×10⁻³.
- Conclusion [paper]: best is "a local lifting operator (… nearest neighbors) with the Nyström method for restriction". RBF is cheapest; kriging and GH "are certainly to be preferred" when samples are clustered or non-uniform; LP restriction is worse because of inaccurate derivatives.
- **Geometric point [paper].** An interpolating lift defines a surface through the samples that only approximates the true manifold. Projecting along the foliation defined by the restriction's Jacobian (their eq. 13), rather than orthogonally onto the lifted surface (eq. 14), cut the error about 10× (7.31×10⁻⁵ vs 8.81×10⁻⁴). These are reduced *trajectory* errors under different projection equations, not a direct test of decoder round trips. It is still the only quantitative evidence we found that the lift and the restriction actually used should be kept consistent, which is weak support for design 3.

**Papaioannou, Talmon, Kevrekidis & Siettos, Chaos 32:083113 (2022); arXiv:2110.03625.**
- They pose the pre-image as argmin_c ‖y* − R(L(y*|c))‖, a round-trip criterion, but do not report it [paper].
- Local GH lifting matched forecasts in the original space to about 3 digits. Local polyharmonic RBF (p = 1, 50 neighbours) "generally resulted in poor reconstructions", attributed to ill-conditioning [paper].
- Test sizes: 5-D synthetic and 10-D FOREX data, 2–3 coordinates.

**dos Santos, Giovanis, Kontolati, Loukrezis & Shields, arXiv:2109.13805 (IJNME 2022) [paper].**
- This is a lift *relative to a reference*: log-map the k nearest neighbours to the tangent space at the nearest one (or at their Karcher mean), fit local GH from coordinates to tangent vectors, then exponential-map back.
- Bandwidth = 0.25 × (median pairwise distance among the k)².
- Errors were "not strongly influenced by k" (k = 3, 5, 10).
- Kontolati et al. (IJUQ 2022) and Giovanis & Shields (CMAME 2020) use the same tangent-space template on Grassmannians with clustered references, and stress that it holds only locally. This is the template for keeping decoded vectors on your sphere.

**Dsilva et al., J. Chem. Phys. 139:184109 (2013).** Reportedly found LP better than nearest neighbours for lifting; we have this only second-hand.

**Koronaki et al., Chem. Eng. J. 2020.** Not accessed. We could not confirm that it compares GH with neural networks for lifting.

**What does not transfer [inference].**
- Every one of these cases has 2–4 coordinates that determine the state, so "best" there means "best interpolant". In your regime the ranking can flip toward smoothers with a data-driven stopping or ridge rule.
- No paper tests 16-dimensional latent kernels, hubness, or nearest neighbours at cosine 0.3.
- No paper reports restrict(lift(y)) ≈ y.

---

## 4. Pre-image methods and the shape-analysis literature (your item 4)

### 4.1 Kernel-PCA pre-images

**Mika, Schölkopf, Smola, Müller, Scholz & Rätsch, NIPS 11 (1999) [paper].**
- Maximise Σ_iγ_i k(z, x_i), with γ_i = Σ_kβ_kα_i^k. For a Gaussian kernel the fixed point is z ← Σγ_i e^{−‖z−x_i‖²/c}x_i / Σγ_i e^{−‖z−x_i‖²/c}.
- "Any fixed-point … will be a linear combination of the kernel PCA training data."
- Numerical instabilities when the denominator is small are "dealt with by restarting".
- Width c is set by hand.
- [inference] The γ_i can be negative, so the result is affine rather than convex. Far from the data the denominator → 0, so the method is undefined there; it does not extrapolate.

**Kwok & Tsang, IEEE TNN 15(6):1517–1525 (2004) [paper].**
1. Convert feature-space distances to input-space distances (for a Gaussian, d_i² = −(1/β)log(½(K_ii + K_jj − d̃_i²))).
2. Take the n = 10 nearest neighbours. Centre them and take the SVD XH = UΛVᵀ = UZ.
3. Solve ẑ = −½Λ⁻¹Vᵀ(d² − d₀²) and set x̂ = Uẑ + x̄.

It is "noniterative … does not suffer from numerical instabilities or the local minimum problem". It interpolates training points. [inference] The answer lies in the neighbours' affine span, so it can extrapolate within that flat. It becomes undefined for a Gaussian when d̃² ≥ 2.

**Bakır, Weston & Schölkopf, NIPS 16 (2004) [paper].**
- KRR from KPCA coordinates to inputs (§3.6). Parameters were set by hand, in one case "selected in a loop such that new synthesized frames looked subjectively best".
- It "requires that the patterns used during training reasonably well represent the points for which we subsequently want to compute pre-images … pre-images of linear combinations of mapped points … can be far away from training points, making generalization … impossible."

**Arias, Randall & Sapiro, "Connecting the out-of-sample and pre-image problems in kernel methods", CVPR 2007 [paper].**
- Setting: a plain Gaussian kernel, not diffusion maps.
- Write the pre-image objective through the Nyström extension, x = argmin‖Λ^{−1/2}Uᵀk_x − ψ‖². Solve first for the kernel vector, k̂_x = UΛ^{1/2}ψ/‖ψ‖, i.e. invert the Nyström extension. Then recover x by a one-step weighted mean x = Σα_ik̂_ix_i/Σα_ik̂_i (their method D) or by Kwok–Tsang.
- **Normalisation issue.** "If the searched pre-image x is outside the range of the extension, ‖φ̂(x)‖ tends to zero", so ψ is projected to the unit sphere. The older estimates wrongly assume ‖ψ‖ = 1 (an average of mapped points does not have unit norm).
- **Evaluation metric.** Forward/inverse consistency, measured as the collinearity 1 − cos(φ̂(x_pre), ψ).
- **Results.** "Almost no improvement by using the fixed point iterative pre-image". MNIST PSNR: method D 20.48 vs 17.17 for linear PCA.
- [inference] Normalising ψ makes radial extrapolation impossible by construction. That is the same structural fact as §0's range obstacle, seen from the kernel side.

**Honeine & Richard, IEEE Signal Processing Magazine 28(2):77–88 (2011) [paper].**
- Taxonomy: exact, gradient, fixed-point (with the regularised variant of Abrahamsen & Hansen 2009), learned map (Bakır; Zheng & Lai), MDS-based (Kwok–Tsang, localised "in the same spirit as LLE", citing Étyngier et al.), and their conformal map.
- "The resulting preimage lies in the span of the available data."
- Conformal map (from "Solving the pre-image problem in kernel machines: a direct method", MLSP 2009, not accessed): x* = (XXᵀ)⁻¹X(XᵀX − γK⁻¹)α, "independent of the kernel type".
- [inference] With D > N and γ = 0 this reduces to x* = Σα_ix_i: the feature-space coefficients reused in input space. It is linear in α, so it is defined everywhere and extrapolates linearly without bound.

### 4.2 Diffusion-map pre-images in shape analysis: the likely paper you remember

**Most likely candidates: the Paris-Est (ENPC/CERTIS) group, Étyngier, Ségonne, Keriven, then Thorstensen.** Strongest match: **Thorstensen, Étyngier, Ségonne & Keriven, "Diffusion maps as a framework for shape modeling", CVIU 115(4):520–530 (2011)**, which collects the earlier conference papers and studies the pre-image problem explicitly.

**Étyngier, Ségonne & Keriven, "Shape priors using manifold learning techniques", ICCV 2007.** Not accessed; we read the companion ICIP 2007, SSVM 2007 and CERTIS report 07-33 [paper]:
- *Forward map.* Minimise the Laplacian energy with the training embedding fixed. This gives the Nadaraya–Watson map ẑ(x) = Σk(x,x_i)y_i/Σk(x,x_i), with "ẑ(x_i) ≠ y*(x_i)", so they replace the training embedding with ŷ = ẑ(x_i).
- *Inverse.* Locate the Delaunay simplex of the ŷ points containing the target, take its barycentric coordinates λ, and compute S̄ = argmin_S Σλ_i d(S_i, S)², a weighted Karcher mean. The CERTIS report adds the forward-consistency constraint Φ̃(S) = target, solved by gradient descent.
- *Off the data.* They assume the target is inside the convex hull, "were the point outside, we would consider instead its orthogonal projection on the convex-hull … projection of a point located outside the manifold will belong to the border of the manifold."

**Thorstensen, Ségonne & Keriven, "Pre-image as Karcher mean using diffusion maps", SSVM 2009, LNCS 5567:721–732 [paper].**
- Diffusion maps with α = 1 and Nyström extension. The pre-image is argmin over s_Θ of ‖Ψ̂(s_Θ) − Υ‖², where s_Θ = argmin_z Σθ_i d(z, s_i)² is a Karcher mean over the k+1 nearest neighbours of Υ in diffusion space, with θ in the simplex.
- The θ are optimised by projected gradient descent until the *Nyström image matches the target*. They are not read off the Nyström weights.
- CVIU 2011 states the hull assumption explicitly and that L2 distance "amounts to assuming that the manifold M is piecewise-linear". The embedding dimension "cannot be computed automatically and therefore must be guessed".
- USPS denoising beat Dambreville, Kwok–Tsang, Arias, and Carreira-Perpiñán–Lu.

**Relevance [inference].** High in concept, limited in form. This is a forward-consistent pre-image: choose a point on a family of candidate pre-images (Karcher means of neighbours) whose *forward* image matches the target. That is design 3 with the candidate family restricted to the geodesic hull of k+1 neighbours. Three limitations for you:
1. It cannot extrapolate.
2. With K = 16 it needs 17-point simplices.
3. On your sphere a weighted mean of unit vectors falls inside the ball: its squared norm is wᵀGw. For example, 15 equally weighted neighbours with mean *mutual* cosine 0.3 give norm √(1/15 + (14/15)·0.3) ≈ 0.59. Renormalising is a choice suited to a direction target.

Design 3 keeps the forward-matching idea and drops the hull restriction.

**Two other strong matches, not on your list.**
- **Carreira-Perpiñán & Lu, "The Laplacian eigenmaps latent variable model" (LELVM), AISTATS 2007, PMLR 2:59–66 [paper].**
  - Both maps are Nadaraya–Watson. The decoder F(y) = ΣK(y,y_n)x_n/ΣK is "formally identical to a Nadaraya–Watson estimator", and the encoder is E{y|x}.
  - "x_n ≠ F(y_n) and y_n ≠ f(x_n)" (no round trip), defended as smoothing.
  - "The ranges of f and F are the convex hulls", with "shortening of the manifold at the boundaries".
  - Bandwidth: "cross-validation rules tend to give small bandwidths … jagged mappings".
  - This paper states the conditional-mean framing outright.
- **Gerber, Tasdizen & Whitaker, "Dimensionality reduction and principal surfaces via kernel map manifolds", ICCV 2009 [paper].**
  - Nadaraya–Watson in both directions, with the forward parameters optimised to minimise Σ‖g(f(x_i)) − x_i‖².
  - Bandwidths from the average kNN distance; early stopping on held-out projection error.
  - If the bandwidth → ∞ the result collapses to the data mean.
  - This is the cleanest "jointly fit forward and inverse" recipe found.

**Not the one.**
- **Dambreville, Rathi & Tannenbaum (TPAMI 30(8), 2008)** *avoid* the pre-image by working in feature space. Their pre-image is in Rathi, Dambreville & Tannenbaum (SPIE 6064, 2006; not accessed), described by Arias as Mika's formula with kernel values replaced by ½(2 − d̃²).
- **Kilian, Mitra & Pottmann (TOG 2007)** do geodesic interpolation and extrapolation in a mesh shape space; only conceptually related (abstract only).

---

## 5. Local, parametric and neural decoders (your item 6)

### 5.1 LLE barycentric inverse (Saul & Roweis, "Think globally, fit locally", JMLR 4:119–155, 2003, §6.1) [paper]

- **Weights.** For y, take its k nearest probe points in coordinate space. Form G_jk = (y − y_j)·(y − y_k), regularise G ← G + (Δ²/k)Tr(G)I (Δ = 0.1 in their figures), and set w = G⁻¹1/(1ᵀG⁻¹1).
- **Decode.** x = Σw_jx_j. In the paper's words: "to compute the input x for a new output y, identify the nearest neighbors of y among the training outputs, compute the analogous reconstruction weights, and output x = Σ w_j X_j."
- **Drawback (as stated).** "The non-parametric mapping can change discontinuously as query points move between different neighborhoods".
- **§6.2 alternative.** A mixture of conditional linear models, P(x | y, z) = N(Λ_zy + μ_z, Ψ_z), with E[x|y] and E[y|x] in closed form. This is a parametric conditional mean; the number of components is fitted by EM.
- **Behaviour [inference].**
  - The weights are affine and can be negative, so near the data the map can extrapolate linearly outside the neighbours' hull. But the regulariser scales with Tr(G), which grows with distance. With a fixed neighbourhood and a nonzero Δ, far away it drives the weights towards uniform, i.e. towards the neighbour mean, not linear extrapolation. A plain affine decoder on span{1, coordinates} (`dec_linear`) is the cleaner linear-extrapolation control.
  - It approximately interpolates.
  - Neighbour search happens in ℝᴷ, so ambient concentration does not affect it.
  - Cost: O(k³ + kD) per query.
- **LTSA (Zhang & Zha, SIAM J. Sci. Comput. 26(1), 2004, §5) [paper].** Decoder g(τ) = x̄_i + Q_iL_i⁻¹(τ − τ̄_i), with error ≤ noise + tangent residual + alignment error (Thm 5.1). [inference] With intrinsic dimension 40–77 and K = 16, the local tangent space is underdetermined.

### 5.2 Neural decoders

**Diffusion nets (Mishne, Shaham, Cloninger & Cohen, ACHA 47(2):259–285, 2019; arXiv:1506.07840) [paper].**
- Encoder loss: ‖o_e(x) − Ψ(x)‖² + weight decay + η‖(P − λ_jI)o_{e,j}ᵀ‖² (the eigenvector constraint). P may be any normalised affinity, so M_l would do.
- The decoder is an unconstrained MLP.
- η is set by hand (10–100 worked best; a large η "collapses the output of the encoder to the origin").
- Off the data: "points in the diffusion space which are located beyond the original circle, are decoded in the data space along the boundary of the surface. Thus, the range of extension is limited." Their error bounds "do[] not apply to the decoder".

**GRAE (Duque, Morin, Wolf & Moon; IEEE Big Data 2020 / TPAMI 2022; arXiv:2007.07142) [paper].**
- Autoencoder loss with λΣ‖e_i − f(x_i)‖² towards a precomputed embedding ("Isomap, t-SNE, etc." also work).
- λ follows a sigmoid schedule "to avoid the need to tune λ".
- Better reconstructions of held-out interpolations than a plain AE. No analysis of decoding far off the data and no round-trip guarantee.

**Conformal autoencoders (G. A. Kevrekidis, Ahmad, Maggioni, Villar & Y. G. Kevrekidis, arXiv:2408.16138).** About detecting latent dimension by orthogonality of gradients. Not a decoder for this purpose; the brief's attribution to Sroczynski/Bello-Rivas appears to be wrong.

**Verdict [inference].** More tuning, no off-data theory, and no round trip. Worth considering only at N ≳ 50k, where an amortised decoder beats O(N³) kernels; GRAE with your coordinates as targets is the natural form.

### 5.3 Spectral Gaussian-process priors

- **Borovitskiy et al.**, "Matérn Gaussian processes on graphs" (AISTATS 2021) and "… on Riemannian manifolds" (NeurIPS 2020) [paper]: covariance UΦ(Λ)Uᵀ with Φ(λ) = (2ν/κ² + λ)^{−ν} or e^{−κ²λ/4}.
- **Dunson, Wu & Wu**, "Graph based Gaussian processes on restricted domains" (JRSS-B 2022) [paper]: Σ_{i≤K}e^{−μ_it}v_iv_iᵀ from graph-Laplacian eigenpairs, with a Nyström extension to new samples.
- On manifolds the Matérn exponent is −(ν + d/2), not −ν.
- [inference] These are priors over functions on the (site, image) nodes, not decoders from coordinates. Your L_t eigenpairs give such a prior directly. GP regression with it is a principled way to predict one network's quantities on new images from another's (§7).

---

## 6. The "residual" idea (your item 7)

### 6.1 Analogues

**Equation-free lift-then-relax.**
- **Kevrekidis, Gear, Hyman, Kevrekidis, Runborg & Theodoropoulos, Commun. Math. Sci. 1(4):715–762 (2003) [paper].**
  - Restriction ℳ, lifting μ, with the consistency requirement ℳμ = I. The coarse timestepper is ℳ∘T_fine∘μ.
  - Lifting is one-to-many. Unresolved moments are set arbitrarily and "quickly relax to functionals of the lower moments" ("healing").
  - If results are sensitive to the lifting, the remedy is to promote more moments to coarse variables.
- **Gear, Kaper, Kevrekidis & Zagaris, SIADS 4(3):711–732 (2005) [paper].**
  - "Constrained runs": hold the coarse u fixed and find the fast v with d^{m+1}v/dt^{m+1} = 0.
  - Iteration: v₀ ← v₀ + (−1)^mΔ^{m+1}v₀.
  - The result is m-th order close to the slow manifold, *provided u parametrises it as a graph v = v(u)*.
  - Zagaris et al. (M2AN 2009) give O(ε^{m+1}) error and stability bounds.
- **Vandekerckhove, Sonday, Makeev, Roose & Kevrekidis, Comput. Chem. Eng. 35:1949–1958 (2011) [paper].** "InitMan": solve restrict(relax_τ(lift(x))) = x₀ by Newton. It fails when the coarse observable relaxes fast (condition ≈ 10⁸).
- [inference] You have no "fine dynamics" that relaxes an activation at a fixed site, and v = v(u) fails (Fact 1). So relaxation cannot select a unique completion; the completion must come from a reference or a prior. The network's own forward pass from site l onwards *is* a fine model, but only downstream. Use it as a validation check: decode at l, run the network to l+1, embed, and compare with the joint operator's prediction. Do not use it as a healing step.

**Matching operators: the principled "carry the remainder from a reference" [paper].**
- **Debrabant & Samaey** (arXiv:1009.3767): Y = argmin_{R(Z)=U}½‖Z − Y_ref‖², solved as Y = Y_ref + Σ_lλ_l∇R_l with Newton on the multipliers. If the reference already satisfies the constraint, it is returned unchanged.
- **Debrabant, Samaey & Zieliński** (SINUM 2017; arXiv:1511.06171): L2 matching takes the constrained components from the target and **copies everything orthogonal from the prior**. It is non-expansive in the prior and Lipschitz in the target.
- **Lelièvre, Samaey & Zieliński** (arXiv:1801.01740): minimum-relative-entropy matching ν ∝ exp(Σλ_lR_l)μ, which on samples is reweighting. It satisfies a Pythagorean identity, so the information gain from adding a coarse observable is computable.

**SAE error nodes.**
- **Marks et al., "Sparse feature circuits" (ICLR 2025; arXiv:2403.19647) [paper].** x = Σf_i(x)v_i + b + ε(x). Interventions edit features while holding ε fixed, x_d = x̂ + stopgrad(x − x̂). That is exactly x′ = g(y′) + (x − g(y(x))).
- **Ameisen et al., "Circuit tracing" (Transformer Circuits Thread, 2025) [paper].** The local replacement model adds "the difference between the true MLP output on p and the CLT output on p"; "error nodes … have no input edges", i.e. they are frozen exogenous inputs. We read about the first 100k characters; we did not find a statement of how error nodes behave under large perturbations.

**HAVOK (Brunton, Brunton, Proctor, Kaiser & Kutz, Nat. Commun. 8:19, 2017) [paper].** d/dt v = Av + Bv_r: the last delay coordinate is not modelled but fed in as observed forcing. The rank r is set by the Gavish–Donoho optimal hard threshold, a tuning-free rule you could try for choosing K.

**Triangular transport, the principled statistical version [paper + inference].**
- **Sources [paper].** Marzouk, Moselhy, Parno & Spantini ("An introduction to sampling via measure transport", 2016; arXiv:1602.05023), Baptista, Marzouk & Zahm (FoCM 24, 2024; arXiv:2009.10303), and Baptista, Hosseini, Kovachki & Marzouk (SIAM/ASA JUQ 2024; arXiv:2006.06755). A block-triangular map S(y, x) = [S^Y(y); S^X(y, x)] pushes π(x | y) to a reference η, so π(x|y) = S^X(y,·)⁻¹♯η.
- **Abduct-and-decode [inference].** Abduct the reference's latent z_ref = S^X(y_ref, x_ref), then decode x_new = S^X(y_new,·)⁻¹(z_ref). This is the counterfactual / conditional-quantile transfer.
- **Gaussian case.** If π(x|y) = N(g(y), L(y)L(y)ᵀ), this is x_new = g(y_new) + L(y_new)L(y_ref)⁻¹(x_ref − g(y_ref)).
- **Homoscedastic case.** It reduces exactly to your rule, x_new = g(y_new) + (x_ref − g(y_ref)), with g = E[x|y], e.g. design 1's mean.
- **When it would be principled.** When the residual's distribution does not depend on y. On the unit sphere it cannot hold exactly: Fact 1 implies that independent additive residuals would force ‖g(y)‖ to be constant, and no nonsingular Gaussian lives on the sphere. Regressing ‖x − g(y)‖² and the leading residual PCs on y can *detect* how badly it fails, not establish that it holds.
- **What it identifies.** A fitted transport chooses one coupling between conditional laws. It does not identify an image-level counterfactual from observations alone. So residual transport is a well-motivated *heuristic*, not a theorem, in this setting.
- **On the sphere.** Use x_new = Exp_{g̃(y_new)}(Γ_{g̃(y_ref)→g̃(y_new)} Log_{g̃(y_ref)} x_ref), with g̃ = g/‖g‖ and Γ parallel transport, which is closed-form on the sphere. This is the dos Santos et al. tangent-space template (§3.7). It is reasonable away from near-zero means and antipodal ambiguities, but it preserves neither the target coordinates nor the conditional law. Design 3 repairs the coordinates.

### 6.2 Design 3: forward-consistent matching pre-image [inference: assembled from the above; revised after review]

  x* = argmin_{‖x‖=1} ‖x − x₀‖²  subject to  F_l(x) = y,

where:
- F_l is the smooth forward map at site l (§2.2–2.3): F_smooth, not the masked map, with the other sites' rows b_{l′} held at a reference image's values (the preferred single-site variant of §2.3).
- x₀ is the prior guess: design 1's g(y), renormalised; a retrieved real probe activation; or the residual-carrying x₀ = g(y) + (x_ref − g(y_ref)) of §6.1.

**This optimises consistency, not accuracy.** It returns an activation with exactly the requested coordinates that is closest to x₀. That can be atypical of real activations. Use it when an exact round trip is a requirement, and compare it with design 1 and with retrieval.

**Jacobian.** For the Gaussian-on-cosine kernel, p(x,j) = softmax_j(x·x_j/ε′ + log v_j), where ε′ = ε/2 for the chord-distance Gaussian exp(−‖x−y‖²/ε) of §2.1 and ε′ = ε for `dm.py`'s exp(−(1−cos)/ε). So

  ∂b/∂x = (1/ε′)Σ_j p(x,j) f_l(j)(x_j − x̄_p)ᵀ,  x̄_p = Σ_j p(x,j)x_j,

the K×D "covariance of coordinates with activations under the kernel row". This is Marshall–Coifman's gradient formula. With the other sites' rows held fixed, mode k's coordinate at site l is F_{l,k} = const + [(A − μ_kI)⁻¹]_{ll} b_{l,k}/(2c_l). So row k of J is row k of ∂b/∂x times [(A − μ_kI)⁻¹]_{ll}/(2c_l) (away from resonance).

**Iteration (tangent-space SQP).** Do not normalise x₀ + Jᵀλ; that is not a valid step on the sphere. At the current unit x, with d = x₀ − x:
1. Solve min_δ ‖δ − d‖² subject to xᵀδ = 0 and Jδ = y − F_l(x). With B = [J; xᵀ] and r = [y − F_l(x); 0], δ = d + Bᵀ(BBᵀ)⁻¹(r − Bd).
2. Retract: x ← (x + δ)/‖x + δ‖.

Use a damped or rank-revealing solve for (BBᵀ)⁻¹ with a line search, and check the actual constraint residual at every step. Local full row rank of J is needed; feasibility does not guarantee convergence from a given x₀. Each step costs O(NDK) to form J, plus O(K³). Only each step's *local* correction is confined to directions the forward map can see; the iterates as a whole need not preserve x₀'s other components exactly.

**A useful reduction.** F_l depends on x only through the cosines Xᵀx, i.e. through x's projection onto the span of the probe activations (dimension ≤ N − 1 < D). Write x = Ua + x_⊥, with U an orthonormal basis of that span. The constraint involves only a. Given a, the best x_⊥ is √(1 − ‖a‖²) times the unit vector along x₀'s orthogonal component. So the problem reduces to N dimensions: minimise ‖a − a₀‖² + (√(1 − ‖a‖²) − ‖x₀⊥‖)² subject to F_l(Ua) = y and ‖a‖ ≤ 1. For an out-of-distribution reference image, x₀⊥ ≠ 0 is exactly the part of it that the probe cannot see, and it is carried over unchanged. For probe-derived x₀ it is zero, and the direction of x_⊥ is then arbitrary.

**Properties.**
- An exact round trip, under F_smooth, whenever it converges.
- No parameters beyond the forward kernel's ε (plus solver safeguards).

**Feasibility.** Outside the range of F_l (§0) there is no solution. Even coordinates of real activations under the full forward map need not be feasible for this clamped single-site map. A finite penalty ‖F_l(x) − y‖²/τ² + ‖x − x₀‖² does not in general land on the nearest attainable coordinates. If "nearest attainable coordinates" is the goal, first minimise the forward residual, then use distance to x₀ to choose among the minimisers. Étyngier, Carreira-Perpiñán–Lu and diffusion nets all report boundary saturation for their own maps.

**Requirements.**
- The 0/1 kNN kernel has a piecewise-constant forward map with zero gradient almost everywhere, so design 3 needs the smooth kernel.
- To feed the result to the next ResNet block, the site mean and scale (and any other state the network needs at that site) must be restored.

**Literature support.** Weak but convergent: Chiavazzo's gain from restriction-consistent projection, the Thorstensen/Étyngier Nyström-matching constraint, Arias' collinearity criterion, and the matching operators above. Debrabant–Samaey's results concern linear moment constraints in flat space; the sphere-constrained update here is ours.

---

## 7. Transfer between networks (functional maps)

**Ovsjanikov, Ben-Chen, Solomon, Butscher & Guibas, "Functional maps", ACM TOG 31(4):30 (2012) [paper].**
- C solves C·A ≈ B (descriptors) plus Laplacian-commutativity terms.
- C is orthonormal for area-preserving maps, and isometries commute with the Laplace–Beltrami operator.
- Point-to-point recovery is nearest neighbour of CΦ_M's columns in Φ_N, refined by ICP.

**Fumero, Pegoraro, Maiorca, Locatello & Rodolà, "Latent functional maps", NeurIPS 2024; arXiv:2406.14183 [paper].**
- kNN graphs on network latents (cosine is more robust).
- With a known correspondence P, C = Φ_N^†PΦ_M.
- Similarity score 1 − ‖off(CᵀC)‖²_F/‖CᵀC‖²_F.
- Limitations: sensitive to the number of eigenvectors, and it "assumes a correspondence between the domains".

**Moschella et al., "Relative representations", ICLR 2023 [paper].** Represent each sample by its cosines to anchor samples, which is invariant to angle-preserving transforms.

**Coifman & Hirn, "Diffusion maps for changing data" (ACHA 36(1), 2014) [paper].** A diffusion distance between points embedded under two different kernels on the same set involves the cross-Gram ⟨ψ_α^i, ψ_β^j⟩.

**For you [inference].**
- **C is a matrix of cosines.** Your probe images are shared, so P = I. For single-site walks the Sinkhorn eigenvectors are orthonormal in the plain inner product, so C = Φ_BᵀΦ_A, a K×K matrix of eigenvector cosines (the Coifman–Hirn cross-Gram). Its singular values are the cosines of the principal angles between the two spectral subspaces: CCA on spectral embeddings.
- **Per site.** Per-site blocks of joint modes are not orthonormal, so keep two objects apart:
  - *Comparing subspaces.* Take orthonormal bases Q_{A,l}, Q_{B,l} of the column spaces of Φ_{A,l}, Φ_{B,l}. The principal-angle cosines are the singular values of Q_{B,l}ᵀQ_{A,l}.
  - *Mapping coordinates.* To predict an image's B coordinate row from its A row by least squares (Φ_{A,l}W ≈ Φ_{B,l}), use y_B = y_A W_l with W_l = Φ_{A,l}⁺Φ_{B,l}. The functional-map matrix Φ_{B,l}⁺Φ_{A,l} instead maps *function coefficients* from A's basis to B's. The two differ when the blocks are not orthonormal or are rank-deficient.
- **What to keep and drop.** Keep the commutativity term ‖CΛ_A − Λ_BC‖² as a ridge that stops low and high modes mixing. Descriptors and ICP exist for the unknown-correspondence case and are unnecessary.
- **Transferring a new image.** Either take A's forward coordinates then W_l (linear, defined everywhere), or pass the image through B, which you can always do. To decode into B from A's coordinates, train design 1 with B's activations as targets on A's coordinates.
- **Relative representations already.** Your kernel is a function of the new image's cosine row to the probe, so the forward map is anchor-based and works identically for both networks.

---

## 8. Comparison table (decoders)

Here "hull" means the convex hull of the probe activations or coordinates.

| Method | Formula (decode y) | Tuning and automatic rule | Cost (train / query) | Off the data | Interpolates probes? | Round trip with Nyström forward? |
|---|---|---|---|---|---|---|
| Affine (`dec_linear`) | least squares on span{1, y} | None | O(NK²) / O(KD) | Linear extrapolation, unbounded | No | No |
| Kernel ridge / GP, isotropic or ARD (+ linear trend) | k(y)ᵀ(K+σ²I)⁻¹X (+ trend) | bandwidth (or K length scales), σ²; PRESS or held-out error; marginal likelihood for ARD | O(N³) per candidate via Gram / O(N(K+D)) | → trend or mean; variance → prior | Only as σ² → 0 | No (measure it) |
| Geometric harmonics / latent harmonics | k(y)ᵀΨ_SΛ_S⁻¹Ψ_Sᵀ X | ε, δ (or d); approximate LOO or held-out; no published rule | O(N³) / O(N(|S|+D)) | Decays to 0 | Only as δ → 0 | No |
| Multiscale GH with ID (Bermanis) | Σ_s Σ_k g_{ε_s}(y, y_{s_k})c_k^{(s)} | δ = 0.1, err; T from the query | O(l²N log N) per scale | Each level → 0; coarsest level reaches every query | Yes if err = 0 | No |
| Laplacian pyramid / ALP | Σ_ℓ P_ℓ(y,·)d_ℓ | σ₀ (rule exists); vector LOO stopping | O(LN²) / O(LN + ND) | Flat: a fixed combination of values at the extreme probe point; only level 0 stays in the hull | Converges to yes; ALP stops early | No |
| Local RBF / local kriging (Chiavazzo) | Interpolant on k nearest | k, p or θ | O(k³ + kD) per query | RBF: polynomial growth; kriging: trend | Yes | No |
| Nadaraya–Watson (LELVM, Étyngier forward) | ΣK(y,y_n)x_n/ΣK | Bandwidth (CV gives jagged maps) | O(ND) | Nearest-cluster value; hull | No | No |
| LLE barycentric (Saul–Roweis) | Σw_j(y)x_j, affine weights | k, Δ | O(k³ + kD) | Linear just outside the neighbours; towards the neighbour mean far away (trace-scaled regulariser); discontinuous | Approximately | No |
| Kwok–Tsang | Localisation in the neighbour flat | n, kernel width | O(Dn²) | Within the flat; undefined for d̃² ≥ 2 | Yes | No |
| Arias–Randall–Sapiro D | Invert Nyström for k̂_x, then a weighted mean | Kernel width | O(NK + ND) | Radial extrapolation impossible (ψ normalised) | No (smoothed) | Measured, not enforced |
| Thorstensen/Étyngier Karcher pre-image | Weighted Karcher mean of k+1 neighbours, weights chosen so Nyström(x) = y | k; embedding dimension | Iterative, O(T(ND + kD)) | Projects to the hull boundary | Yes | **Yes inside the hull** |
| Matching pre-image (design 3) | argmin‖x − x₀‖ s.t. F_l(x) = y | None beyond the forward kernel's ε | O(NDK) per SQP step | Infeasible outside the range | Under F_masked vs F_smooth: see §2.2 | **Yes when it converges** |
| Diffusion nets / GRAE | MLP decoder | Architecture, η or λ schedule | SGD | Saturates at the boundary (diffusion nets) | No | No |

---

## 9. Ranked designs to test, with implementation notes

All four share the same foundation:
- The doubly stochastic Gaussian-on-cosine kernel (zero diagonal, ε by perplexity), with the half-step out-of-sample row and the L×L chain solve (§2). Before treating it as a drop-in, compare the smooth-kernel embedding with the current graph embedding.
- Use the spatial modes only (Fact 3; the code already does this).
- Split and tune by **image**, not by (site, image) row. For a truly inductive test, keep the test images out of the centring, the kernel, the eigensolve and decoder selection, and bring them in only through the out-of-sample map.
- Report separately on held-out images:
  - the cosine of the decoded vector with the true activation;
  - the captured share 2⟨x̂,x⟩ − ‖x̂‖²;
  - the round-trip error ‖F_l(x̂/‖x̂‖) − y‖, stating which forward map (§2.2) is used.

1. **Conditional-mean regression ladder (KRR/GP).**
   - Steps, in order: mean → affine (`dec_linear`) → isotropic KRR with a linear trend, bandwidth and ridge by PRESS (`dec_krr`) → ARD (length scales from marginal-likelihood gradients on the PCA scores of X, ridge by PRESS) only if it beats isotropic KRR on held-out images.
   - For raw activations, regress the raw centred activations directly (Fact 1).
   - Why first: it estimates the identifiable quantity (Fact 1), every parameter has a data-driven rule, it reverts gracefully off the data, and geometric harmonics is a special case.
   - Unknown: whether any stationary kernel in ℝ¹⁶ beats the affine decoder by much. No paper tested this regime.
2. **ALP with the vector LOO stopping rule.**
   - The 2020 version: zero the diagonal, then normalise. σ₀ = cloud diameter, with Leeb's floor σ_ℓ ≥ σ*.
   - Why: nearly tuning-free and no inversions; an independent check on design 1.
   - Weaknesses: the LOO is approximate, so confirm the level on held-out images; extrapolation is flat; the kernel is isotropic (Fernández et al. give no ARD variant).
3. **Matching pre-image around a reference (§6.2)**, if exact consistency is required.
   - x₀ = renormalised design-1 mean (the "typical activation with these coordinates"), a retrieved real probe activation (a plausible single output), or residual transport from a chosen reference image (the "this image, moved to these coordinates" counterfactual).
   - Why: the only exact round trip, and the closest principled form of the residual idea. Use the span reduction of §6.2.
   - Diagnose first how far the residual departs from independence of y (it cannot hold exactly on the sphere, §6.1).
   - Unknown: convergence with K = 16 constraints, and how much of the change lands in directions irrelevant to the network downstream. Check by running the network from site l and re-embedding at l+1.
4. **The affine decoder as the extrapolation control.** It is the clean way to show what linear extrapolation in coordinates produces. Track the kernel-support diagnostic 1/v_x (§2.2) of the decoded vectors to see how fast they leave the probe's support. Not recommended as a primary decoder; the LLE barycentric decoder adds discontinuities without adding extrapolation.

**Not recommended now.** Neural decoders (until N ≳ 50k), Berry–Harlim α-normalisations at d ≈ 50, exact interpolators (untruncated GH, double-diffusion-map lifting with tiny ε₂, local RBF), and the 0/1 kNN kernel, whose forward map is piecewise constant.

**Open choices that change the ranking more than the literature does** (from the review):
- Must the forward map preserve the existing graph coordinates, or may it replace them?
- Will queries supply all layers, one normalised activation, or one raw activation plus the rest of the network state?
- Is success defined by squared error, cosine, a realistic activation, a downstream intervention, or an exact round trip under one specified forward map?

---

## 10. Where the literature gives no answer, and claims we are unsure of

**No answer in the literature.**
1. Decoding when the coordinates do not determine the state. Every lifting comparison (Chiavazzo, Evangelou, Papaioannou, Koronaki, dos Santos) has 2–4 coordinates parametrising the manifold. Nothing published says which lifting is best when K is much smaller than the intrinsic dimension. Our argument for smoothers with LOO rules is inference.
2. Out-of-sample extension of a supra-Laplacian / inflated dynamic Laplacian. The L×L chain solve in §2.3 is derived here. It is a direct consequence of the eigen-equation, and three independent derivations agree, but no paper states it.
3. Bandwidth rules for kernels on 16-dimensional spectral coordinates. Published lifts use 2–3 coordinates and hand-set ε.
4. Round-trip consistency. Only Arias et al. measure it, only Étyngier/Thorstensen enforce it, and only Chiavazzo gives (indirect) evidence that it matters.
5. Meaningful extrapolation in coordinate space. Every method's off-data behaviour is a convention (→ 0, → mean, → trend, flat, linear, or saturating). For a row-stochastic forward map, the targets outside its range have no pre-image at all, and the range itself is only bounded from outside.
6. Hubness *inside* the coordinate space, and how Sinkhorn scaling interacts with the eigenvectors' scale per site.

**Claims flagged as uncertain.**
- Rabin & Coifman 2012 was not accessed; the LP description comes from four papers that restate it consistently.
- Étyngier et al. ICCV 2007 was not accessed; its content is inferred from the ICIP 2007, SSVM 2007 and CERTIS 07-33 companion versions by the same authors.
- The volume number of the double-diffusion-maps J. Comput. Phys. paper (article 112072) is not confirmed (485 from memory).
- Koronaki et al. 2020 (Chem. Eng. J.) was not accessed; we cannot confirm that it compares GH with neural nets.
- Honeine & Richard's MLSP 2009 conformal-map paper was not accessed; the formula is from their 2011 survey.
- Zass & Shashua 2006, Dinu et al. 2015 and Smith et al. 2017: abstracts only.
- Radovanović et al. 2010 and Feldbauer & Flexer 2019 were read partially.
- We found no paper titled "Accuracy analysis of constrained runs". The nearest are Zagaris et al. 2009 and Vandekerckhove et al. 2011.
- The soft-CSLS reading of Sinkhorn on the sphere, the kernel-support diagnostic, the range obstacle, the norm identity in Fact 1, the temporal-mode observation, the single-site forward variants, and the Jacobian, SQP step and span reduction in §6.2 are our derivations. They are elementary but not taken from a paper; an independent GPT-6.2 review corrected several of them (the self-mask condition, the SQP step, the per-site transfer formula, the SNEkhorn ε estimate, and claims about LOO exactness and decoded norms).
- Exhaustive negative claims ("no paper does X") cover only the papers we reviewed.
- Pan et al. (ACM MM 2025, arXiv:2508.02538) use Sinkhorn explicitly for hubness reduction, with training scalings applied to new queries; read only via a web summary.

### Relation to the code already in `invert/`

`invert/dm.py` already implements:
- the doubly stochastic Gaussian-on-cosine kernel with perplexity calibration (§2.1);
- the half-step out-of-sample row (§2.2);
- the chain solve for joint modes (§2.3);
- LOO-selected KRR with a linear trend (§3.6, isotropic);
- the 2020-version ALP (§3.4);
- LOO-selected geometric harmonics (§3.1).

The literature supports each choice. The additions and corrections suggested by this survey are:
- the explicit self-mask distinction between F_masked and F_smooth (§2.2);
- checking each mode's distance from resonance in the chain solve, not only the condition number (§2.3);
- `dec_gh`'s projection LOO is approximate, so confirm with held-out images (§3.1);
- ARD only if it beats isotropic KRR on held-out images (§3.6);
- Leeb's bandwidth floor for ALP (§3.3);
- the kernel-support diagnostic 1/v_x (§2.2);
- the matching pre-image with the closed-form Jacobian, SQP step and span reduction (§6.2);
- residual transport as a heuristic, with diagnostics of residual dependence on y (§6.1);
- for transfer, W_l = Φ_{A,l}⁺Φ_{B,l} for coordinates and Q_BᵀQ_A for principal angles (§7);
- a frozen-radius extension for the kNN rule if it is kept (§2.1).

---

## 11. References (with URLs; access status)

**Read in full or in the relevant sections.**
- R. R. Coifman, S. Lafon. Geometric harmonics: a novel tool for multiscale out-of-sample extension of empirical functions. *Appl. Comput. Harmon. Anal.* 21:31–52, 2006. doi:10.1016/j.acha.2005.07.005. https://www.math.ucdavis.edu/~saito/data/diffgeomharm/ACHA-Special/geomharm.pdf
- S. Lafon, Y. Keller, R. R. Coifman. Data fusion and multicue data matching by diffusion maps. *IEEE TPAMI* 28(11):1784–1797, 2006. doi:10.1109/TPAMI.2006.223. https://www.ceremade.dauphine.fr/~cohen/mia2006/papers/SLafondata_fusion_by_diffusion_maps-lafon-keller-coifman.pdf
- A. Bermanis, A. Averbuch, R. R. Coifman. Multiscale data sampling and function extension. *ACHA* 34:15–29, 2013. https://www.cs.tau.ac.il/~amir1/PS/MSE1.pdf ; corrections https://www.cs.tau.ac.il/~amir1/PS/Errors_in_the_paper.pdf
- A. Bermanis, G. Wolf, A. Averbuch. Cover-based bounds on the numerical rank of Gaussian kernels. *ACHA* 36:302–315, 2014. https://www.cs.tau.ac.il/~amir1/PS/Cover.pdf
- A. Fernández, N. Rabin, D. Fishelov, J. R. Dorronsoro. Auto-adaptive Laplacian pyramids for high-dimensional data analysis. arXiv:1311.6594. https://arxiv.org/abs/1311.6594 ; ESANN 2016, pp. 59–64 (not read) https://www.esann.org/sites/default/files/proceedings/legacy/es2016-131.pdf ; Auto-adaptive multi-scale Laplacian pyramids for modeling non-uniform data. *Eng. Appl. AI* 93:103682, 2020. https://repositorio.uam.es/handle/10486/702591
- W. Leeb. Properties of Laplacian pyramids for extension and denoising. arXiv:1909.07974. https://arxiv.org/abs/1909.07974
- N. Rabin, D. Fishelov. Multi-scale kernels for Nyström based extension schemes. *Appl. Math. Comput.*, 2017. doi:10.1016/j.amc.2017.02.025. http://www.math.tau.ac.il/~daliaf/Rabin-Fishelov-Multi-Scale-Kernels-2016.pdf
- E. Chiavazzo, C. W. Gear, C. J. Dsilva, N. Rabin, I. G. Kevrekidis. Reduced models in chemical kinetics via nonlinear data-mining. *Processes* 2(1):112–140, 2014. doi:10.3390/pr2010112. https://arxiv.org/abs/1307.6849
- N. Evangelou, F. Dietrich, E. Chiavazzo, D. Lehmberg, M. Meila, I. G. Kevrekidis. Double diffusion maps and their latent harmonics for scientific computations in latent space. *J. Comput. Phys.* 112072, 2023. doi:10.1016/j.jcp.2023.112072. https://arxiv.org/abs/2204.12536
- P. G. Papaioannou, R. Talmon, I. G. Kevrekidis, C. Siettos. Time-series forecasting using manifold learning, radial basis function interpolation, and geometric harmonics. *Chaos* 32:083113, 2022. https://arxiv.org/abs/2110.03625
- K. R. dos Santos, D. G. Giovanis, K. Kontolati, D. Loukrezis, M. D. Shields. Grassmannian diffusion maps based surrogate modeling via geometric harmonics. *IJNME*, 2022. https://arxiv.org/abs/2109.13805
- K. Kontolati, D. Loukrezis, K. R. dos Santos, D. G. Giovanis, M. D. Shields. Manifold learning-based polynomial chaos expansions for high-dimensional surrogate models. *IJUQ*, 2022. https://arxiv.org/abs/2107.09814
- D. G. Giovanis, M. D. Shields. Data-driven surrogates for high dimensional models using Gaussian process regression on the Grassmann manifold. *CMAME* 113269, 2020. https://arxiv.org/abs/2003.11910
- E. D. Koronaki, N. Evangelou, Y. M. Psarellis, A. G. Boudouvis, I. G. Kevrekidis. From partial data to out-of-sample parameter and observation estimation with diffusion maps and geometric harmonics. *Comput. Chem. Eng.*, 2023. https://arxiv.org/abs/2301.11728
- C. J. Dsilva, R. Talmon, R. R. Coifman, I. G. Kevrekidis. Parsimonious representation of nonlinear dynamical systems through manifold learning. *ACHA* 44(3):759–773, 2018. https://arxiv.org/abs/1505.06118
- I. G. Kevrekidis, C. W. Gear, J. M. Hyman, P. G. Kevrekidis, O. Runborg, C. Theodoropoulos. Equation-free, coarse-grained multiscale computation. *Commun. Math. Sci.* 1(4):715–762, 2003. https://arxiv.org/abs/physics/0209043
- C. W. Gear, T. J. Kaper, I. G. Kevrekidis, A. Zagaris. Projecting to a slow manifold: singularly perturbed systems and legacy codes. *SIADS* 4(3):711–732, 2005. https://arxiv.org/abs/physics/0405074
- A. Zagaris, C. W. Gear, T. J. Kaper, I. G. Kevrekidis. Analysis of the accuracy and convergence of equation-free projection to a slow manifold. *ESAIM: M2AN* 43:757–784, 2009. https://arxiv.org/abs/0707.1614
- C. Vandekerckhove, B. E. Sonday, A. G. Makeev, D. Roose, I. G. Kevrekidis. A common approach to the computation of coarse-scale steady states and to consistent initialization on a slow manifold. *Comput. Chem. Eng.* 35:1949–1958, 2011. https://arxiv.org/abs/1004.5089
- K. Debrabant, G. Samaey. A micro/macro algorithm to accelerate Monte Carlo simulation of SDEs. https://arxiv.org/abs/1009.3767 ; K. Debrabant, G. Samaey, P. Zieliński. *SIAM J. Numer. Anal.*, 2017. https://arxiv.org/abs/1511.06171 ; T. Lelièvre, G. Samaey, P. Zieliński. https://arxiv.org/abs/1801.01740
- S. Mika, B. Schölkopf, A. Smola, K.-R. Müller, M. Scholz, G. Rätsch. Kernel PCA and de-noising in feature spaces. NIPS 11, 1999. https://papers.nips.cc/paper/1491-kernel-pca-and-de-noising-in-feature-spaces.pdf
- J. T. Kwok, I. W. Tsang. The pre-image problem in kernel methods. *IEEE TNN* 15(6):1517–1525, 2004. https://www.cse.ust.hk/~jamesk/papers/tnn04b.pdf
- G. H. Bakır, J. Weston, B. Schölkopf. Learning to find pre-images. NIPS 16, 2004. https://papers.nips.cc/paper/2417-learning-to-find-pre-images.pdf
- P. Arias, G. Randall, G. Sapiro. Connecting the out-of-sample and pre-image problems in kernel methods. CVPR 2007. doi:10.1109/CVPR.2007.383038. https://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_CVPR_2007/data/papers/0070.pdf
- P. Honeine, C. Richard. Preimage problem in kernel-based machine learning. *IEEE Signal Processing Magazine* 28(2):77–88, 2011. https://www.cedric-richard.fr/Articles/honeine2011preimage.pdf
- P. Étyngier, F. Ségonne, R. Keriven. Projection onto a shape manifold for image segmentation with prior. ICIP 2007. http://imagine.enpc.fr/publications/papers/ICIP2007.pdf ; CERTIS report 07-33: http://imagine.enpc.fr/publications/papers/CERTIS0733.pdf ; SSVM 2007: http://imagine.enpc.fr/publications/papers/SSVM07.pdf
- P. Étyngier, F. Ségonne, R. Keriven. Shape priors using manifold learning techniques. ICCV 2007. doi:10.1109/ICCV.2007.4409040 (**not accessed**; see companions above)
- P. Thorstensen, F. Ségonne, R. Keriven. Pre-image as Karcher mean using diffusion maps: application to shape and image denoising. SSVM 2009, LNCS 5567:721–732. http://imagine.enpc.fr/publications/papers/SSVM09.pdf
- N. Thorstensen, P. Étyngier, F. Ségonne, R. Keriven. Diffusion maps as a framework for shape modeling. *CVIU* 115(4):520–530, 2011. https://csc.univie.ac.at/paper/ThoEtySegKer11.pdf
- S. Dambreville, Y. Rathi, A. Tannenbaum. A framework for image segmentation using shape models and kernel space shape priors. *IEEE TPAMI* 30(8):1385–1399, 2008. https://pmc.ncbi.nlm.nih.gov/articles/PMC2800082
- M. Á. Carreira-Perpiñán, Z. Lu. The Laplacian eigenmaps latent variable model. AISTATS 2007, PMLR 2:59–66. https://proceedings.mlr.press/v2/carreira-perpinan07a.html
- S. Gerber, T. Tasdizen, R. Whitaker. Dimensionality reduction and principal surfaces via kernel map manifolds. ICCV 2009, pp. 529–536. https://www.sci.utah.edu/~tolga/pubs/GerberICCV2009.pdf
- M. Ovsjanikov, M. Ben-Chen, J. Solomon, A. Butscher, L. Guibas. Functional maps: a flexible representation of maps between shapes. *ACM TOG* 31(4):30, 2012. http://www.lix.polytechnique.fr/~maks/papers/obsbg_fmaps.pdf
- M. Fumero, M. Pegoraro, V. Maiorca, F. Locatello, E. Rodolà. Latent functional maps: a spectral framework for representation alignment. NeurIPS 2024. https://arxiv.org/abs/2406.14183
- L. Moschella, V. Maiorca, M. Fumero, A. Norelli, F. Locatello, E. Rodolà. Relative representations enable zero-shot latent space communication. ICLR 2023. https://arxiv.org/abs/2209.15430
- L. Zelnik-Manor, P. Perona. Self-tuning spectral clustering. NIPS 17, 2004. https://papers.nips.cc/paper_files/paper/2004/hash/40173ea48d9567f1f393b20c855bb40b-Abstract.html
- T. Berry, J. Harlim. Variable bandwidth diffusion kernels. *ACHA* 40:68–96, 2016. https://arxiv.org/abs/1406.5064
- R. R. Coifman, Y. Shkolnisky, F. J. Sigworth, A. Singer. Graph Laplacian tomography from unknown random projections. *IEEE TIP* 17(10):1891–1899, 2008. https://web.math.princeton.edu/~amits/publications/2Dtomography_published.pdf
- T. Berry, T. Sauer. Consistent manifold representation for topological data analysis. *Found. Data Sci.* 1(1):1–38, 2019. https://arxiv.org/abs/1606.02353
- N. F. Marshall, R. R. Coifman. Manifold learning with bi-stochastic kernels. *IMA J. Appl. Math.* 84(3):455–482, 2019. https://arxiv.org/abs/1711.06711
- C. L. Wormell, S. Reich. Spectral convergence of diffusion maps: improved error bounds and an alternative normalisation. *SIAM J. Numer. Anal.* 59(3):1687–1734, 2021. https://arxiv.org/abs/2006.02037
- B. Landa, R. R. Coifman, Y. Kluger. Doubly stochastic normalization of the Gaussian kernel is robust to heteroskedastic noise. *SIMODS* 3(1):388–413, 2021. https://arxiv.org/abs/2006.00402
- B. Landa, X. Cheng. Robust inference of manifold density and geometry by doubly stochastic scaling. arXiv:2209.08004. https://arxiv.org/abs/2209.08004
- X. Cheng, B. Landa. Bi-stochastically normalized graph Laplacian: convergence to manifold Laplacian and robustness to outlier noise. arXiv:2206.11386. https://arxiv.org/abs/2206.11386
- R. R. Coifman, M. J. Hirn. Bi-stochastic kernels via asymmetric affinity functions. *ACHA* 35(1):177–180, 2013. https://arxiv.org/abs/1209.0237
- H. Van Assel, T. Vayer, R. Flamary, N. Courty. SNEkhorn: dimension reduction with symmetric entropic affinities. NeurIPS 2023. https://arxiv.org/abs/2305.13797
- A.-A. Pooladian, J. Niles-Weed. Entropic estimation of optimal transport maps. arXiv:2109.12004. https://arxiv.org/abs/2109.12004
- Y. Bengio, J.-F. Paiement, P. Vincent, O. Delalleau, N. Le Roux, M. Ouimet. Out-of-sample extensions for LLE, Isomap, MDS, Eigenmaps, and spectral clustering. NIPS 16, 2004. https://papers.nips.cc/paper_files/paper/2003/hash/cf05968255451bdefe3c5bc64d550517-Abstract.html
- C. Fowlkes, S. Belongie, F. Chung, J. Malik. Spectral grouping using the Nyström method. *IEEE TPAMI* 26(2):214–225, 2004. https://people.cs.umass.edu/~mahadeva/cs791bb/reading/fowlkes-nystrom.pdf
- D. Schnitzer, A. Flexer, M. Schedl, G. Widmer. Local and global scaling reduce hubs in space. *JMLR* 13:2871–2902, 2012. https://www.jmlr.org/papers/v13/schnitzer12a.html
- A. Conneau, G. Lample, M. Ranzato, L. Denoyer, H. Jégou. Word translation without parallel data. ICLR 2018. https://arxiv.org/abs/1710.04087
- G. Mishne, U. Shaham, A. Cloninger, I. Cohen. Diffusion nets. *ACHA* 47(2):259–285, 2019. https://arxiv.org/abs/1506.07840
- A. F. Duque, S. Morin, G. Wolf, K. R. Moon. Extendable and invertible manifold learning with geometry regularized autoencoders. IEEE Big Data 2020; TPAMI version "Geometry regularized autoencoders", doi:10.1109/TPAMI.2022.3222104 (TPAMI version not read). https://arxiv.org/abs/2007.07142 ; code https://github.com/KevinMoonLab/GRAE
- N. D. Lawrence, J. Quiñonero-Candela. Local distance preservation in the GP-LVM through back constraints. ICML 2006. https://inverseprobability.com/publications/files/backConstraints.pdf
- L. K. Saul, S. T. Roweis. Think globally, fit locally: unsupervised learning of low dimensional manifolds. *JMLR* 4:119–155, 2003. https://jmlr.org/papers/volume4/saul03a/saul03a.pdf
- Z. Zhang, H. Zha. Principal manifolds and nonlinear dimension reduction via local tangent space alignment. *SIAM J. Sci. Comput.* 26(1):313–338, 2004. https://arxiv.org/abs/cs/0212008
- G. A. Kevrekidis, Z. Ahmad, M. Maggioni, S. Villar, Y. G. Kevrekidis. Thinner latent spaces: detecting dimension and imposing invariance with conformal autoencoders. arXiv:2408.16138. https://arxiv.org/abs/2408.16138
- R. R. Coifman, M. J. Hirn. Diffusion maps for changing data. *ACHA* 36(1):79–107, 2014. https://arxiv.org/abs/1209.0245
- V. Borovitskiy, I. Azangulov, A. Terenin, P. Mostowsky, M. Deisenroth, N. Durrande. Matérn Gaussian processes on graphs. AISTATS 2021. https://arxiv.org/abs/2010.15538 ; V. Borovitskiy, A. Terenin, P. Mostowsky, M. Deisenroth. Matérn Gaussian processes on Riemannian manifolds. NeurIPS 2020. https://arxiv.org/abs/2006.10160
- D. B. Dunson, H.-T. Wu, N. Wu. Graph based Gaussian processes on restricted domains. *JRSS-B*, 2022. https://arxiv.org/abs/2010.07242
- G. Froyland, P. Koltai. Detecting the birth and death of finite-time coherent sets. *CPAM* 76(12):3642–3684, 2023. https://arxiv.org/abs/2103.16286
- G. Froyland, M. Kalia, P. Koltai. Spectral clustering of time-evolving networks using the inflated dynamic Laplacian for graphs / Multi-set spectral clustering of time-evolving networks using the supra-Laplacian. arXiv:2409.11984. https://arxiv.org/abs/2409.11984
- R. Banisch, P. Koltai. Understanding the geometry of transport: diffusion maps for Lagrangian trajectory data unravel coherent sets. *Chaos* 27:035804, 2017. https://arxiv.org/abs/1603.04709
- O. Lindenbaum, A. Yeredor, M. Salhov, A. Averbuch. Multi-view diffusion maps. *Information Fusion* 55:127–149, 2020. https://arxiv.org/abs/1508.05550
- S. L. Brunton, B. W. Brunton, J. L. Proctor, E. Kaiser, J. N. Kutz. Chaos as an intermittently forced linear system. *Nat. Commun.* 8:19, 2017. https://arxiv.org/abs/1608.05306
- S. Marks, C. Rager, E. J. Michaud, Y. Belinkov, D. Bau, A. Mueller. Sparse feature circuits. ICLR 2025. https://arxiv.org/abs/2403.19647
- E. Ameisen, J. Lindsey, A. Pearce, W. Gurnee, et al. Circuit tracing: revealing computational graphs in language models. Transformer Circuits Thread, 2025. https://transformer-circuits.pub/2025/attribution-graphs/methods.html (partially read)
- R. Baptista, Y. Marzouk, O. Zahm. On the representation and learning of monotone triangular transport maps. *Found. Comput. Math.* 24:2063–2108, 2024. https://arxiv.org/abs/2009.10303
- Y. Marzouk, T. Moselhy, M. Parno, A. Spantini. An introduction to sampling via measure transport. *Handbook of UQ*, 2016. https://arxiv.org/abs/1602.05023
- R. Baptista, B. Hosseini, N. Kovachki, Y. Marzouk. Conditional sampling with monotone GANs. *SIAM/ASA JUQ* 12(3):868–900, 2024. https://arxiv.org/abs/2006.06755

**Partially read.**
- M. Radovanović, A. Nanopoulos, M. Ivanović. Hubs in space. *JMLR* 11:2487–2531, 2010. https://www.jmlr.org/papers/v11/radovanovic10a.html
- R. Feldbauer, A. Flexer. A comprehensive empirical comparison of hubness reduction in high-dimensional spaces. *KAIS* 59:137–166, 2019. https://pmc.ncbi.nlm.nih.gov/articles/PMC7327987/
- Pan et al. Hubness reduction with dual bank Sinkhorn normalization for cross-modal retrieval. ACM MM 2025. https://arxiv.org/abs/2508.02538

**Abstract only, or not accessed.**
- N. Rabin, R. R. Coifman. Heterogeneous datasets representation and learning using diffusion maps and Laplacian pyramids. SDM 2012, pp. 189–199. https://epubs.siam.org/doi/10.1137/1.9781611972825.17 (not accessed; closed access)
- R. Zass, A. Shashua. Doubly stochastic normalization for spectral clustering. NIPS 19, 2006. https://proceedings.neurips.cc/paper/2006/hash/4fa177df22864518b2d7818d4db5db2d-Abstract.html (abstract only)
- G. Dinu, A. Lazaridou, M. Baroni. Improving zero-shot learning by mitigating the hubness problem. ICLR 2015 workshop. https://arxiv.org/abs/1412.6568 (abstract only)
- S. L. Smith, D. H. P. Turban, S. Hamblin, N. Y. Hammerla. Offline bilingual word vectors, orthogonal transformations and the inverted softmax. ICLR 2017. https://arxiv.org/abs/1702.03859 (abstract only)
- P. Honeine, C. Richard. Solving the pre-image problem in kernel machines: a direct method. IEEE MLSP 2009 (not accessed).
- Y. Rathi, S. Dambreville, A. Tannenbaum. Statistical shape analysis using kernel PCA. SPIE 6064, 2006. doi:10.1117/12.641417 (not accessed).
- E. D. Koronaki et al. A data-driven reduced-order model of nonlinear processes based on diffusion maps and artificial neural networks. *Chem. Eng. J.* 397:125475, 2020 (not accessed).
- N. Thorstensen, Manifold learning and applications to shape and image processing, PhD thesis, ENPC 2009 (pastel-00005860), and P. Étyngier, Statistical learning, shape manifolds and applications to image segmentation, PhD thesis, ENPC 2008 (pastel-00004040). Not accessed (HAL bot wall).
- M. Kilian, N. J. Mitra, H. Pottmann. Geometric modeling in shape space. *ACM TOG* 26(3), 2007 (abstract only).
- C. E. Rasmussen, C. K. I. Williams. *Gaussian Processes for Machine Learning*. MIT Press, 2006 (standard formulas quoted, not re-read).
- C. J. Dsilva et al. *J. Chem. Phys.* 139:184109, 2013 (second-hand only).
