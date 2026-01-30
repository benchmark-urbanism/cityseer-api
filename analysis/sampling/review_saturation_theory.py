"""
Statistical Review: Network Saturation and Design Effects for Betweenness Sampling

Reviewing whether the proposed DEFF = 1 + α × (reach/N)² framework is theoretically sound.

Author: Statistical Review
Date: 2026-01-30
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR / "paper" / "figures"

print("=" * 70)
print("STATISTICAL REVIEW: NETWORK SATURATION THEORY")
print("=" * 70)

# =============================================================================
# SECTION 1: IS "NETWORK SATURATION" THE RIGHT FRAMING?
# =============================================================================

print("""
QUESTION 1: Is reach/N the right quantity to condition on?

CONCERN: The ratio reach/N conflates two distinct phenomena:

1. REACHABILITY SATURATION: As distance increases, reach → N
   This is a property of the DISTANCE THRESHOLD relative to network diameter.

2. SAMPLING FRACTION: The fraction of reachable nodes we sample = p
   This is our DESIGN CHOICE.

The proposed model DEFF = 1 + α × (reach/N)² implies that accuracy degrades
simply because MORE nodes are reachable, independent of how many we sample.

But this conflates:
- The SIZE of the population we're estimating over (reach)
- The FRACTION of the total network that population represents (reach/N)

STATISTICAL PRINCIPLE:
In standard survey sampling, what matters is:
- n (sample size)
- N_pop (population size being estimated)
- The DESIGN (how samples are selected)

The ratio N_pop/N_total shouldn't directly affect variance unless it changes
the correlation structure of the estimand.
""")

# =============================================================================
# SECTION 2: WHAT ACTUALLY CAUSES VARIANCE IN BETWEENNESS ESTIMATION?
# =============================================================================

print("""
QUESTION 2: What is the actual source of variance in sampled betweenness?

BETWEENNESS DEFINITION:
    B(v) = Σ_{s≠v≠t} σ_st(v) / σ_st

where σ_st(v) = number of shortest paths from s to t passing through v.

SAMPLED ESTIMATOR (sampling sources with probability p):
    B̂(v) = (1/p) × Σ_{s∈S} Σ_{t: t reachable from s} σ_st(v) / σ_st

The variance comes from:
1. Which sources s are sampled (design variance)
2. How much the contribution from each source varies (population variance)

KEY INSIGHT: The contribution from source s to node v's betweenness is:
    C_s(v) = Σ_{t≠s,v} σ_st(v) / σ_st

The variance of B̂(v) depends on Var(C_s(v)) across sources s.
""")

# =============================================================================
# SECTION 3: WHEN DOES CORRELATION BETWEEN SOURCES MATTER?
# =============================================================================

print("""
QUESTION 3: When are source contributions correlated?

For independent sampling of sources, the standard result is:
    Var(B̂(v)) = (1-p)/(n×p) × Var_s(C_s(v))

The "design effect" framework applies when samples are CLUSTERED or
STRATIFIED, not when they're independent.

HOWEVER, there's a subtlety with betweenness:

The contributions C_s(v) and C_{s'}(v) from different sources s, s' are
NOT independent random variables - they share network structure.

Specifically, if s and s' are "similar" (close in the network, similar
centrality), their contributions to v may be correlated.

BUT: This correlation exists regardless of reach/N. It depends on:
- Network topology
- The specific node v
- The positions of s and s' relative to v

The ratio reach/N doesn't directly determine this correlation.
""")

# =============================================================================
# SECTION 4: WHAT MIGHT ACTUALLY BE HAPPENING AT LARGE DISTANCES?
# =============================================================================

print("""
QUESTION 4: What changes at regional scales that could increase variance?

Several hypotheses:

HYPOTHESIS A: PATH LENGTH DISTRIBUTION
At large distances, paths are longer. Betweenness contributions involve
counting paths, and longer paths may have higher variance in how they
distribute across intermediate nodes.

    → This would affect Var(C_s(v)), not DEFF

HYPOTHESIS B: BOUNDARY EFFECTS
At large distances, many nodes are near the "boundary" of reachability.
These boundary nodes have truncated path counts, introducing bias/variance.

    → This is about EDGE EFFECTS, not saturation per se

HYPOTHESIS C: BETWEENNESS DISTRIBUTION CHANGES
At regional scales, the betweenness distribution may become more skewed
or heavy-tailed, making estimation harder.

    → This affects the ESTIMAND, not the sampling design

HYPOTHESIS D: EFFECTIVE SAMPLE SIZE REDUCTION
If we're sampling p of reach nodes, but reach ≈ N, we're sampling p×N nodes.
The "effective" information per sample might decrease because...

    → This needs more careful analysis
""")

# =============================================================================
# SECTION 5: A MORE PRINCIPLED APPROACH
# =============================================================================

print("""
QUESTION 5: What would a principled variance model look like?

For Horvitz-Thompson / Hájek estimators, the variance is:

    Var(B̂(v)) ≈ (1/n²) × Σ_s Σ_{s'} (π_ss' - π_s×π_{s'}) × C_s(v) × C_{s'}(v) / (π_s × π_{s'})

For simple random sampling with probability p:
    π_s = p  (first-order inclusion)
    π_ss' = p² (second-order, assuming independence)

This gives the standard formula with no design effect for independent sampling.

A design effect > 1 would arise if:
    π_ss' > π_s × π_{s'} for "similar" s, s'

This happens in CLUSTER sampling (if s is selected, nearby s' more likely).
Our sampling is NOT clustered - it's simple random.

CONCLUSION: The design effect framework may not be the right model.
""")

# =============================================================================
# SECTION 6: ALTERNATIVE EXPLANATION FOR THE 20KM DIP
# =============================================================================

print("""
QUESTION 6: What else could explain the betweenness dip at 20km?

Let's consider the SPECIFIC observation:
- 10km: ρ_betweenness = 0.993
- 20km: ρ_betweenness = 0.975

The drop is about 0.018 in Spearman correlation. Possible causes:

1. EFFECTIVE SAMPLE SIZE IS LOWER THAN CALCULATED
   The probing uses 25th percentile reach, which at 20km severely
   underestimates actual reach (6,700 vs 27,600). But the actual
   eff_n is still ~2,700 which should be plenty.

2. THE MODEL IS MISCALIBRATED FOR BETWEENNESS AT REGIONAL SCALES
   The model was fit on synthetic networks up to 5km. Betweenness
   at 20km may have different statistical properties.

3. BETWEENNESS HAS HIGHER INTRINSIC VARIANCE AT LARGE SCALES
   As distance → network diameter, betweenness concentrates on
   a few key "bridge" nodes. The distribution becomes more skewed,
   making rank correlation harder to preserve.

4. FINITE POPULATION CORRECTION
   When sampling without replacement from a finite population,
   variance decreases. But we're not accounting for this properly.

Let's check hypothesis 3 - does betweenness become more concentrated?
""")

# =============================================================================
# NUMERICAL ANALYSIS: BETWEENNESS DISTRIBUTION AT DIFFERENT SCALES
# =============================================================================

print("\n" + "=" * 70)
print("NUMERICAL ANALYSIS")
print("=" * 70)

# Simulate how betweenness distribution might change with scale
# Using a simplified model

np.random.seed(42)

def simulate_betweenness_distribution(n_nodes: int, concentration: float) -> np.ndarray:
    """
    Simulate a betweenness-like distribution.

    Higher concentration = more skewed (few nodes have most betweenness)
    """
    # Use Pareto-like distribution
    raw = np.random.pareto(concentration, n_nodes)
    return raw / raw.sum()  # Normalize

def spearman_under_sampling(true_vals: np.ndarray, sample_frac: float, n_sims: int = 100) -> tuple:
    """
    Simulate Spearman correlation when estimating from samples.
    """
    n = len(true_vals)
    n_sample = max(10, int(n * sample_frac))

    rhos = []
    for _ in range(n_sims):
        # Sample and estimate (simplified model)
        sample_idx = np.random.choice(n, n_sample, replace=False)

        # Estimated values have noise proportional to 1/sqrt(contribution from sampled)
        noise_scale = 0.1 / np.sqrt(sample_frac)
        estimated = true_vals * (1 + np.random.normal(0, noise_scale, n))

        # Spearman correlation
        from scipy.stats import spearmanr
        rho, _ = spearmanr(true_vals, estimated)
        rhos.append(rho)

    return np.mean(rhos), np.std(rhos)

print("\nSimulating betweenness estimation at different concentration levels:")
print("-" * 60)

concentrations = [2.0, 1.5, 1.0, 0.5]  # Lower = more concentrated
sample_frac = 0.1

for conc in concentrations:
    true_betw = simulate_betweenness_distribution(1000, conc)
    gini = 1 - 2 * np.sum(np.cumsum(np.sort(true_betw)) / np.sum(true_betw)) / len(true_betw)
    mean_rho, std_rho = spearman_under_sampling(true_betw, sample_frac)
    print(f"  Concentration={conc:.1f} (Gini={gini:.2f}): ρ = {mean_rho:.3f} ± {std_rho:.3f}")

# =============================================================================
# SECTION 7: REVISED THEORETICAL FRAMEWORK
# =============================================================================

print("""

REVISED THEORETICAL FRAMEWORK
=============================

After review, the reach/N "saturation" framing has issues:

1. It's not clear why reach/N should directly affect variance
2. The design effect framework applies to clustered sampling, not ours
3. The correlation between source contributions doesn't depend on reach/N

BETTER HYPOTHESIS: Scale-dependent variance of betweenness contributions

At different distance scales, the variance of C_s(v) (contribution from
source s to node v's betweenness) changes:

    Var(B̂(v)) ∝ Var_s(C_s(v)) / (reach × p)

If Var_s(C_s(v)) INCREASES with distance (because betweenness becomes
more concentrated/skewed), then we need more samples to compensate.

PROPOSED MODEL:

    eff_n_adjusted = reach × p / (1 + γ × CV²(C_s(v)))

where CV(C_s(v)) is the coefficient of variation of source contributions,
which may increase with distance.

This is more principled than reach/N because it's based on the actual
statistical quantity that drives variance.

PRACTICAL IMPLICATION:

Rather than using reach/N, we should empirically measure how Var(C_s(v))
scales with distance on real networks, then build that into the model.

The current k parameter attempts to capture this but was fit on networks
where the scale-dependent variance effect wasn't fully expressed.
""")

# =============================================================================
# FINAL RECOMMENDATION
# =============================================================================

print("""
FINAL RECOMMENDATION
====================

1. The reach/N "saturation" model is NOT theoretically well-founded
   for independent random sampling. Don't use it.

2. The observed betweenness dip at 20km is real but likely due to:
   - Higher intrinsic variance of betweenness at regional scales
   - Model miscalibration (fit on 5km max, extrapolating to 20km)

3. For a principled fix, we should:
   a) Empirically characterize how Var(C_s(v)) scales with distance
   b) Fit a distance-dependent variance model
   c) Use that to adjust required sample size

4. SHORT-TERM FIX: Simply use a higher target ρ for betweenness
   (e.g., target 0.98 internally to achieve 0.95 observed), or
   use a distance-dependent safety margin.

5. The k parameter as currently formulated doesn't help because
   it was fit in a regime where the problematic effects don't appear.
""")

# Generate a summary figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

distances = [5, 10, 20]
observed_rho = [1.0, 0.993, 0.975]
predicted_rho = [1.0, 0.97, 0.97]  # What the model predicts

ax.plot(distances, observed_rho, 'bo-', markersize=10, lw=2, label='Observed ρ (betweenness)')
ax.plot(distances, predicted_rho, 'g--', markersize=8, lw=2, label='Model prediction')
ax.axhline(0.95, color='red', ls=':', label='Target ρ = 0.95')

ax.fill_between(distances, [0.95]*3, observed_rho, alpha=0.2, color='blue')

ax.set_xlabel('Distance (km)', fontsize=12)
ax.set_ylabel('Spearman ρ', fontsize=12)
ax.set_title('Betweenness Accuracy: Observed vs Model\n(Gap increases at regional scales)', fontsize=12)
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.9, 1.01)
ax.set_xticks(distances)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'saturation_theory_review.pdf', bbox_inches='tight')
print(f"\nSaved: {FIGURES_DIR / 'saturation_theory_review.pdf'}")
