# Title Options for Adaptive Sampling Paper

## Paper Core Contribution

The paper's central insight is that **effective sample size (reachability x probability) is a sufficient statistic** for predicting centrality estimation accuracy. This enables:
- Per-distance adaptive sampling that maintains consistent accuracy across scales
- ~2x speedup while preserving ranking accuracy (Spearman rho >= 0.95)
- Practical regional-scale network analysis that was previously computationally prohibitive

---

## Proposed Titles

### Option 1: "Effective Sample Size: A Unifying Predictor for Multi-Scale Network Centrality Approximation"

**Pros:**
- Leads with the key theoretical contribution (effective sample size)
- "Unifying predictor" signals that this is a single principle that explains accuracy across metrics, topologies, and scales
- Clear and precise language appropriate for academic venue
- 12 words - good length

**Cons:**
- "Unifying predictor" is slightly abstract for practitioners
- Doesn't explicitly mention urban networks or streets
- "Approximation" may sound less rigorous than "estimation"

---

### Option 2: "Why Sampling Works at Long Distances: Effective Sample Size and Adaptive Network Centrality"

**Pros:**
- Conversational hook ("Why Sampling Works") draws readers in
- Implicitly addresses the puzzle that motivates the paper (sampling works well at long distances but fails at short)
- Memorable and specific
- 13 words

**Cons:**
- "Why Sampling Works" phrasing may be too informal for CEUS
- Could be misread as only about long distances (the paper is about all distances)
- Doesn't mention urban/street networks

---

### Option 3: "Adaptive Per-Distance Sampling for Urban Network Centrality: An Effective Sample Size Approach"

**Pros:**
- Comprehensive - covers method (adaptive sampling), domain (urban networks), and theory (effective sample size)
- "Per-Distance" signals the key innovation (calibrating probability at each distance threshold)
- Clearly positions for urban analytics audience
- 13 words

**Cons:**
- Somewhat technical and descriptive rather than insight-driven
- "Per-Distance" may not resonate with readers unfamiliar with multi-scale analysis
- Less memorable than more focused alternatives

---

### Option 4: "Reachability-Aware Sampling for Distance-Bounded Network Centrality in Urban Street Networks"

**Pros:**
- "Reachability-Aware" captures the mechanism (sampling adapts based on how many nodes are reachable)
- "Distance-Bounded" is established terminology from Cooper (2015)
- Explicitly names the application domain (urban street networks)
- 12 words

**Cons:**
- "Reachability-Aware" is novel terminology that may not immediately convey the contribution
- Doesn't explicitly mention "effective sample size" (the core theoretical concept)
- Dense with technical terms

---

### Option 5: "Precision Scales with Importance: Adaptive Sampling for Multi-Scale Urban Centrality Analysis"

**Pros:**
- "Precision Scales with Importance" is a memorable, insight-driven phrase that captures a key finding: high-centrality nodes automatically get better estimates
- Positions the practical benefit (you get precision where it matters)
- "Multi-Scale" signals the breadth of application
- 12 words

**Cons:**
- "Precision Scales with Importance" requires explanation - may be cryptic to some readers
- Doesn't mention effective sample size explicitly
- Could be misinterpreted as about weighting or prioritisation rather than a natural property of sampling

---

## Evaluation Against Criteria

| Criterion | Option 1 | Option 2 | Option 3 | Option 4 | Option 5 |
|-----------|----------|----------|----------|----------|----------|
| Memorable and specific | Medium | High | Medium | Medium | High |
| Communicates key insight | High | High | Medium | Medium | High |
| Appropriate for CEUS | High | Medium | High | High | High |
| Under 15 words | Yes (12) | Yes (13) | Yes (13) | Yes (12) | Yes (12) |
| Includes domain context | No | No | Yes | Yes | Yes |
| Includes theoretical contribution | Yes | Yes | Yes | Partial | Partial |

---

## Editor Suggestions Assessment

**"Effective Sample Size: A Unifying Framework for Efficient Network Centrality Computation"**
- Good: Leads with the key concept
- Concern: "Framework" overstates the contribution (this is an empirical finding + algorithm, not a general framework)
- Concern: "Efficient...Computation" is generic; doesn't distinguish from many other papers on fast centrality

**"Why Sampling Fails at Short Distances (And How to Fix It)"**
- Good: Memorable, identifies a real problem practitioners face
- Concern: Framing around "failure" is negative; the paper is more about understanding why it succeeds
- Concern: Informal parenthetical may not suit CEUS style
- Could work better as a blog post title

---

## Final Recommendation: Option 1 (with minor refinement)

### Recommended Title:

**"Effective Sample Size as a Unifying Predictor for Multi-Scale Network Centrality Estimation"**

(Changed "A Unifying Predictor" to "as a Unifying Predictor" and "Approximation" to "Estimation" for flow)

### Rationale:

1. **Leads with the key insight**: The title immediately signals what's new (effective sample size as the key quantity) rather than burying it after the method description.

2. **Academic but clear**: "Unifying predictor" is precise language that signals both theoretical contribution (one quantity explains accuracy) and practical value (can be used to predict/calibrate sampling).

3. **Appropriate scope**: "Multi-Scale" captures that this applies across distance thresholds, which is the core challenge. Avoids overspecifying (urban/street) while remaining relevant to CEUS readers.

4. **Memorable**: Readers will remember "effective sample size" as the takeaway, which is exactly what they need for their own implementations.

5. **Honest about contribution level**: This is an empirical finding with practical application, not a grand framework. The title reflects that appropriately.

### Alternative (if domain specificity needed):

If reviewers or editors request explicit urban context, consider:

**"Effective Sample Size as a Unifying Predictor for Multi-Scale Centrality in Urban Street Networks"** (14 words)

This adds domain context while preserving the insight-driven structure.

---

## Summary Table

| Rank | Title | Key Strength |
|------|-------|--------------|
| 1 | Effective Sample Size as a Unifying Predictor for Multi-Scale Network Centrality Estimation | Leads with theoretical insight, appropriate tone |
| 2 | Adaptive Per-Distance Sampling for Urban Network Centrality: An Effective Sample Size Approach | Comprehensive, clearly describes method |
| 3 | Precision Scales with Importance: Adaptive Sampling for Multi-Scale Urban Centrality Analysis | Memorable phrase, highlights practical benefit |
| 4 | Why Sampling Works at Long Distances: Effective Sample Size and Adaptive Network Centrality | Engaging hook, explains the puzzle |
| 5 | Reachability-Aware Sampling for Distance-Bounded Network Centrality in Urban Street Networks | Technically precise, uses established terminology |

---

*Generated: 2026-01-29*
