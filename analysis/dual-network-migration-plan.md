# Dual-Network Migration Plan

## Purpose

This document proposes a low-disruption migration from the current mixed primal/dual execution model toward a dual-network canonical architecture for node-based centrality and accessibility analysis.

The intent is to simplify the internal implementation, make angular path semantics more defensible, and reduce user-facing confusion without forcing an abrupt API break.

## Executive Summary

Recommended direction:

- Accept primal or dual graphs at the Python boundary.
- Canonicalize node-based analysis onto a dual graph internally.
- Expose path semantics in terms of `path_cost` and `cutoff_cost`, not primal vs dual execution details.
- Keep segment centrality separate and do not migrate it to dual execution.
- Deprecate primal-angular execution paths gradually, with compatibility wrappers and explicit warnings.

The least disruptive migration is not a flag day. It is a staged transition with internal canonicalization first, then API simplification, then removal of legacy execution paths only after release cycles and validation.

## Why Change

### Current problems

- The current API exposes implementation detail: users choose primal vs dual, shortest vs simplest, and sometimes implicitly rely on topology-specific behaviour.
- Angular analysis on primal graphs is stateful in a way that ordinary node-based Dijkstra and Brandes semantics do not represent cleanly.
- Sidestep-suppression logic is compensating for a weak state model rather than being a clean property of the graph representation.
- Supporting both primal and dual execution paths multiplies testing and maintenance cost.

### Why dual should be canonical

- Dual nodes encode segment state directly.
- Angular transitions become segment-to-segment transitions with clearer semantics.
- Shortest and angular traversals can share more machinery.
- Brandes-style angular betweenness becomes much more plausible on dual graphs than on primal graphs.

## Non-Goals

- This migration does not require removing support for primal graph input.
- This migration does not require changing segment centrality to run on dual graphs.
- This migration does not require a single public breaking release.
- This migration does not require exposing dual graphs to users who do not care about internal topology.

## Design Principles

1. Preserve user intent, not existing internal structure.
2. Avoid silent behavioural changes where rankings or reachability may materially differ.
3. Keep old entry points working during the migration window.
4. Make new semantics explicit before removing old semantics.
5. Test old and new execution modes in parallel until confidence is high.

## Target Architecture

### Canonical internal model

- Node-based centrality and accessibility computations use a dual graph internally.
- Primal input graphs are converted to dual once near the Python boundary.
- Results are mapped back to the expected user-facing node keys or edge/segment representation as needed.

### Public API model

Users should choose:

- `path_cost`
  - `metric`
  - `angular`
  - reserved for future: `angular_then_metric`
- `cutoff_cost`
  - `metric`
  - reserved for future: `angular`

Users should not need to choose:

- primal execution vs dual execution for node-based analysis
- primal-angular vs dual-angular implementation variants

### Things that remain separate

- Segment centrality stays on its own execution path.
- Dual graph construction utilities remain public for users who explicitly want topology conversion.
- Low-level Rust graph and traversal APIs can continue to expose raw topology details for debugging and research use.

## Migration Strategy

## Phase 0: Prepare and Measure

### Goals

- Establish current behaviour and performance baselines.
- Identify user-visible surfaces that mention primal vs dual execution.
- Mark known semantic limitations explicitly.

### Tasks

- Inventory all public APIs, docs, tutorials, and plugin controls that expose primal/dual choices.
- Record baseline outputs for representative primal and dual analyses on fixture networks.
- Add regression tests for:
  - shortest path parity
  - simplest path parity where expected
  - dual-angular betweenness invariants
  - target-aggregation semantics
  - directional slope behaviour
- Create benchmark fixtures for runtime and memory before internal canonicalization.

### Exit criteria

- Baseline result snapshots and benchmark numbers are available.
- Known incompatibilities are documented before code changes begin.

## Phase 1: Internal Dual Canonicalization Behind Existing APIs

### Goals

- Keep the public API stable.
- Shift node-based execution onto dual internally where possible.
- Preserve current call signatures.

### Tasks

- Introduce an internal canonicalization layer in Python near `network_structure_from_nx` or the metrics entry points.
- If input is primal and the requested workflow is node-based, generate or reuse a dual representation internally.
- Thread topology metadata through the Rust and Python layers only as needed for correctness and debugging.
- Keep existing methods such as `centrality_shortest` and `centrality_simplest` callable with the same parameters.
- Implement dual-angular betweenness using the dual state model, not the old primal sidestep logic.

### Compatibility rule

- Existing user code should continue to run unchanged in this phase.
- Where outputs are expected to shift, log a warning only if the change is material and intentional.

### Exit criteria

- Node-based public APIs behave the same or intentionally better under the same signatures.
- Existing integration tests pass.
- New dual-internal tests pass.

## Phase 2: Introduce the New Semantics Explicitly

### Goals

- Add the future-facing API without removing the legacy one.
- Teach users to think in terms of path semantics rather than topology internals.

### Tasks

- Add new high-level parameters such as `path_cost` and `cutoff_cost`.
- Map old combinations onto the new semantics internally.
- Update docs and examples to prefer the new API shape.
- Add warnings for legacy topology-selection options where they are public.

### Suggested compatibility mapping

- Old shortest node centrality:
  - `path_cost="metric"`
  - `cutoff_cost="metric"`
- Old simplest node centrality:
  - `path_cost="angular"`
  - `cutoff_cost="metric"`
- Old explicit dual execution:
  - continue to accept, but mark as advanced/internal

### Exit criteria

- New API is fully documented.
- Old API remains functional.
- New examples and plugin controls prefer the new API.

## Phase 3: Deprecate Legacy Topology-Selection Surfaces

### Goals

- Reduce user-facing topology choices.
- Remove the need to understand primal vs dual for standard workflows.

### Tasks

- Deprecate public options, docs, and UI controls that ask users to select primal vs dual for node-based analysis.
- Keep explicit dual-conversion utilities public for advanced users.
- Emit release-note guidance with concrete before/after examples.
- If necessary, add a one-release compatibility helper that rewrites old parameters to the new ones.

### Exit criteria

- Legacy topology-selection surfaces are marked deprecated in code and docs.
- Examples no longer teach the old model by default.

## Phase 4: Remove Legacy Primal-Angular Execution Paths

### Goals

- Remove the most fragile implementation paths.
- Reduce maintenance burden and test matrix size.

### Tasks

- Delete primal-angular traversal and betweenness code paths once the deprecation window closes.
- Remove primal-specific sidestep suppression from node-based angular betweenness.
- Simplify Rust internals around shared traversal state where shortest and dual-angular can share machinery.
- Keep any remaining primal-only methods only if they still have a distinct, justified use case.

### Exit criteria

- No production node-based centrality path depends on primal-angular execution.
- Internal topology branching is substantially reduced.

## Backward Compatibility Plan

### What should remain stable

- Accepting primal `networkx` graphs from users.
- High-level method names during at least one deprecation window.
- Existing result container shapes where practical.

### What may change

- Internal traversal topology.
- Exact angular betweenness values where current behaviour is known to be theoretically weak or buggy.
- Some edge-case route choices if they depended on legacy sidestep suppression rather than proper dual semantics.

### Mitigation

- Use warnings, not silent removal.
- Publish migration examples.
- Offer a temporary `legacy_mode=True` or similar escape hatch only if needed for a short period.
- Capture old-vs-new fixture outputs and explain intentional differences in release notes.

## Testing Plan

### Core regression suites

- Primal input -> internal dual canonicalization should preserve node identity mapping.
- Shortest-path outputs should remain stable where topology conversion is not supposed to alter metric behaviour.
- Angular simplest-path outputs should match expected dual semantics.
- Angular betweenness on dual should satisfy:
  - insertion-order invariance
  - symmetry on symmetric fixtures
  - no abrupt adjacent-segment drop-offs on known counterexamples
- Slope penalties should remain directional and target-aggregation-safe.

### Differential tests

- Run old and new implementations side by side on:
  - mock graph
  - diamond graph
  - decomposed graph
  - representative larger city sample

### Performance tests

- primal input with on-the-fly dual conversion
- cached dual reuse across repeated analyses
- shortest vs angular under shared traversal machinery

## Documentation Plan

### Docs updates

- Reframe docs around path semantics rather than topology choice.
- Explain that primal input remains supported.
- Explain that node-based angular analysis is executed on a segment-state internal model.
- Keep advanced docs for explicit dual graph utilities and visualisation.

### Release-note messaging

- Describe the architecture change as an internal canonicalization and API simplification.
- Be explicit about which outputs may change and why.
- Provide before/after code snippets.

## Plugin and UI Plan

- Default UI controls to path semantics, not topology type.
- Hide advanced topology controls unless the user opts into advanced mode.
- Preserve loading of existing project settings by mapping old values to the new internal representation.

## Rollout Recommendation

Recommended release sequence:

1. Minor release:
   - add tests
   - add internal dual canonicalization
   - no API break
2. Minor release:
   - add new `path_cost` and `cutoff_cost` API
   - document preferred usage
   - start deprecation warnings
3. Minor or major release:
   - remove deprecated topology-selection paths after at least one release cycle

If output changes are expected to materially affect user analyses, treat the final removal as a major release.

## Risks

- Result changes may be interpreted as regressions even when they are theoretical corrections.
- Dual canonicalization can add runtime cost if conversion is repeated unnecessarily.
- Mapping results back to user expectations may be confusing if not documented clearly.
- Some advanced users may rely on explicit primal execution for research comparisons.

## Risk Mitigations

- Cache internal dual representations where practical.
- Provide a documented comparison mode during the migration window.
- Publish a short technical note explaining why dual-angular semantics are preferred.
- Keep low-level explicit topology tools for research workflows.

## Recommended Immediate Next Steps

1. Finish the current correctness work with dedicated dual-angular betweenness fixtures.
2. Add an internal dual-canonicalization prototype without changing the public API.
3. Draft the future API around `path_cost` and `cutoff_cost`.
4. Add release-note and docs scaffolding before deprecations begin.
5. Only then remove legacy primal-angular execution paths.

## Open Questions

- Should shortest node centrality also always canonicalize to dual, or only angular workflows at first?
- How should result mapping behave when dual nodes do not correspond one-to-one with the user’s original conceptual nodes?
- Is a temporary legacy compatibility flag worth the maintenance cost?
- Should accessibility methods migrate on the same schedule as centrality, or one release later?

