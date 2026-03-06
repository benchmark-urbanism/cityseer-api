# Dual-Network Migration Plan

## Purpose

This document proposes a low-disruption migration from the current mixed primal/dual execution model toward a dual-network canonical architecture for node-based centrality and accessibility analysis.

The intent is to simplify the internal implementation, make angular path semantics more defensible, and reduce user-facing confusion while making one explicit correctness break: angular workflows will require a dual graph and primal-angular execution will be removed immediately.

## Executive Summary

Recommended direction:

- Keep accepting primal or dual graphs at the Python boundary.
- Enforce dual-only execution for all angular workflows immediately.
- Keep metric shortest-path workflows topology-agnostic: they continue to accept primal or dual graphs.
- Drop primal-angular immediately instead of carrying it as a compatibility mode.
- Standardise all supported node-based cutoffs on metric distance or equivalent walk time.
- Preserve explicit dual-conversion utilities and guide users to convert before angular analysis.
- Use this enforcement step as the first concrete move toward a unified traversal architecture.

This is a one-shot correctness transition for angular analysis and the basis for consolidating the supported modes onto one traversal framework.

## Compatibility Contract

The intended public compatibility contract for this migration is:

- Backward compatible:
  - metric shortest-path analysis on primal graphs
  - metric shortest-path analysis on dual graphs
  - existing metric distance and equivalent walk-time cutoffs
  - existing angular workflows that already use dual graphs
  - existing segment-centrality workflows, unless changed explicitly in a separate migration
- Intentional breaking change:
  - any angular workflow called on a non-dual graph will raise with a conversion message

No other public behavioural break is intended. Internal traversal consolidation is allowed only insofar as it preserves current metric results and current dual-angular semantics where those are already valid.

## Why Change

### Current problems

- The current API exposes implementation detail: users choose primal vs dual, shortest vs simplest, and sometimes implicitly rely on topology-specific behaviour.
- Angular analysis on primal graphs is stateful in a way that ordinary node-based Dijkstra and Brandes semantics do not represent cleanly.
- Sidestep-suppression logic is compensating for a weak state model rather than being a clean property of the graph representation.
- Supporting both primal and dual execution paths multiplies testing and maintenance cost.
- Carrying multiple cutoff semantics would add complexity without helping the intended supported modes.

### Why dual should be canonical

- Dual nodes encode segment state directly.
- Angular transitions become segment-to-segment transitions with clearer semantics.
- Shortest and angular traversals can share more machinery.
- Brandes-style angular betweenness becomes much more plausible on dual graphs than on primal graphs.
- Once primal-angular is removed, the remaining supported modes can share one traversal skeleton with different route-cost functions.

## Non-Goals

- This migration does not require removing support for primal graph input.
- This migration does not require changing segment centrality to run on dual graphs.
- This migration does intentionally introduce one public correctness break: angular workflows on primal graphs will raise.
- This migration does not require exposing dual graphs to users who do not care about internal topology.
- This migration does not attempt to preserve primal-angular as a legacy fallback.

## Design Principles

1. Preserve user intent, not existing internal structure.
2. Avoid silent behavioural changes where rankings or reachability may materially differ.
3. Prefer explicit failure over silently running a semantically weak angular workflow on a primal graph.
4. Keep non-angular entry points working during the migration window.
5. Test new enforcement and unchanged metric execution side by side.
6. Unify around one traversal engine where route cost and cutoff cost are explicit, separate accumulators.

## Target Architecture

### Canonical internal model

- Angular node-based centrality and accessibility computations use a dual graph.
- Metric node-based computations continue to support either primal or dual graphs.
- The supported node-based modes after migration are:
  - metric on primal
  - metric on dual
  - angular on dual
- Primal-angular is not supported.
- All supported node-based modes use metric distance or equivalent walk time as the reachability cutoff accumulator.
- Routing cost and cutoff cost are treated as separate internal channels.

### Public API model

Users should choose:

- `path_cost`
  - `metric`
  - `angular`
- `cutoff_cost`
  - `metric`
  - equivalent walk time derived from metric cost

Users should not need to guess whether angular analysis is valid on a primal graph. The library should reject that case directly.

### Things that remain separate

- Segment centrality stays on its own execution path.
- Dual graph construction utilities remain public for users who explicitly want topology conversion.
- Low-level Rust graph and traversal APIs can continue to expose raw topology details for debugging and research use.
- The traversal engine may be shared even if result aggregation remains mode-specific.

## Migration Strategy

## Recommended Transition

The recommended implementation is:

- if a workflow is angular, require `is_dual == true`
- if a workflow is metric, allow either primal or dual
- treat cutoff accumulation as metric distance or equivalent walk time in every supported mode
- consolidate internals around one traversal skeleton with mode-specific route-cost increments
- do not silently convert primal to dual inside Rust in the first step
- provide a precise error message telling users to call `graphs.nx_to_dual(...)`

This yields the least ambiguous semantics with the smallest implementation surface while keeping the long-term architecture coherent.

## Phase 0: Prepare and Measure

### Goals

- Establish current behaviour and performance baselines.
- Identify every public angular entry point.
- Mark primal-angular execution as unsupported going forward.

### Tasks

- Inventory all public APIs, docs, tutorials, and plugin controls that expose primal/dual choices.
- Inventory all public APIs, docs, tutorials, and plugin controls that expose angular choices.
- Record baseline outputs for representative primal and dual analyses on fixture networks.
- Add regression tests for:
  - shortest path parity
  - simplest path parity on dual graphs only
  - dual-angular betweenness invariants
  - target-aggregation semantics
  - shared cutoff semantics across shortest and angular
  - directional slope behaviour
- Add failure tests for:
  - angular centrality on primal raises
  - angular Dijkstra on primal raises
  - angular accessibility on primal raises
- Create benchmark fixtures for runtime and memory before internal canonicalization.

### Exit criteria

- Baseline result snapshots and benchmark numbers are available.
- Known incompatibilities are documented before code changes begin.
- All angular entry points to be guarded are enumerated.

## Phase 1: Enforce Dual-Only Angular Workflows

### Goals

- Make angular semantics explicit and defensible immediately.
- Preserve current method names where possible.
- Keep metric workflows unchanged.
- Collapse the supported execution matrix to the three intended cases only.

### Tasks

- Preserve topology metadata on `NetworkStructure`, ideally as `is_dual`.
- Add validation helpers in Rust and/or Python for angular workflows:
  - `centrality_simplest`
  - `dijkstra_tree_simplest`
  - angular accessibility and land-use methods
  - plugin-driven angular workflows
- Raise a clear error when angular workflows are called on a primal graph.
- Keep existing methods such as `centrality_shortest` callable on either topology.
- Ensure shortest-path Dijkstra, shortest centrality, and segment centrality are not accidentally restricted.
- Remove or hard-disable any remaining primal-angular internal code paths instead of leaving them partially reachable.
- Implement or finish dual-angular betweenness using the dual state model, not the old primal sidestep logic.
- Update docs and examples so every angular workflow shows explicit dual conversion.

### Compatibility rule

- Existing metric user code should continue to run unchanged.
- Existing angular user code on dual graphs should continue to run.
- Existing angular user code on primal graphs should fail fast with a migration message.
- No internal fallback should silently run primal-angular.

### Exit criteria

- Every angular public entry point rejects primal graphs.
- Every metric public entry point still accepts primal and dual graphs.
- Existing integration tests pass after expected angular-primal fixture updates.
- New enforcement tests pass.

## Phase 2: Unify Semantics Around Cost Model

### Goals

- Reduce topology talk in the API surface.
- Teach users to think in terms of cost semantics rather than topology internals.
- Consolidate implementation around one traversal framework.

### Tasks

- Add new high-level parameters such as `path_cost` and `cutoff_cost`.
- Map old combinations onto the new semantics internally.
- Refactor internal traversal state so all supported modes share:
  - one frontier/heap implementation
  - one cutoff accumulator
  - one route-cost accumulator
  - one predecessor/sigma framework where valid
- Update docs and examples to prefer the new API shape.
- Add warnings for legacy topology-selection options where they are public.
- Keep the dual-only angular guard in place.

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
- Old metric API remains functional.
- Old angular API remains functional only on dual graphs.
- New examples and plugin controls prefer the new API.
- Internals no longer rely on a separate primal-angular traversal model.

## Phase 3: Internal Traversal Consolidation

### Goals

- Implement one shared traversal engine for the three supported modes.
- Reduce duplicate shortest/simplest code without changing public behaviour.

### Tasks

- Introduce a shared internal state model for supported modes.
- Separate:
  - `route_cost`
  - `cutoff_cost`
  - predecessor tracking
  - `sigma` / path counts where Brandes applies
- Parameterise edge relaxation by mode:
  - shortest: metric route increment
  - angular on dual: angular route increment
- Keep cutoff relaxation metric/time-based in all modes.
- Merge duplicated Dijkstra logic where practical after behaviour is covered by tests.
- Keep explicit dual-conversion utilities public for advanced users.
- Emit release-note guidance with concrete before/after examples.

### Exit criteria

- All three supported modes run through the same traversal skeleton.
- No remaining production angular code path depends on primal execution.
- Cutoff semantics are consistent across shortest and angular workflows.

## Backward Compatibility Plan

### What should remain stable

- Accepting primal `networkx` graphs from users.
- High-level metric method names during at least one deprecation window.
- Existing result container shapes where practical.
- Metric analysis behaviour on primal and dual graphs.
- Existing dual-angular workflows.

### What may change

- Angular workflows on primal graphs will stop running and raise instead.
- Exact angular betweenness values where current behaviour is known to be theoretically weak or buggy.
- Some edge-case dual-angular route choices if they depended on legacy sidestep suppression rather than proper dual semantics.
- Internal traversal implementation may change even where public results stay stable.

### Intended break boundary

- The only intentional public break in this migration is angular-on-primal rejection.
- Any additional observed break in metric workflows should be treated as a regression, not as accepted migration fallout.

### Mitigation

- Use warnings, not silent removal.
- Use direct errors for invalid angular-primal calls.
- Publish migration examples.
- Do not offer a hidden fallback to primal-angular execution.
- Capture old-vs-new fixture outputs and explain intentional differences in release notes.

## Testing Plan

### Core regression suites

- Shortest-path outputs should remain stable where topology conversion is not supposed to alter metric behaviour.
- Angular simplest-path outputs should match expected dual semantics.
- Shortest and angular should agree on cutoff reachability where the metric/time cutoff is the same and route choice does not change reach.
- Angular betweenness on dual should satisfy:
  - insertion-order invariance
  - symmetry on symmetric fixtures
  - no abrupt adjacent-segment drop-offs on known counterexamples
- Slope penalties should remain directional and target-aggregation-safe.
- Angular-on-primal should raise consistently across all guarded public entry points.
- Shared traversal refactors should not change shortest results on primal or dual.

### Differential tests

- Run metric workflows before and after the guard on:
  - mock graph
  - diamond graph
  - decomposed graph
  - representative larger city sample
- Run angular workflows on dual fixtures and verify unchanged behaviour where intended.

### Performance tests

- shortest on primal
- shortest on dual
- angular on dual
- explicit Python-side dual conversion cost, if that becomes part of the workflow
- shared traversal engine overhead vs current split implementation

## Documentation Plan

### Docs updates

- Reframe docs around path semantics rather than topology choice.
- Explain that primal input remains supported for metric workflows.
- Explain clearly that angular workflows require dual graphs.
- Explain that node-based angular analysis is executed on a segment-state internal model.
- Explain that all supported node-based modes use metric/time reachability cutoffs.
- Keep advanced docs for explicit dual graph utilities and visualisation.

### Release-note messaging

- Describe the architecture change as a correctness guard plus architecture simplification.
- Be explicit that angular-on-primal now raises.
- Be explicit about which outputs may change and why.
- Explain that cutoff semantics are now consistently metric/time-based across supported modes.
- Be explicit that metric workflows remain backward compatible on both primal and dual graphs.
- Provide before/after code snippets.

## Plugin and UI Plan

- Default UI controls to path semantics, not topology type.
- Hide advanced topology controls unless the user opts into advanced mode.
- When an old project requests angular analysis on a primal graph, show a conversion error with remediation guidance.

## Rollout Recommendation

Recommended release sequence:

1. Minor release:
   - add tests
   - add dual-only angular validation
   - remove primal-angular execution paths
   - document the correction clearly
2. Minor release:
   - add new `path_cost` and `cutoff_cost` API
   - refactor to a shared traversal skeleton
   - document preferred usage
   - reduce public emphasis on topology choice
3. Minor or major release:
   - simplify remaining topology-selection paths after observation

If the team treats angular-on-primal raising as a semantic correction rather than a mere feature change, document it prominently even if shipped in a minor release.

## Risks

- Result changes may be interpreted as regressions even when they are theoretical corrections.
- Users may experience the angular-primal failure as a regression if release notes are weak.
- Mapping results back to user expectations may be confusing if not documented clearly.
- Some advanced users may rely on explicit primal execution for research comparisons.
- Refactoring to a shared traversal engine may accidentally perturb shortest-path behaviour if tests are incomplete.

## Risk Mitigations

- Provide a documented conversion recipe in every angular example.
- Keep explicit dual utilities easy to discover.
- Publish a short technical note explaining why dual-angular semantics are preferred.
- Keep low-level explicit topology tools for research workflows.
- Land the traversal refactor only after the guard and tests are in place.

## Recommended Immediate Next Steps

1. Preserve or add `is_dual` metadata through Python and Rust.
2. Add angular topology validators at all public entry points.
3. Add raising tests for angular-on-primal.
4. Finish the current correctness work with dedicated dual-angular betweenness fixtures.
5. Remove primal-angular internals rather than maintaining them as dead compatibility branches.
6. Update docs, plugin messaging, and examples to require explicit `nx_to_dual(...)` before angular analysis.
7. Refactor supported modes onto one traversal skeleton with separate route and cutoff accumulators.

## Implementation Checklist

### Architecture constraints

- Supported node-based routing modes after migration:
  - metric on primal
  - metric on dual
  - angular on dual
- Unsupported immediately after migration:
  - angular on primal
- Cutoff semantics in all supported modes:
  - metric distance
  - or equivalent walk time derived from metric distance
- No mode should use angular cutoff thresholds in this migration.

### Core code changes

- Add or preserve `is_dual` on ingested `NetworkStructure`.
- Add a shared validation helper for angular workflows.
- Call that helper from:
  - `dijkstra_tree_simplest`
  - `centrality_simplest`
  - angular accessibility and land-use entry points
  - any plugin algorithm using angular routing
- Ensure metric shortest-path entry points do not call the angular guard.
- Remove or delete primal-angular-only internal branches, helpers, and comments.
- Remove any misleading docs or comments implying primal-angular remains supported.

### Shared traversal refactor

- Define the shared per-node state required by all supported modes:
  - `visited`
  - `discovered`
  - `route_cost`
  - `cutoff_cost` or `agg_seconds`
  - predecessor pointer or predecessor list
  - `sigma` for Brandes-capable modes
  - any mode-specific bearing or segment state needed for dual-angular only
- Define a mode selector with only the supported values:
  - shortest
  - simplest_dual
- Refactor relaxation logic so it computes:
  - route increment
  - cutoff increment
  - predecessor update policy
- Keep cutoff increment metric/time-based in both modes.
- Ensure the heap ordering key is always route cost, not cutoff cost.
- Ensure threshold filtering is always applied on cutoff cost, not route cost.
- Reuse the same traversal skeleton in:
  - shortest Dijkstra tree
  - simplest Dijkstra tree on dual
  - shortest centrality traversal
  - simplest centrality traversal on dual
- Keep segment traversal separate unless the shared skeleton naturally covers it without extra complexity.

### Betweenness-specific work

- Keep Brandes for shortest on primal and dual.
- Implement Brandes for angular only on dual.
- Remove primal sidestep suppression from dual-angular Brandes paths.
- Verify predecessor semantics and sigma counting on dual-angular fixtures.
- Confirm target aggregation remains unchanged.

### Public API enforcement steps

- Add the guard at the highest public Python entry points where angular workflows are requested.
- Add the same guard in Rust public angular entry points as a defensive backstop.
- Ensure the guard fires before heavy computation starts.
- Ensure the guard message is identical or near-identical across entry points.

### Error-message requirements

- State that angular analysis requires a dual graph.
- Say how to convert:
  - `G_dual = graphs.nx_to_dual(G)`
  - then ingest the dual graph
- Keep the message short and deterministic.

### Tests

- Add unit tests that angular centrality on primal raises.
- Add unit tests that angular Dijkstra on primal raises.
- Add unit tests that angular accessibility and land-use methods on primal raise, if applicable.
- Add unit tests that angular workflows on dual still succeed.
- Add regression tests for dual-angular betweenness where the old sidestep logic failed.
- Add regression tests that shortest on primal still matches existing expectations.
- Add regression tests that shortest on dual still matches existing expectations.
- Add regression tests for shared cutoff behaviour:
  - angular route choice differs from shortest
  - cutoff filtering remains metric/time-based
- Keep metric primal and metric dual tests unchanged and passing.

### Detailed implementation sequence

1. Add or restore `is_dual` metadata on Python graph ingestion and Rust `NetworkStructure`.
2. Identify every public angular entry point in Python, Rust, and plugin code.
3. Add a shared angular topology validator.
4. Add failing tests for angular-on-primal at every public entry point.
5. Make those tests pass with clear, user-facing errors.
6. Identify all remaining primal-angular internal helpers and code paths.
7. Remove or isolate those code paths so they can no longer be called.
8. Add or update dual-angular regression fixtures before refactoring traversal internals.
9. Define the shared traversal state and mode enum.
10. Refactor one lowest-risk path first:
    - shortest Dijkstra tree
11. Refactor the dual-angular Dijkstra tree to use the same skeleton.
12. Refactor shortest centrality to use the shared traversal.
13. Refactor simplest centrality on dual to use the shared traversal.
14. Re-run shortest, simplest, and betweenness regression suites after each refactor step.
15. Only after traversal parity is stable, simplify or delete obsolete duplicated helpers.
16. Update docs, examples, and plugin messaging.
17. Add release-note text and migration examples.
18. Run full targeted test suite and benchmark comparison.

### Docs and examples

- Update centrality docs to say angular requires dual.
- Update intro docs and README examples where angular analysis appears.
- Update plugin help text and parameter descriptions.
- Add a minimal migration example from primal to dual.

### Release communications

- Add a release-note item titled similar to:
  - `Angular analysis now requires dual graphs`
- Explain this as a correctness safeguard.
- Explain that primal-angular has been removed.
- Explain that supported modes now share metric/time reachability cutoffs.
- State explicitly that this is the only intended breaking change.
- Link to the migration example.

## Open Questions

- Should shortest node centrality also always canonicalize to dual later, or should metric workflows remain topology-agnostic permanently?
- Should Python eventually auto-convert primal to dual for angular convenience, or should explicit dual conversion remain the public contract?
- Should accessibility methods migrate on the same schedule as centrality, or one release later?
