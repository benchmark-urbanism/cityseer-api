use crate::common;

use crate::common::MetricResult;
use crate::common::WALKING_SPEED;
use crate::graph::{NetworkStructure, NodeVisit};
use numpy::PyArray1;
use petgraph::prelude::*;
use petgraph::stable_graph::NodeIndex;
use petgraph::visit::IntoEdgeReferences;
use petgraph::Direction;
use pyo3::exceptions;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering as AtomicOrdering;

const ANGULAR_ROUTE_TIE_BREAK_FACTOR: f32 = 1e-6;
/// Minimum float-comparison tolerance for both shortest-path and angular routing.
/// IMPORTANT: tolerance must always be >= this value to avoid missed ties from
/// floating-point noise. All callers must default to at least this value.
const TIE_EPSILON: f32 = 1e-4;
/// Maximum tolerance percentage above which we warn the user.
const TOLERANCE_WARN_PCT: f32 = 2.0;

/// Validate and convert tolerance from user-facing percentage to a decimal multiplier.
/// E.g. 1.0% → 0.01 internally, clamped to at least TIE_EPSILON.
fn validate_tolerance(tolerance: Option<f32>) -> PyResult<f32> {
    let pct = tolerance.unwrap_or(0.0);
    if pct < 0.0 || pct > 100.0 {
        return Err(exceptions::PyValueError::new_err(format!(
            "Tolerance must be between 0 and 100 (percent), got {pct}"
        )));
    }
    if pct > TOLERANCE_WARN_PCT {
        log::warn!(
            "Tolerance {pct:.1}% is high — values above {TOLERANCE_WARN_PCT}% increasingly \
             diffuse route concentration, especially at larger distance thresholds."
        );
    }
    Ok((pct / 100.0).max(TIE_EPSILON))
}

/// Sparse origin-destination weight matrix for OD-weighted centrality.
///
/// Stores per-pair trip weights in a nested HashMap for O(1) lookup.
/// Constructed once and passed to centrality functions; can be reused across calls.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct OdMatrix {
    map: HashMap<usize, HashMap<usize, f32>>,
}

#[pymethods]
impl OdMatrix {
    #[new]
    #[pyo3(signature = (origins, destinations, weights))]
    fn new(origins: Vec<usize>, destinations: Vec<usize>, weights: Vec<f32>) -> PyResult<Self> {
        if origins.len() != destinations.len() || origins.len() != weights.len() {
            return Err(exceptions::PyValueError::new_err(format!(
                "origins ({}), destinations ({}), and weights ({}) must have equal length",
                origins.len(),
                destinations.len(),
                weights.len()
            )));
        }
        let mut map: HashMap<usize, HashMap<usize, f32>> = HashMap::new();
        for i in 0..origins.len() {
            map.entry(origins[i])
                .or_default()
                .insert(destinations[i], weights[i]);
        }
        Ok(OdMatrix { map })
    }

    /// Number of non-zero OD pairs.
    fn len(&self) -> usize {
        self.map.values().map(|d| d.len()).sum()
    }

    /// Number of unique origin nodes.
    fn n_origins(&self) -> usize {
        self.map.len()
    }
}

// =========================================================================
// Generic centrality result type — replaces all specialised result structs
// =========================================================================

#[pyclass]
pub struct CentralityResult {
    #[pyo3(get)]
    distances: Vec<u32>,
    #[pyo3(get)]
    node_keys_py: Vec<Py<PyAny>>,
    #[pyo3(get)]
    node_indices: Vec<usize>,

    /// Named closeness metrics: Vec of (name, MetricResult).
    closeness_metrics: Vec<(String, MetricResult)>,
    /// Named betweenness metrics: Vec of (name, MetricResult).
    betweenness_metrics: Vec<(String, MetricResult)>,
    /// Optional cycles metric (circuit rank).
    cycles_metric: Option<MetricResult>,

    #[pyo3(get)]
    pub reachability_totals: Vec<u32>,
    #[pyo3(get)]
    pub sampled_source_count: u32,
}

impl CentralityResult {
    pub fn new(
        distances: Vec<u32>,
        node_keys_py: Vec<Py<PyAny>>,
        node_indices: Vec<usize>,
        closeness_names: &[String],
        betweenness_names: &[String],
        compute_cycles: bool,
        capacity: usize,
        init_val: f32,
    ) -> Self {
        let closeness_metrics = closeness_names
            .iter()
            .map(|name| (name.clone(), MetricResult::new(&distances, capacity, init_val)))
            .collect();
        let betweenness_metrics = betweenness_names
            .iter()
            .map(|name| (name.clone(), MetricResult::new(&distances, capacity, init_val)))
            .collect();
        let cycles_metric = if compute_cycles {
            Some(MetricResult::new(&distances, capacity, init_val))
        } else {
            None
        };
        CentralityResult {
            distances,
            node_keys_py,
            node_indices,
            closeness_metrics,
            betweenness_metrics,
            cycles_metric,
            reachability_totals: Vec::new(),
            sampled_source_count: 0,
        }
    }
}

#[pymethods]
impl CentralityResult {
    /// Returns all computed metrics as a flat dict: {name: {distance: array}}.
    /// Combines closeness, betweenness, and cycles (if computed) into one namespace.
    #[getter]
    pub fn metrics(&self) -> HashMap<String, HashMap<u32, Py<PyArray1<f64>>>> {
        let mut result = HashMap::new();
        for (name, metric) in &self.closeness_metrics {
            result.insert(name.clone(), metric.load_compact(&self.node_indices));
        }
        for (name, metric) in &self.betweenness_metrics {
            result.insert(name.clone(), metric.load_compact(&self.node_indices));
        }
        if let Some(ref cycles) = self.cycles_metric {
            result.insert("cycles".to_string(), cycles.load_compact(&self.node_indices));
        }
        result
    }
}

// NodeDistance for heap
struct NodeDistance {
    node_idx: usize,
    metric: f32,
}

impl Ord for NodeDistance {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed for min-heap: smaller metric = higher priority.
        // total_cmp provides a total ordering over all f32 values including NaN.
        other.metric.total_cmp(&self.metric)
    }
}

impl PartialOrd for NodeDistance {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for NodeDistance {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for NodeDistance {}

/// Node state for Brandes-style Dijkstra with multi-predecessor tracking.
///
/// Unlike `NodeVisit` which stores a single predecessor, this tracks ALL predecessors
/// on shortest paths and counts the number of shortest paths (sigma) from the source.
/// Used internally for Brandes betweenness centrality with multi-predecessor shortest-path tracking.
#[derive(Clone)]
struct BrandesTraversalState {
    visited: bool,
    preds: SmallVec<[usize; 2]>,
    sigma: f64,
    node_idx: usize,
    route_cost: f32,
    agg_seconds: f32,
}

impl BrandesTraversalState {
    fn new(node_idx: usize) -> Self {
        Self {
            visited: false,
            preds: SmallVec::new(),
            sigma: 0.0,
            node_idx,
            route_cost: f32::INFINITY,
            agg_seconds: f32::INFINITY,
        }
    }
}

type AngularEndpointSlots = Vec<SmallVec<[String; 2]>>;

struct BrandesTraversal {
    visited_state_indices: Vec<usize>,
    reached_node_indices: Vec<usize>,
    state: Vec<BrandesTraversalState>,
    best_route_cost: Vec<f32>,
    best_agg_seconds: Vec<f32>,
}

#[derive(Clone, Copy)]
struct AngularTreeState {
    visited: bool,
    pred_state_idx: Option<usize>,
    route_metric: f32,
    simpl_dist: f32,
    agg_seconds: f32,
}

impl AngularTreeState {
    fn new() -> Self {
        Self {
            visited: false,
            pred_state_idx: None,
            route_metric: f32::INFINITY,
            simpl_dist: f32::INFINITY,
            agg_seconds: f32::INFINITY,
        }
    }
}

struct SourceSamplingPlan {
    sample_probability: Option<f32>,
    sampling_weights: Option<Vec<f32>>,
    sample_randoms: Vec<f32>,
    sources: Vec<usize>,
    node_live: Vec<bool>,
}

impl SourceSamplingPlan {
    /// Returns true when Bernoulli sampling with IPW is active.
    #[inline]
    fn is_sampling(&self) -> bool {
        self.sample_probability.is_some()
    }
}

impl NetworkStructure {
    #[inline]
    fn reached_node_indices(best_route_cost: &[f32]) -> Vec<usize> {
        best_route_cost
            .iter()
            .enumerate()
            .filter_map(|(node_idx, &cost)| cost.is_finite().then_some(node_idx))
            .collect()
    }

    /// Compute circuit rank (edges - nodes + 1) for each distance threshold
    /// using only the reachable subgraph from the traversal. O(reachable edges).
    fn circuit_ranks_from_traversal(
        &self,
        traversal: &BrandesTraversal,
        distances: &[u32],
    ) -> Vec<f32> {
        let d_n = distances.len();
        let mut node_counts = vec![0usize; d_n];
        for &node_idx in &traversal.reached_node_indices {
            let cost = traversal.best_route_cost[node_idx];
            for i in 0..d_n {
                if cost <= distances[i] as f32 {
                    node_counts[i] += 1;
                }
            }
        }
        let mut edge_counts = vec![0usize; d_n];
        let mut seen = HashSet::new();
        for &node_idx in &traversal.reached_node_indices {
            let node_index = NodeIndex::new(node_idx);
            for edge_ref in self.graph.edges_directed(node_index, Direction::Outgoing) {
                let nb_idx = edge_ref.target().index();
                if nb_idx == node_idx {
                    continue;
                }
                if !traversal.best_route_cost[nb_idx].is_finite() {
                    continue;
                }
                let edge_key = (
                    node_idx.min(nb_idx),
                    node_idx.max(nb_idx),
                    edge_ref.weight().edge_idx,
                );
                if !seen.insert(edge_key) {
                    continue;
                }
                let edge_cost =
                    traversal.best_route_cost[node_idx].max(traversal.best_route_cost[nb_idx]);
                for i in 0..d_n {
                    if edge_cost <= distances[i] as f32 {
                        edge_counts[i] += 1;
                    }
                }
            }
        }
        (0..d_n)
            .map(|i| {
                if node_counts[i] == 0 {
                    0.0
                } else {
                    (edge_counts[i] as isize - node_counts[i] as isize + 1).max(0) as f32
                }
            })
            .collect()
    }

    #[inline]
    fn angular_state_idx(node_idx: usize, endpoint_slot: usize) -> usize {
        (node_idx * 2) + endpoint_slot
    }

    fn dual_node_endpoint_slots(&self) -> PyResult<AngularEndpointSlots> {
        let node_bound = self.node_bound();
        let mut endpoint_slots = vec![SmallVec::<[String; 2]>::new(); node_bound];

        for edge_ref in self.graph.edge_references() {
            let shared_key = edge_ref
                .weight()
                .shared_primal_node_key
                .as_deref()
                .ok_or_else(|| {
                    exceptions::PyValueError::new_err(format!(
                        "dual edge {} -> {} is missing shared_primal_node_key metadata",
                        edge_ref.source().index(),
                        edge_ref.target().index()
                    ))
                })?;

            for node_idx in [edge_ref.source().index(), edge_ref.target().index()] {
                let slots = &mut endpoint_slots[node_idx];
                if slots.iter().any(|slot_key| slot_key == shared_key) {
                    continue;
                }
                if slots.len() >= 2 {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "dual node {} references more than two primal endpoints",
                        node_idx
                    )));
                }
                slots.push(shared_key.to_string());
            }
        }

        Ok(endpoint_slots)
    }

    /// Two-phase Brandes Dijkstra for angular (simplest-path) centrality with tolerance.
    ///
    /// Phase 1: Exact Dijkstra (TIE_EPSILON only) on the doubled state space to compute
    /// final settled angular costs, visit order, and exact predecessors.
    ///
    /// Phase 2: Rebuild predecessors and sigma using the user's tolerance against final
    /// (immutable) angular costs. Same correctness guarantee as the shortest-path variant:
    /// processing in settled order ensures all predecessors are finalised before their
    /// successors.
    fn dijkstra_brandes_angular(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        tolerance: f32,
        endpoint_slots: &[SmallVec<[String; 2]>],
        upstream: bool,
    ) -> BrandesTraversal {
        assert!(
            tolerance >= TIE_EPSILON,
            "Tolerance must be >= TIE_EPSILON to avoid float-comparison bugs"
        );
        let direction = if upstream {
            Direction::Incoming
        } else {
            Direction::Outgoing
        };
        let node_count = self.node_bound();
        let state_count = node_count * 2;
        let mut states = (0..state_count)
            .map(|state_idx| BrandesTraversalState::new(state_idx / 2))
            .collect::<Vec<_>>();
        let mut visited_state_indices = Vec::new();
        let mut best_route_cost = vec![f32::INFINITY; node_count];
        let mut best_agg_seconds = vec![f32::INFINITY; node_count];

        // Phase 1: Exact Dijkstra — settled angular costs and predecessors (TIE_EPSILON).
        let mut active = BinaryHeap::new();
        best_route_cost[src_idx] = 0.0;
        best_agg_seconds[src_idx] = 0.0;
        for slot in 0..2 {
            let src_state_idx = Self::angular_state_idx(src_idx, slot);
            states[src_state_idx].sigma = 1.0;
            states[src_state_idx].route_cost = 0.0;
            states[src_state_idx].agg_seconds = 0.0;
            active.push(NodeDistance {
                node_idx: src_state_idx,
                metric: 0.0,
            });
        }

        while let Some(NodeDistance {
            node_idx: state_idx,
            ..
        }) = active.pop()
        {
            if states[state_idx].visited {
                continue;
            }
            states[state_idx].visited = true;
            visited_state_indices.push(state_idx);

            let current_node_idx = states[state_idx].node_idx;
            let current_entry_slot = state_idx % 2;
            let current_node_index = NodeIndex::new(current_node_idx);

            for edge_ref in self.graph.edges_directed(current_node_index, direction) {
                let next_node_idx = if upstream {
                    edge_ref.source().index()
                } else {
                    edge_ref.target().index()
                };
                let edge_payload = edge_ref.weight();
                let shared_key = edge_payload
                    .shared_primal_node_key
                    .as_deref()
                    .expect("validated dual edge is missing shared_primal_node_key metadata");
                let current_shared_slot = endpoint_slots[current_node_idx]
                    .iter()
                    .position(|slot_key| slot_key == shared_key)
                    .expect("validated shared_primal_node_key missing from source dual node");
                // Downstream: edge must connect at exit slot (opposite of entry).
                // Upstream: edge must connect at entry slot (same as entry).
                let slot_mismatch = if upstream {
                    current_shared_slot != current_entry_slot
                } else {
                    current_shared_slot != 1 - current_entry_slot
                };
                if slot_mismatch {
                    continue;
                }
                let next_shared_slot = endpoint_slots[next_node_idx]
                    .iter()
                    .position(|slot_key| slot_key == shared_key)
                    .expect("validated shared_primal_node_key missing from target dual node");
                // Downstream: neighbor entered from shared slot.
                // Upstream: neighbor entered from opposite of shared slot.
                let next_state_idx = if upstream {
                    Self::angular_state_idx(next_node_idx, 1 - next_shared_slot)
                } else {
                    Self::angular_state_idx(next_node_idx, next_shared_slot)
                };

                let (from, to) = if upstream {
                    (next_node_idx, current_node_idx)
                } else {
                    (current_node_idx, next_node_idx)
                };
                let edge_seconds =
                    self.edge_travel_seconds(from, to, edge_payload, speed_m_s, false);
                let candidate_seconds = states[state_idx].agg_seconds + edge_seconds;
                if candidate_seconds > max_seconds as f32 {
                    continue;
                }
                if states[next_state_idx].visited {
                    continue;
                }
                let candidate_route = states[state_idx].route_cost
                    + edge_payload.angle_sum
                    + (ANGULAR_ROUTE_TIE_BREAK_FACTOR * edge_payload.length);

                let cur_cost = states[next_state_idx].route_cost;
                let improved = candidate_route < cur_cost;
                let tied = candidate_route <= cur_cost * (1.0 + TIE_EPSILON);

                if improved {
                    if candidate_route < cur_cost * (1.0 - TIE_EPSILON) {
                        states[next_state_idx].preds.clear();
                        states[next_state_idx].sigma = states[state_idx].sigma;
                    } else {
                        states[next_state_idx].sigma += states[state_idx].sigma;
                    }
                    states[next_state_idx].route_cost = candidate_route;
                    states[next_state_idx].agg_seconds = candidate_seconds;
                    states[next_state_idx].preds.push(state_idx);
                    active.push(NodeDistance {
                        node_idx: next_state_idx,
                        metric: candidate_route,
                    });
                } else if tied && !states[next_state_idx].preds.contains(&state_idx) {
                    if candidate_seconds < states[next_state_idx].agg_seconds {
                        states[next_state_idx].agg_seconds = candidate_seconds;
                    }
                    states[next_state_idx].preds.push(state_idx);
                    states[next_state_idx].sigma += states[state_idx].sigma;
                }

                let next_node_idx = states[next_state_idx].node_idx;
                let best_cost = best_route_cost[next_node_idx];
                if candidate_route < best_cost * (1.0 - TIE_EPSILON) {
                    best_route_cost[next_node_idx] = candidate_route;
                    best_agg_seconds[next_node_idx] = candidate_seconds;
                } else if candidate_route <= best_cost * (1.0 + TIE_EPSILON) {
                    best_agg_seconds[next_node_idx] =
                        best_agg_seconds[next_node_idx].min(candidate_seconds);
                }
            }
        }

        // Phase 2: Rebuild predecessors and sigma using user tolerance against final costs.
        // Re-iterates edges using settled costs (immutable after Phase 1). Processing in
        // settled order guarantees sigma accumulates correctly (all predecessors finalised first).
        if tolerance > TIE_EPSILON {
            let mut visit_pos = vec![usize::MAX; state_count];
            for (pos, &idx) in visited_state_indices.iter().enumerate() {
                visit_pos[idx] = pos;
            }
            // Clear Phase 1 predecessors and sigma (keep distances).
            for &idx in &visited_state_indices {
                states[idx].preds.clear();
                states[idx].sigma = 0.0;
            }
            for slot in 0..2 {
                let src_state_idx = Self::angular_state_idx(src_idx, slot);
                states[src_state_idx].sigma = 1.0;
            }
            for (pos, &u_state_idx) in visited_state_indices.iter().enumerate() {
                let u_node_idx = states[u_state_idx].node_idx;
                let u_entry_slot = u_state_idx % 2;
                let u_node_index = NodeIndex::new(u_node_idx);
                for edge_ref in self.graph.edges_directed(u_node_index, direction) {
                    let next_node_idx = if upstream {
                        edge_ref.source().index()
                    } else {
                        edge_ref.target().index()
                    };
                    let edge_payload = edge_ref.weight();
                    let shared_key = edge_payload
                        .shared_primal_node_key
                        .as_deref()
                        .expect("validated dual edge is missing shared_primal_node_key metadata");
                    let u_shared_slot = endpoint_slots[u_node_idx]
                        .iter()
                        .position(|slot_key| slot_key == shared_key)
                        .expect("validated shared_primal_node_key missing from source dual node");
                    let slot_mismatch = if upstream {
                        u_shared_slot != u_entry_slot
                    } else {
                        u_shared_slot != 1 - u_entry_slot
                    };
                    if slot_mismatch {
                        continue;
                    }
                    let next_shared_slot = endpoint_slots[next_node_idx]
                        .iter()
                        .position(|slot_key| slot_key == shared_key)
                        .expect("validated shared_primal_node_key missing from target dual node");
                    let v_state_idx = if upstream {
                        Self::angular_state_idx(next_node_idx, 1 - next_shared_slot)
                    } else {
                        Self::angular_state_idx(next_node_idx, next_shared_slot)
                    };
                    // Only consider successors settled after U.
                    if visit_pos[v_state_idx] <= pos {
                        continue;
                    }
                    let candidate_route = states[u_state_idx].route_cost
                        + edge_payload.angle_sum
                        + (ANGULAR_ROUTE_TIE_BREAK_FACTOR * edge_payload.length);
                    if candidate_route <= states[v_state_idx].route_cost * (1.0 + tolerance)
                        && !states[v_state_idx].preds.contains(&u_state_idx)
                    {
                        states[v_state_idx].preds.push(u_state_idx);
                        states[v_state_idx].sigma += states[u_state_idx].sigma;
                    }
                }
            }
        }

        let reached_node_indices = Self::reached_node_indices(&best_route_cost);

        BrandesTraversal {
            visited_state_indices,
            reached_node_indices,
            state: states,
            best_route_cost,
            best_agg_seconds,
        }
    }

    fn sorted_brandes_state_indices(traversal: &BrandesTraversal) -> Vec<usize> {
        let mut sorted_state_indices: Vec<usize> = traversal
            .visited_state_indices
            .iter()
            .filter(|&&state_idx| traversal.state[state_idx].sigma > 0.0)
            .copied()
            .collect();
        sorted_state_indices.sort_by(|a, b| {
            traversal.state[*b]
                .route_cost
                .partial_cmp(&traversal.state[*a].route_cost)
                .unwrap_or(Ordering::Equal)
        });
        sorted_state_indices
    }

    fn best_angular_target_states(
        traversal: &BrandesTraversal,
        node_idx: usize,
        sec_threshold: f32,
        tolerance: f32,
    ) -> SmallVec<[usize; 2]> {
        let mut best_state_indices = SmallVec::<[usize; 2]>::new();
        let best_route_cost = traversal.best_route_cost[node_idx];
        let best_agg_seconds = traversal.best_agg_seconds[node_idx];
        if !best_route_cost.is_finite()
            || !best_agg_seconds.is_finite()
            || best_agg_seconds > sec_threshold
        {
            return best_state_indices;
        }

        for slot in 0..2 {
            let state_idx = Self::angular_state_idx(node_idx, slot);
            let state = &traversal.state[state_idx];
            if state.sigma == 0.0 || state.agg_seconds > sec_threshold {
                continue;
            }
            if state.route_cost <= best_route_cost * (1.0 + tolerance) {
                best_state_indices.push(state_idx);
            }
        }

        best_state_indices
    }

    /// Brandes backpropagation with N independent channels.
    /// Each channel has its own target_seed and delta accumulator.
    /// `on_credit` receives the node index and a slice of N credit values.
    fn brandes_backprop_multi<FInclude, FCredit>(
        traversal: &BrandesTraversal,
        sorted_state_indices: &[usize],
        src_node_idx: usize,
        target_seeds: &[&[f64]],
        include_state: FInclude,
        mut on_credit: FCredit,
    ) where
        FInclude: Fn(&BrandesTraversalState) -> bool,
        FCredit: FnMut(usize, &[f64]),
    {
        let n_channels = target_seeds.len();
        let state_len = traversal.state.len();
        let mut deltas: Vec<Vec<f64>> = (0..n_channels)
            .map(|_| vec![0.0f64; state_len])
            .collect();
        let mut credits_buf = vec![0.0f64; n_channels];

        for &state_idx in sorted_state_indices {
            let state = &traversal.state[state_idx];
            if !include_state(state) {
                continue;
            }
            let sigma_w = state.sigma;
            if sigma_w == 0.0 {
                continue;
            }

            // Check if any channel has non-zero dependency
            let mut any_nonzero = false;
            for ch in 0..n_channels {
                if target_seeds[ch][state_idx] + deltas[ch][state_idx] != 0.0 {
                    any_nonzero = true;
                    break;
                }
            }
            if !any_nonzero {
                continue;
            }

            // Propagate to predecessors
            for &pred_state_idx in &state.preds {
                let sigma_v = traversal.state[pred_state_idx].sigma;
                if sigma_v == 0.0 {
                    continue;
                }
                let factor = sigma_v / sigma_w;
                for ch in 0..n_channels {
                    let dependency =
                        target_seeds[ch][state_idx] + deltas[ch][state_idx];
                    deltas[ch][pred_state_idx] += factor * dependency;
                }
            }

            if state.node_idx == src_node_idx {
                continue;
            }

            // Compute credits and emit
            let mut any_credit = false;
            for ch in 0..n_channels {
                let dependency =
                    target_seeds[ch][state_idx] + deltas[ch][state_idx];
                let credit = (dependency - target_seeds[ch][state_idx]).max(0.0);
                credits_buf[ch] = credit;
                if credit > 0.0 {
                    any_credit = true;
                }
            }
            if any_credit {
                on_credit(state.node_idx, &credits_buf);
            }
        }
    }

    /// Validate and expand compact sampling_weights to node_bound() length.
    ///
    /// Accepts either node_count() (compact, one per live node in node_indices order)
    /// or node_bound() (sparse, indexed by raw node index) length.
    /// Returns a node_bound()-sized Vec where gap positions default to 0.0.
    fn expand_sampling_weights(&self, weights: &[f32]) -> PyResult<Vec<f32>> {
        let nc = self.node_count();
        let nb = self.node_bound();
        if weights.len() == nb {
            // Already sparse — validate and return as-is.
            for (i, &w) in weights.iter().enumerate() {
                if w < 0.0 || w > 1.0 {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "sampling_weights[{}] = {} is out of range [0.0, 1.0]",
                        i, w
                    )));
                }
            }
            Ok(weights.to_vec())
        } else if weights.len() == nc {
            // Compact — expand to sparse via node_indices mapping.
            for (i, &w) in weights.iter().enumerate() {
                if w < 0.0 || w > 1.0 {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "sampling_weights[{}] = {} is out of range [0.0, 1.0]",
                        i, w
                    )));
                }
            }
            let node_indices = self.node_indices();
            let mut expanded = vec![0.0f32; nb];
            for (pos, &idx) in node_indices.iter().enumerate() {
                expanded[idx] = weights[pos];
            }
            Ok(expanded)
        } else {
            Err(exceptions::PyValueError::new_err(format!(
                "sampling_weights length ({}) must match node_count ({}) or node_bound ({})",
                weights.len(),
                nc,
                nb,
            )))
        }
    }

    /// Compute Tobler's hiking function slope penalty for an edge.
    ///
    /// Returns a multiplier on edge length: ~1.0 on flat ground, >1.0 uphill,
    /// slightly <1.0 on gentle downhill (~-2.86° optimal).
    /// If either node lacks z, returns 1.0 (no penalty).
    ///
    /// Based on: Tobler, W. (1993). "Three Presentations on Geographical Analysis and Modeling."
    /// v = 6 * exp(-3.5 * |slope + 0.05|) km/h
    #[inline]
    fn slope_penalty(&self, from_idx: usize, to_idx: usize, length_2d: f32) -> f32 {
        if length_2d <= 0.0 {
            return 1.0;
        }
        let from_z = self.graph[NodeIndex::new(from_idx)].z;
        let to_z = self.graph[NodeIndex::new(to_idx)].z;
        match (from_z, to_z) {
            (Some(z_from), Some(z_to)) => {
                let slope = (z_to - z_from) as f32 / length_2d;
                // Tobler flat reference: exp(-3.5 * |0 + 0.05|) = exp(-0.175)
                const FLAT_FACTOR: f32 = 0.839_457;
                let slope_factor = (-3.5_f32 * (slope + 0.05).abs()).exp();
                FLAT_FACTOR / slope_factor
            }
            _ => 1.0,
        }
    }

    #[inline]
    fn edge_travel_seconds(
        &self,
        from_idx: usize,
        to_idx: usize,
        edge_payload: &crate::graph::EdgePayload,
        speed_m_s: f32,
        use_impedance: bool,
    ) -> f32 {
        if !edge_payload.seconds.is_nan() {
            return edge_payload.seconds;
        }
        let slope_pen = self.slope_penalty(from_idx, to_idx, edge_payload.length);
        let imp_factor = if use_impedance {
            edge_payload.imp_factor
        } else {
            1.0
        };
        (edge_payload.length * imp_factor * slope_pen) / speed_m_s
    }

    pub(crate) fn validate_dijkstra_inputs(&self, src_idx: usize, speed_m_s: f32) -> PyResult<()> {
        if src_idx >= self.node_bound() {
            return Err(exceptions::PyValueError::new_err(format!(
                "src_idx {} out of range for network with node_bound {}",
                src_idx,
                self.node_bound()
            )));
        }
        if self.graph.node_weight(NodeIndex::new(src_idx)).is_none() {
            return Err(exceptions::PyValueError::new_err(format!(
                "src_idx {} does not exist in the graph",
                src_idx
            )));
        }
        if !speed_m_s.is_finite() || speed_m_s <= 0.0 {
            return Err(exceptions::PyValueError::new_err(format!(
                "speed_m_s must be finite and positive, got {}",
                speed_m_s
            )));
        }
        Ok(())
    }

    fn prepare_source_sampling(
        &self,
        sample_probability: Option<f32>,
        sampling_weights: Option<Vec<f32>>,
        random_seed: Option<u64>,
        node_indices: &[usize],
    ) -> PyResult<SourceSamplingPlan> {
        let sampling_weights = match sampling_weights {
            Some(w) => Some(self.expand_sampling_weights(&w)?),
            None => None,
        };
        if let Some(prob) = sample_probability {
            if prob <= 0.0 || prob > 1.0 {
                return Err(exceptions::PyValueError::new_err(
                    "sample_probability must be in (0.0, 1.0]",
                ));
            }
        }

        let n = self.node_bound();
        let node_live: Vec<bool> = {
            let mut live = vec![false; n];
            for &idx in node_indices {
                live[idx] = self.is_node_live_unchecked(idx);
            }
            live
        };
        let sources = node_indices.to_vec();
        let sample_randoms = if sample_probability.is_some() {
            let mut rng = if let Some(seed) = random_seed {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_rng(&mut rand::rng())
            };
            (0..self.node_bound()).map(|_| rng.random()).collect()
        } else {
            Vec::new()
        };

        Ok(SourceSamplingPlan {
            sample_probability,
            sampling_weights,
            sample_randoms,
            sources,
            node_live,
        })
    }

    /// Determine whether a source should run and return `(wt, ipw)`.
    ///
    /// - `wt` is the source node weight, IPW-corrected: `weight(src)` in exact mode,
    ///   `weight(src) / p` under Bernoulli sampling.
    /// - `ipw` is the pure inverse-probability factor (independent of node weight):
    ///   `1.0` in exact mode, `1.0 / p` under sampling. Used for quantities that must be
    ///   sampling-corrected but not node-weighted (e.g. circuit ranks).
    ///
    /// Returns `None` when the source is not sampled (Bernoulli rejection) or `p <= 0`.
    #[inline]
    fn sample_source_weight(
        &self,
        src_idx: usize,
        sample_probability: Option<f32>,
        sampling_weights: Option<&[f32]>,
        sample_randoms: &[f32],
        sampled_source_count: &AtomicU32,
    ) -> Option<(f32, f32)> {
        let node_weight = self.get_node_weight_unchecked(src_idx);
        if let Some(prob) = sample_probability {
            let mut p = prob;
            if let Some(weights) = sampling_weights {
                p *= weights[src_idx];
            }
            if p <= 0.0 {
                return None;
            }
            if sample_randoms[src_idx] >= p {
                return None;
            }
            sampled_source_count.fetch_add(1, AtomicOrdering::Relaxed);
            Some((node_weight / p, 1.0 / p))
        } else {
            Some((node_weight, 1.0))
        }
    }

    fn dijkstra_tree_shortest_inner(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
    ) -> (Vec<usize>, Vec<NodeVisit>) {
        let mut tree_map = vec![NodeVisit::new(); self.node_bound()];
        let mut visited_nodes = Vec::new();
        tree_map[src_idx].agg_seconds = 0.0;
        tree_map[src_idx].discovered = true;
        tree_map[src_idx].short_dist = 0.0;
        let mut active = BinaryHeap::new();
        active.push(NodeDistance {
            node_idx: src_idx,
            metric: 0.0,
        });

        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
            if tree_map[node_idx].visited {
                continue;
            }
            tree_map[node_idx].visited = true;
            visited_nodes.push(node_idx);
            let current_node_index = NodeIndex::new(node_idx);
            // Downstream (Outgoing): forward paths from source to targets.
            for edge_ref in self
                .graph
                .edges_directed(current_node_index, Direction::Outgoing)
            {
                let nb_nd_idx = edge_ref.target();
                let nb_idx = nb_nd_idx.index();
                let edge_payload = edge_ref.weight();
                if nb_idx == node_idx || tree_map[nb_idx].visited {
                    continue;
                }
                if let Some(pred_idx) = tree_map[node_idx].pred {
                    if nb_idx == pred_idx {
                        continue;
                    }
                }
                let edge_seconds =
                    self.edge_travel_seconds(node_idx, nb_idx, edge_payload, speed_m_s, true);
                let total_seconds = tree_map[node_idx].agg_seconds + edge_seconds;
                if total_seconds > max_seconds as f32 {
                    continue;
                }
                if total_seconds < tree_map[nb_idx].agg_seconds {
                    tree_map[nb_idx].short_dist = total_seconds * speed_m_s;
                    tree_map[nb_idx].agg_seconds = total_seconds;
                    tree_map[nb_idx].pred = Some(node_idx);
                    tree_map[nb_idx].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_idx,
                        metric: total_seconds,
                    });
                }
            }
        }

        (visited_nodes, tree_map)
    }

    fn dijkstra_tree_angular(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        endpoint_slots: &[SmallVec<[String; 2]>],
    ) -> (Vec<usize>, Vec<NodeVisit>) {
        let node_count = self.node_bound();
        let state_count = node_count * 2;
        let mut states = vec![AngularTreeState::new(); state_count];
        let mut tree_map = vec![NodeVisit::new(); node_count];
        let mut visited_nodes = vec![src_idx];
        let mut reached_node_flags = vec![false; node_count];
        reached_node_flags[src_idx] = true;

        tree_map[src_idx].discovered = true;
        tree_map[src_idx].visited = true;
        tree_map[src_idx].simpl_dist = 0.0;
        tree_map[src_idx].agg_seconds = 0.0;

        let mut active = BinaryHeap::new();
        for slot in 0..2 {
            let state_idx = Self::angular_state_idx(src_idx, slot);
            states[state_idx].route_metric = 0.0;
            states[state_idx].simpl_dist = 0.0;
            states[state_idx].agg_seconds = 0.0;
            active.push(NodeDistance {
                node_idx: state_idx,
                metric: 0.0,
            });
        }

        while let Some(NodeDistance {
            node_idx: state_idx,
            ..
        }) = active.pop()
        {
            if states[state_idx].visited {
                continue;
            }
            states[state_idx].visited = true;

            let current_node_idx = state_idx / 2;
            let current_entry_slot = state_idx % 2;
            let current_node_index = NodeIndex::new(current_node_idx);

            for edge_ref in self
                .graph
                .edges_directed(current_node_index, Direction::Outgoing)
            {
                let next_node_idx = edge_ref.target().index();
                let edge_payload = edge_ref.weight();
                let shared_key = edge_payload
                    .shared_primal_node_key
                    .as_deref()
                    .expect("validated dual edge is missing shared_primal_node_key metadata");

                let current_shared_slot = endpoint_slots[current_node_idx]
                    .iter()
                    .position(|slot_key| slot_key == shared_key)
                    .expect("validated shared_primal_node_key missing from source dual node");
                if current_shared_slot != 1 - current_entry_slot {
                    continue;
                }

                let next_shared_slot = endpoint_slots[next_node_idx]
                    .iter()
                    .position(|slot_key| slot_key == shared_key)
                    .expect("validated shared_primal_node_key missing from target dual node");
                let next_state_idx = Self::angular_state_idx(next_node_idx, next_shared_slot);

                let edge_seconds = self.edge_travel_seconds(
                    current_node_idx,
                    next_node_idx,
                    edge_payload,
                    speed_m_s,
                    false,
                );
                let candidate_seconds = states[state_idx].agg_seconds + edge_seconds;
                if candidate_seconds > max_seconds as f32 {
                    continue;
                }

                let candidate_simpl = states[state_idx].simpl_dist + edge_payload.angle_sum;
                let candidate_metric =
                    candidate_simpl + (ANGULAR_ROUTE_TIE_BREAK_FACTOR * edge_payload.length);

                let improved = candidate_metric + TIE_EPSILON < states[next_state_idx].route_metric;
                let tied =
                    (candidate_metric - states[next_state_idx].route_metric).abs() <= TIE_EPSILON;

                if improved
                    || (tied
                        && candidate_seconds < states[next_state_idx].agg_seconds
                        && state_idx != next_state_idx)
                {
                    states[next_state_idx].route_metric = candidate_metric;
                    states[next_state_idx].simpl_dist = candidate_simpl;
                    states[next_state_idx].agg_seconds = candidate_seconds;
                    states[next_state_idx].pred_state_idx = Some(state_idx);
                    active.push(NodeDistance {
                        node_idx: next_state_idx,
                        metric: candidate_metric,
                    });

                    let next_visit = &mut tree_map[next_node_idx];
                    if !reached_node_flags[next_node_idx] {
                        reached_node_flags[next_node_idx] = true;
                        visited_nodes.push(next_node_idx);
                    }
                    let node_improved = candidate_simpl + TIE_EPSILON < next_visit.simpl_dist;
                    let node_tied = (candidate_simpl - next_visit.simpl_dist).abs() <= TIE_EPSILON;
                    if !next_visit.discovered
                        || node_improved
                        || (node_tied && candidate_seconds < next_visit.agg_seconds)
                    {
                        next_visit.discovered = true;
                        // This is the collapsed node-level tree view, not the oriented
                        // state heap. Mark the node as visited when we establish its
                        // current best angular arrival.
                        next_visit.visited = true;
                        next_visit.simpl_dist = candidate_simpl;
                        next_visit.agg_seconds = candidate_seconds;
                        next_visit.pred = Some(current_node_idx);
                    }
                }
            }
        }

        (visited_nodes, tree_map)
    }

    /// Two-phase Brandes Dijkstra for shortest-path centrality with tolerance.
    ///
    /// Phase 1: Exact Dijkstra (TIE_EPSILON only) to compute final settled distances,
    /// visit order, and exact predecessors.
    ///
    /// Phase 2: Rebuild predecessors and sigma using the user's tolerance against final
    /// (immutable) distances. Processing in settled order guarantees correct sigma
    /// accumulation because all predecessors of a node have lower cost and are processed
    /// first. This eliminates the predecessor drift bug where single-pass tolerance
    /// tracking accumulates stale predecessors against tentative distances.
    fn dijkstra_brandes_shortest(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        tolerance: f32,
        upstream: bool,
    ) -> BrandesTraversal {
        assert!(
            tolerance >= TIE_EPSILON,
            "Tolerance must be >= TIE_EPSILON to avoid float-comparison bugs"
        );
        let direction = if upstream {
            Direction::Incoming
        } else {
            Direction::Outgoing
        };
        let node_count = self.node_bound();
        let mut states = (0..node_count)
            .map(BrandesTraversalState::new)
            .collect::<Vec<_>>();
        let mut visited_state_indices = Vec::new();
        let mut best_route_cost = vec![f32::INFINITY; node_count];
        let mut best_agg_seconds = vec![f32::INFINITY; node_count];

        // Phase 1: Exact Dijkstra — settled distances and predecessors (TIE_EPSILON).
        states[src_idx].sigma = 1.0;
        states[src_idx].route_cost = 0.0;
        states[src_idx].agg_seconds = 0.0;
        best_route_cost[src_idx] = 0.0;
        best_agg_seconds[src_idx] = 0.0;

        let mut active = BinaryHeap::new();
        active.push(NodeDistance {
            node_idx: src_idx,
            metric: 0.0,
        });

        while let Some(NodeDistance {
            node_idx: state_idx,
            ..
        }) = active.pop()
        {
            if states[state_idx].visited {
                continue;
            }
            states[state_idx].visited = true;
            visited_state_indices.push(state_idx);

            let current_node_idx = states[state_idx].node_idx;
            let current_node_index = NodeIndex::new(current_node_idx);
            for edge_ref in self.graph.edges_directed(current_node_index, direction) {
                let nb_idx = if upstream {
                    edge_ref.source().index()
                } else {
                    edge_ref.target().index()
                };
                let edge_payload = edge_ref.weight();
                if nb_idx == current_node_idx {
                    continue;
                }
                let (from, to) = if upstream {
                    (nb_idx, current_node_idx)
                } else {
                    (current_node_idx, nb_idx)
                };
                let edge_seconds =
                    self.edge_travel_seconds(from, to, edge_payload, speed_m_s, true);
                let candidate_seconds = states[state_idx].agg_seconds + edge_seconds;
                if candidate_seconds > max_seconds as f32 {
                    continue;
                }
                if states[nb_idx].visited {
                    continue;
                }
                let candidate_route = candidate_seconds * speed_m_s;
                let improved = candidate_seconds < states[nb_idx].agg_seconds;
                let tied = candidate_seconds <= states[nb_idx].agg_seconds * (1.0 + TIE_EPSILON);

                if improved {
                    if candidate_seconds < states[nb_idx].agg_seconds * (1.0 - TIE_EPSILON) {
                        states[nb_idx].preds.clear();
                        states[nb_idx].sigma = states[state_idx].sigma;
                    } else {
                        states[nb_idx].sigma += states[state_idx].sigma;
                    }
                    states[nb_idx].route_cost = candidate_route;
                    states[nb_idx].agg_seconds = candidate_seconds;
                    states[nb_idx].preds.push(state_idx);
                    active.push(NodeDistance {
                        node_idx: nb_idx,
                        metric: candidate_route,
                    });
                } else if tied && !states[nb_idx].preds.contains(&state_idx) {
                    if candidate_seconds < states[nb_idx].agg_seconds {
                        states[nb_idx].agg_seconds = candidate_seconds;
                    }
                    states[nb_idx].preds.push(state_idx);
                    states[nb_idx].sigma += states[state_idx].sigma;
                }

                best_route_cost[nb_idx] = states[nb_idx].route_cost;
                best_agg_seconds[nb_idx] = states[nb_idx].agg_seconds;
            }
        }

        // Phase 2: Rebuild predecessors and sigma using user tolerance against final distances.
        // Re-iterates edges using settled distances (immutable after Phase 1). Processing in
        // settled order guarantees sigma accumulates correctly (all predecessors finalised first).
        if tolerance > TIE_EPSILON {
            let mut visit_pos = vec![usize::MAX; node_count];
            for (pos, &idx) in visited_state_indices.iter().enumerate() {
                visit_pos[idx] = pos;
            }
            for &idx in &visited_state_indices {
                states[idx].preds.clear();
                states[idx].sigma = 0.0;
            }
            states[src_idx].sigma = 1.0;
            for (pos, &u_idx) in visited_state_indices.iter().enumerate() {
                let u_node_index = NodeIndex::new(states[u_idx].node_idx);
                for edge_ref in self.graph.edges_directed(u_node_index, direction) {
                    let v_idx = if upstream {
                        edge_ref.source().index()
                    } else {
                        edge_ref.target().index()
                    };
                    if v_idx == u_idx {
                        continue;
                    }
                    if visit_pos[v_idx] <= pos {
                        continue;
                    }
                    let (from, to) = if upstream {
                        (v_idx, states[u_idx].node_idx)
                    } else {
                        (states[u_idx].node_idx, v_idx)
                    };
                    let edge_seconds =
                        self.edge_travel_seconds(from, to, edge_ref.weight(), speed_m_s, true);
                    let path_seconds = states[u_idx].agg_seconds + edge_seconds;
                    if path_seconds <= states[v_idx].agg_seconds * (1.0 + tolerance)
                        && !states[v_idx].preds.contains(&u_idx)
                    {
                        states[v_idx].preds.push(u_idx);
                        states[v_idx].sigma += states[u_idx].sigma;
                    }
                }
            }
        }

        let reached_node_indices = Self::reached_node_indices(&best_route_cost);

        BrandesTraversal {
            visited_state_indices,
            reached_node_indices,
            state: states,
            best_route_cost,
            best_agg_seconds,
        }
    }
}

#[pymethods]
impl NetworkStructure {
    #[pyo3(signature = (src_idx, max_seconds, speed_m_s))]
    pub fn dijkstra_tree_shortest(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
    ) -> PyResult<(Vec<usize>, Vec<NodeVisit>)> {
        self.validate_dijkstra_inputs(src_idx, speed_m_s)?;
        Ok(self.dijkstra_tree_shortest_inner(src_idx, max_seconds, speed_m_s))
    }

    /// Per-node hit counts from bounded Dijkstra traversals over the given sources.
    ///
    /// For each distance threshold, counts how many of the sources reach each node
    /// within that metric distance (one traversal per source, to the largest
    /// threshold). Backs the sampling pilot (cityseer.sampling.estimate_polled_reach):
    /// on an undirected network a node's hit count is binomial in its reach, so
    /// hits / m * n estimates reach at every threshold from one traversal set.
    /// Returns one Vec of length node_bound() per distance, indexed by raw node index.
    #[pyo3(signature = (src_idxs, distances, speed_m_s))]
    pub fn poll_reach_hits(
        &self,
        py: Python,
        src_idxs: Vec<usize>,
        distances: Vec<u32>,
        speed_m_s: f32,
    ) -> PyResult<Vec<Vec<u32>>> {
        if distances.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "poll_reach_hits requires at least one distance",
            ));
        }
        for src_idx in &src_idxs {
            self.validate_dijkstra_inputs(*src_idx, speed_m_s)?;
        }
        let max_dist = *distances.iter().max().unwrap();
        let max_seconds = (max_dist as f32 / speed_m_s).ceil() as u32;
        let bound = self.node_bound();
        let n_dist = distances.len();
        let counts = py.detach(move || {
            src_idxs
                .par_iter()
                .fold(
                    || vec![vec![0u32; bound]; n_dist],
                    |mut acc, src_idx| {
                        let (visited, tree_map) =
                            self.dijkstra_tree_shortest_inner(*src_idx, max_seconds, speed_m_s);
                        for node_idx in visited {
                            let dist = tree_map[node_idx].short_dist;
                            for (di, thresh) in distances.iter().enumerate() {
                                if dist <= *thresh as f32 {
                                    acc[di][node_idx] += 1;
                                }
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![vec![0u32; bound]; n_dist],
                    |mut a, b| {
                        for (a_row, b_row) in a.iter_mut().zip(b) {
                            for (a_val, b_val) in a_row.iter_mut().zip(b_row) {
                                *a_val += b_val;
                            }
                        }
                        a
                    },
                )
        });
        Ok(counts)
    }

    #[pyo3(signature = (src_idx, max_seconds, speed_m_s))]
    pub fn dijkstra_tree_simplest(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
    ) -> PyResult<(Vec<usize>, Vec<NodeVisit>)> {
        self.validate_dual_for_angular("dijkstra_tree_simplest")?;
        self.validate_dijkstra_inputs(src_idx, speed_m_s)?;
        let endpoint_slots = self.dual_node_endpoint_slots()?;
        Ok(self.dijkstra_tree_angular(src_idx, max_seconds, speed_m_s, &endpoint_slots))
    }

    // =========================================================================
    // Combined centrality (closeness + betweenness from single Dijkstra)
    // =========================================================================

    /// Compute node centrality using shortest paths with a single Dijkstra per source.
    ///
    /// Closeness and betweenness metrics are specified as lists of (name, expression)
    /// pairs. Expressions use variables `c` (metric distance) and `p` (normalised
    /// progress = c / threshold). Each expression is parsed once per thread via `meval`
    /// and evaluated per reached node (closeness) or per shortest path (betweenness).
    ///
    /// When `sample_probability` is set, Bernoulli sampling with inverse-probability
    /// weighting (IPW) is used.
    #[pyo3(signature = (
        distances=None,
        minutes=None,
        closeness_exprs=None,
        betweenness_exprs=None,
        compute_cycles=None,
        speed_m_s=None,
        tolerance=None,
        sample_probability=None,
        sampling_weights=None,
        random_seed=None,
        pbar_disabled=None
    ))]
    pub fn centrality_shortest(
        &self,
        distances: Option<Vec<u32>>,
        minutes: Option<Vec<f32>>,
        closeness_exprs: Option<Vec<(String, String)>>,
        betweenness_exprs: Option<Vec<(String, String)>>,
        compute_cycles: Option<bool>,
        speed_m_s: Option<f32>,
        tolerance: Option<f32>,
        sample_probability: Option<f32>,
        sampling_weights: Option<Vec<f32>>,
        random_seed: Option<u64>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralityResult> {
        let closeness_exprs = closeness_exprs.unwrap_or_default();
        let betweenness_exprs = betweenness_exprs.unwrap_or_default();
        let compute_cycles = compute_cycles.unwrap_or(false);
        if closeness_exprs.is_empty() && betweenness_exprs.is_empty() && !compute_cycles {
            return Err(exceptions::PyValueError::new_err(
                "At least one of closeness_exprs, betweenness_exprs, or compute_cycles must be provided.",
            ));
        }
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let tolerance = validate_tolerance(tolerance)?;
        let (distances, seconds) = common::pair_distances_and_time(
            speed_m_s, distances, minutes,
        )?;
        // Validate all expressions up front
        let closeness_validated = common::validate_metric_exprs(&closeness_exprs)?;
        let betweenness_validated = common::validate_metric_exprs(&betweenness_exprs)?;

        let max_walk_seconds = *seconds
            .iter()
            .max()
            .expect("Seconds vector should not be empty");
        let node_keys_py = self.node_keys_py(py);
        let node_indices = self.node_indices();
        let sampling_plan = self.prepare_source_sampling(
            sample_probability,
            sampling_weights,
            random_seed,
            &node_indices,
        )?;
        let n = self.node_bound();
        let closeness_names: Vec<String> = closeness_validated.iter().map(|(n, _)| n.clone()).collect();
        let betweenness_names: Vec<String> = betweenness_validated.iter().map(|(n, _)| n.clone()).collect();
        let mut res = CentralityResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
            &closeness_names,
            &betweenness_names,
            compute_cycles,
            n,
            0.0,
        );

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        // Atomic counters for tracking source reachability when sampling
        let source_reachability_totals: Vec<AtomicU32> =
            distances.iter().map(|_| AtomicU32::new(0)).collect();
        let sampled_source_count = AtomicU32::new(0);

        let result = py.detach(move || {
            sampling_plan.sources.par_iter().for_each(|src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                // A buffer (non-live) source only needs a traversal when betweenness is being
                // computed (it counts routes from every node) or when sampling (closeness/cycles
                // target-aggregate onto live nodes via buffer sources). Otherwise skip Dijkstra.
                if !sampling_plan.node_live[*src_idx]
                    && betweenness_validated.is_empty()
                    && !sampling_plan.is_sampling()
                {
                    return;
                }

                let Some((wt, ipw)) = self.sample_source_weight(
                    *src_idx,
                    sampling_plan.sample_probability,
                    sampling_plan.sampling_weights.as_deref(),
                    &sampling_plan.sample_randoms,
                    &sampled_source_count,
                ) else {
                    return;
                };

                let traversal = self.dijkstra_brandes_shortest(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    tolerance,
                    sampling_plan.is_sampling(),
                );

                // Parse all expressions once per thread
                let closeness_fns: Vec<_> = closeness_validated
                    .iter()
                    .map(|(_, expr)| common::parse_metric_expr(expr))
                    .collect();
                let betw_fns: Vec<_> = betweenness_validated
                    .iter()
                    .map(|(_, expr)| common::parse_metric_expr(expr))
                    .collect();

                // IPW-only weight for cycles (no node weight, just sampling correction).
                let cycles_wt = ipw;

                // --- Closeness accumulation ---
                // In exact mode closeness/cycles aggregate at the source, so only live nodes
                // contribute; when sampling they target-aggregate onto live nodes, so buffer
                // sources are needed. (Buffer sources still run for betweenness above.)
                if (!closeness_fns.is_empty() || compute_cycles)
                    && (sampling_plan.is_sampling() || sampling_plan.node_live[*src_idx])
                {
                    let is_sampling = sampling_plan.is_sampling();
                    // Cycles
                    let source_cycle_scores = if compute_cycles {
                        Some(self.circuit_ranks_from_traversal(&traversal, &distances))
                    } else {
                        None
                    };
                    if let Some(ref scores) = source_cycle_scores {
                        if !is_sampling {
                            if let Some(ref cycles_metric) = res.cycles_metric {
                                for i in 0..distances.len() {
                                    cycles_metric.metric[i][*src_idx]
                                        .fetch_add(scores[i] as f64, AtomicOrdering::Relaxed);
                                }
                            }
                        }
                    }
                    for &to_idx in &traversal.reached_node_indices {
                        if to_idx == *src_idx {
                            continue;
                        }
                        if !traversal.best_agg_seconds[to_idx].is_finite() {
                            continue;
                        }
                        if is_sampling && !sampling_plan.node_live[to_idx] {
                            continue;
                        }
                        if is_sampling {
                            for i in 0..distances.len() {
                                if traversal.best_route_cost[to_idx] <= distances[i] as f32 {
                                    source_reachability_totals[i]
                                        .fetch_add(1, AtomicOrdering::Relaxed);
                                }
                            }
                        }
                        let agg_idx = if is_sampling { to_idx } else { *src_idx };
                        // Gravity weighting. Non-sampling aggregates at the source, so
                        // weight by the destination; sampling aggregates at the
                        // destination, so weight by the (IPW-corrected) source `wt`.
                        // Both yield A(N) = sum_j w_j * f(d(N, j)).
                        let cw = if is_sampling {
                            wt
                        } else {
                            self.get_node_weight_unchecked(to_idx)
                        };
                        let cost = traversal.best_route_cost[to_idx];
                        for i in 0..distances.len() {
                            let distance = distances[i];
                            if cost <= distance as f32 {
                                let p = cost / distance as f32;
                                // Evaluate each closeness expression
                                for (expr_idx, f) in closeness_fns.iter().enumerate() {
                                    let val = f(cost, p) * cw;
                                    res.closeness_metrics[expr_idx].1.metric[i][agg_idx]
                                        .fetch_add(val as f64, AtomicOrdering::Relaxed);
                                }
                                // Cycles: target-based broadcast when sampling.
                                if is_sampling {
                                    if let Some(ref scores) = source_cycle_scores {
                                        if let Some(ref cycles_metric) = res.cycles_metric {
                                            cycles_metric.metric[i][to_idx].fetch_add(
                                                (scores[i] * cycles_wt) as f64,
                                                AtomicOrdering::Relaxed,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // --- Betweenness backpropagation ---
                if !betw_fns.is_empty() {
                    let n_betw = betw_fns.len();
                    let sorted_state_indices = Self::sorted_brandes_state_indices(&traversal);
                    let mut target_seeds: Vec<Vec<f64>> = (0..n_betw)
                        .map(|_| vec![0.0f64; traversal.state.len()])
                        .collect();

                    for d_idx in 0..distances.len() {
                        let dist_threshold = distances[d_idx] as f32;
                        for seed in &mut target_seeds {
                            seed.fill(0.0);
                        }

                        for &to_idx in &traversal.reached_node_indices {
                            if to_idx == *src_idx {
                                continue;
                            }
                            if traversal.best_route_cost[to_idx] > dist_threshold {
                                continue;
                            }

                            // Count every ordered pair, including routes that pass through the inner
                            // area from buffer to buffer (`live` is an output filter, not a source
                            // restriction). Directed pairs count fully; undirected pairs are halved,
                            // which is exactly the global /2 for the two symmetric orderings.
                            let pair_count = if self.is_directed { 1.0 } else { 0.5 };
                            // Destination weight; combined with the source weight `wt`
                            // applied to the final credit, this gives product weighting
                            // w_s * w_t per O-D pair.
                            let seg_scale = self.get_node_weight_unchecked(to_idx) as f64;
                            let cost = traversal.best_route_cost[to_idx];
                            let p = cost / dist_threshold;
                            for (expr_idx, f) in betw_fns.iter().enumerate() {
                                target_seeds[expr_idx][to_idx] +=
                                    pair_count * seg_scale * f(cost, p) as f64;
                            }
                        }

                        let seed_refs: Vec<&[f64]> =
                            target_seeds.iter().map(|s| s.as_slice()).collect();
                        Self::brandes_backprop_multi(
                            &traversal,
                            &sorted_state_indices,
                            *src_idx,
                            &seed_refs,
                            |state| state.route_cost <= dist_threshold,
                            |inter_node_idx, credits| {
                                for (expr_idx, &credit) in credits.iter().enumerate() {
                                    if credit > 0.0 {
                                        res.betweenness_metrics[expr_idx].1.metric[d_idx]
                                            [inter_node_idx]
                                            .fetch_add(
                                                credit * wt as f64,
                                                AtomicOrdering::Relaxed,
                                            );
                                    }
                                }
                            },
                        );
                    }
                }
            });

            // Sampling metadata
            if sampling_plan.is_sampling() {
                res.sampled_source_count = sampled_source_count.load(AtomicOrdering::Relaxed);
                res.reachability_totals = source_reachability_totals
                    .iter()
                    .map(|a| a.load(AtomicOrdering::Relaxed))
                    .collect();
            }

            res
        });

        Ok(result)
    }

    /// Compute node centrality using simplest (angular) paths on the dual graph.
    ///
    /// Angular routing is evaluated on two directed states per segment. Each
    /// source segment seeds both orientations into a single Brandes traversal.
    ///
    /// Expressions use `c` (angular cost) and `p` (normalised time progress =
    /// agg_seconds / max_seconds).
    #[pyo3(signature = (
        distances=None,
        minutes=None,
        closeness_exprs=None,
        betweenness_exprs=None,
        speed_m_s=None,
        tolerance=None,
        sample_probability=None,
        sampling_weights=None,
        random_seed=None,
        pbar_disabled=None
    ))]
    pub fn centrality_simplest(
        &self,
        distances: Option<Vec<u32>>,
        minutes: Option<Vec<f32>>,
        closeness_exprs: Option<Vec<(String, String)>>,
        betweenness_exprs: Option<Vec<(String, String)>>,
        speed_m_s: Option<f32>,
        tolerance: Option<f32>,
        sample_probability: Option<f32>,
        sampling_weights: Option<Vec<f32>>,
        random_seed: Option<u64>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralityResult> {
        self.validate_dual_for_angular("centrality_simplest")?;
        let closeness_exprs = closeness_exprs.unwrap_or_default();
        let betweenness_exprs = betweenness_exprs.unwrap_or_default();
        if closeness_exprs.is_empty() && betweenness_exprs.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "At least one of closeness_exprs or betweenness_exprs must be provided.",
            ));
        }
        let tolerance = validate_tolerance(tolerance)?;
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, seconds) = common::pair_distances_and_time(
            speed_m_s, distances, minutes,
        )?;
        let closeness_validated = common::validate_metric_exprs(&closeness_exprs)?;
        let betweenness_validated = common::validate_metric_exprs(&betweenness_exprs)?;

        let max_walk_seconds = *seconds
            .iter()
            .max()
            .expect("Seconds vector should not be empty");
        let node_keys_py = self.node_keys_py(py);
        let node_indices = self.node_indices();
        let sampling_plan = self.prepare_source_sampling(
            sample_probability,
            sampling_weights,
            random_seed,
            &node_indices,
        )?;
        let n = self.node_bound();
        let closeness_names: Vec<String> = closeness_validated.iter().map(|(n, _)| n.clone()).collect();
        let betweenness_names: Vec<String> = betweenness_validated.iter().map(|(n, _)| n.clone()).collect();
        let mut res = CentralityResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
            &closeness_names,
            &betweenness_names,
            false, // no cycles for simplest
            n,
            0.0,
        );
        let angular_endpoint_slots = self.dual_node_endpoint_slots()?;

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        // Atomic counters for tracking source reachability when sampling
        let source_reachability_totals: Vec<AtomicU32> =
            seconds.iter().map(|_| AtomicU32::new(0)).collect();
        let sampled_source_count = AtomicU32::new(0);

        let result = py.detach(move || {
            sampling_plan.sources.par_iter().for_each(|src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                // A buffer (non-live) source only needs a traversal when betweenness is being
                // computed (it counts routes from every node) or when sampling (closeness/cycles
                // target-aggregate onto live nodes via buffer sources). Otherwise skip Dijkstra.
                if !sampling_plan.node_live[*src_idx]
                    && betweenness_validated.is_empty()
                    && !sampling_plan.is_sampling()
                {
                    return;
                }

                let Some((wt, _ipw)) = self.sample_source_weight(
                    *src_idx,
                    sampling_plan.sample_probability,
                    sampling_plan.sampling_weights.as_deref(),
                    &sampling_plan.sample_randoms,
                    &sampled_source_count,
                ) else {
                    return;
                };

                let traversal = self.dijkstra_brandes_angular(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    tolerance,
                    &angular_endpoint_slots,
                    sampling_plan.is_sampling(),
                );

                // Parse all expressions once per thread
                let closeness_fns: Vec<_> = closeness_validated
                    .iter()
                    .map(|(_, expr)| common::parse_metric_expr(expr))
                    .collect();
                let betw_fns: Vec<_> = betweenness_validated
                    .iter()
                    .map(|(_, expr)| common::parse_metric_expr(expr))
                    .collect();

                // --- Closeness accumulation ---
                // Exact mode aggregates at the source (live only); sampling target-aggregates
                // onto live nodes (buffer sources needed). Buffer sources still run for betweenness.
                if !closeness_fns.is_empty()
                    && (sampling_plan.is_sampling() || sampling_plan.node_live[*src_idx])
                {
                    let is_sampling = sampling_plan.is_sampling();
                    for &to_idx in &traversal.reached_node_indices {
                        if to_idx == *src_idx {
                            continue;
                        }
                        let best_simpl_dist = traversal.best_route_cost[to_idx];
                        let best_agg_seconds = traversal.best_agg_seconds[to_idx];
                        if !best_simpl_dist.is_finite() || !best_agg_seconds.is_finite() {
                            continue;
                        }
                        if is_sampling && !sampling_plan.node_live[to_idx] {
                            continue;
                        }
                        if is_sampling {
                            for i in 0..seconds.len() {
                                if best_agg_seconds <= seconds[i] as f32 {
                                    source_reachability_totals[i]
                                        .fetch_add(1, AtomicOrdering::Relaxed);
                                }
                            }
                        }
                        let agg_idx = if is_sampling { to_idx } else { *src_idx };
                        // Gravity weighting. Non-sampling aggregates at the source, so
                        // weight by the destination; sampling aggregates at the
                        // destination, so weight by the (IPW-corrected) source `wt`.
                        // Both yield A(N) = sum_j w_j * f(d(N, j)).
                        let cw = if is_sampling {
                            wt
                        } else {
                            self.get_node_weight_unchecked(to_idx)
                        };
                        // c = angular cost, p = normalised time progress
                        let c = best_simpl_dist;
                        for i in 0..seconds.len() {
                            let sec = seconds[i];
                            if best_agg_seconds <= sec as f32 {
                                let p = best_agg_seconds / sec as f32;
                                for (expr_idx, f) in closeness_fns.iter().enumerate() {
                                    let val = f(c, p) * cw;
                                    res.closeness_metrics[expr_idx].1.metric[i][agg_idx]
                                        .fetch_add(val as f64, AtomicOrdering::Relaxed);
                                }
                            }
                        }
                    }
                }

                // --- Betweenness backpropagation ---
                if !betw_fns.is_empty() {
                    let n_betw = betw_fns.len();
                    let sorted_state_indices = Self::sorted_brandes_state_indices(&traversal);
                    let mut target_seeds: Vec<Vec<f64>> = (0..n_betw)
                        .map(|_| vec![0.0f64; traversal.state.len()])
                        .collect();

                    for d_idx in 0..seconds.len() {
                        let sec_threshold = seconds[d_idx] as f32;
                        for seed in &mut target_seeds {
                            seed.fill(0.0);
                        }

                        for &to_idx in &traversal.reached_node_indices {
                            if to_idx == *src_idx {
                                continue;
                            }
                            let best_state_indices = Self::best_angular_target_states(
                                &traversal,
                                to_idx,
                                sec_threshold,
                                tolerance,
                            );
                            if best_state_indices.is_empty() {
                                continue;
                            }
                            // Count every ordered pair, including routes that pass through the inner
                            // area from buffer to buffer (`live` is an output filter, not a source
                            // restriction). Directed pairs count fully; undirected pairs are halved,
                            // which is exactly the global /2 for the two symmetric orderings.
                            let pair_count = if self.is_directed { 1.0 } else { 0.5 };
                            // Destination weight; combined with the source weight `wt`
                            // applied to the final credit, this gives product weighting
                            // w_s * w_t per O-D pair.
                            let seg_scale = self.get_node_weight_unchecked(to_idx) as f64;
                            let sigma_total: f64 = best_state_indices
                                .iter()
                                .map(|&state_idx| traversal.state[state_idx].sigma)
                                .sum();
                            if sigma_total == 0.0 {
                                continue;
                            }
                            // For betweenness expressions: c = angular cost, p = time progress
                            let c = traversal.best_route_cost[to_idx];
                            let p = traversal.best_agg_seconds[to_idx] / sec_threshold;
                            for &state_idx in &best_state_indices {
                                let sigma_frac =
                                    traversal.state[state_idx].sigma / sigma_total;
                                for (expr_idx, f) in betw_fns.iter().enumerate() {
                                    target_seeds[expr_idx][state_idx] +=
                                        pair_count * seg_scale * f(c, p) as f64 * sigma_frac;
                                }
                            }
                        }

                        let seed_refs: Vec<&[f64]> =
                            target_seeds.iter().map(|s| s.as_slice()).collect();
                        Self::brandes_backprop_multi(
                            &traversal,
                            &sorted_state_indices,
                            *src_idx,
                            &seed_refs,
                            |state| state.agg_seconds <= sec_threshold,
                            |inter_node_idx, credits| {
                                for (expr_idx, &credit) in credits.iter().enumerate() {
                                    if credit > 0.0 {
                                        res.betweenness_metrics[expr_idx].1.metric[d_idx]
                                            [inter_node_idx]
                                            .fetch_add(
                                                credit * wt as f64,
                                                AtomicOrdering::Relaxed,
                                            );
                                    }
                                }
                            },
                        );
                    }
                }
            });

            // Sampling metadata
            if sampling_plan.is_sampling() {
                res.sampled_source_count = sampled_source_count.load(AtomicOrdering::Relaxed);
                res.reachability_totals = source_reachability_totals
                    .iter()
                    .map(|a| a.load(AtomicOrdering::Relaxed))
                    .collect();
            }

            res
        });

        Ok(result)
    }

    // =========================================================================
    // OD-weighted betweenness (Brandes multi-predecessor shortest paths)
    // =========================================================================

    /// Compute OD-weighted betweenness centrality using shortest paths.
    ///
    /// Uses Brandes multi-predecessor Dijkstra from each source that has
    /// outbound OD trips. Betweenness expressions use `c` (metric distance)
    /// and `p` (normalised progress = c / threshold).
    #[pyo3(signature = (
        od_matrix,
        distances=None,
        minutes=None,
        betweenness_exprs=None,
        speed_m_s=None,
        tolerance=None,
        pbar_disabled=None
    ))]
    pub fn betweenness_od_shortest(
        &self,
        od_matrix: &OdMatrix,
        distances: Option<Vec<u32>>,
        minutes: Option<Vec<f32>>,
        betweenness_exprs: Option<Vec<(String, String)>>,
        speed_m_s: Option<f32>,
        tolerance: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralityResult> {
        let betweenness_exprs = betweenness_exprs.unwrap_or_default();
        if betweenness_exprs.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "betweenness_exprs must contain at least one expression.",
            ));
        }
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, seconds) = common::pair_distances_and_time(
            speed_m_s, distances, minutes,
        )?;
        let betweenness_validated = common::validate_metric_exprs(&betweenness_exprs)?;
        let max_walk_seconds = *seconds
            .iter()
            .max()
            .expect("Seconds vector should not be empty");
        let tolerance = validate_tolerance(tolerance)?;

        let od_map = &od_matrix.map;
        let n = self.node_bound();

        let node_keys_py = self.node_keys_py(py);
        let node_indices = self.node_indices();
        let betweenness_names: Vec<String> = betweenness_validated.iter().map(|(n, _)| n.clone()).collect();
        let res = CentralityResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
            &[],
            &betweenness_names,
            false,
            n,
            0.0,
        );

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        let result = py.detach(move || {
            node_indices.par_iter().for_each(|src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !self.is_node_live_unchecked(*src_idx) {
                    return;
                }
                let src_dests = match od_map.get(src_idx) {
                    Some(dests) => dests,
                    None => return,
                };

                let traversal = self.dijkstra_brandes_shortest(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    tolerance,
                    false,
                );
                let betw_fns: Vec<_> = betweenness_validated
                    .iter()
                    .map(|(_, expr)| common::parse_metric_expr(expr))
                    .collect();
                let n_betw = betw_fns.len();

                let sorted_visited = Self::sorted_brandes_state_indices(&traversal);

                let mut target_seeds: Vec<Vec<f64>> = (0..n_betw)
                    .map(|_| vec![0.0f64; traversal.state.len()])
                    .collect();
                for d_idx in 0..distances.len() {
                    let dist_threshold = distances[d_idx] as f32;
                    for seed in &mut target_seeds {
                        seed.fill(0.0);
                    }

                    for (&dest, &od_w) in src_dests {
                        if traversal.best_route_cost[dest] > dist_threshold {
                            continue;
                        }
                        let cost = traversal.best_route_cost[dest];
                        let p = cost / dist_threshold;
                        for (expr_idx, f) in betw_fns.iter().enumerate() {
                            target_seeds[expr_idx][dest] +=
                                od_w as f64 * f(cost, p) as f64;
                        }
                    }

                    let seed_refs: Vec<&[f64]> =
                        target_seeds.iter().map(|s| s.as_slice()).collect();
                    Self::brandes_backprop_multi(
                        &traversal,
                        &sorted_visited,
                        *src_idx,
                        &seed_refs,
                        |state| state.route_cost <= dist_threshold,
                        |inter_node_idx, credits| {
                            for (expr_idx, &credit) in credits.iter().enumerate() {
                                if credit > 0.0 {
                                    res.betweenness_metrics[expr_idx].1.metric[d_idx]
                                        [inter_node_idx]
                                        .fetch_add(credit, AtomicOrdering::Relaxed);
                                }
                            }
                        },
                    );
                }
            });

            res
        });

        Ok(result)
    }

    /// Demand-weighted (flow) betweenness from a singly / origin-constrained spatial interaction model.
    ///
    /// Each origin distributes its full weight across reachable destinations in proportion to
    /// `W_d * decay(c)`, where `decay` is the supplied expression evaluated on `c` (metric cost)
    /// and `p` (normalised progress to the threshold) — the gravity model is one instance of this
    /// spatial interaction form. The allocated origin-destination flows are then routed along
    /// shortest paths via Brandes back-propagation, accumulating flow betweenness at intermediate
    /// nodes. Origins and destinations are each aggregated by node first, so several snapped points
    /// sharing a node contribute their summed weight (and a node only triggers one Dijkstra). When
    /// `closest_destination` is true, an origin routes its full weight to its single nearest
    /// reachable destination instead of allocating across all of them.
    #[pyo3(signature = (
        origins,
        destinations,
        decay_fn,
        distances=None,
        minutes=None,
        closest_destination=false,
        metric_name=None,
        speed_m_s=None,
        tolerance=None,
        pbar_disabled=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn betweenness_demand_shortest(
        &self,
        origins: Vec<(usize, f64)>,
        destinations: Vec<(usize, f64)>,
        decay_fn: String,
        distances: Option<Vec<u32>>,
        minutes: Option<Vec<f32>>,
        closest_destination: bool,
        metric_name: Option<String>,
        speed_m_s: Option<f32>,
        tolerance: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralityResult> {
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, seconds) = common::pair_distances_and_time(speed_m_s, distances, minutes)?;
        // Validate the decay expression up front via the shared metric-expression path.
        let metric_name = metric_name.unwrap_or_else(|| "demand".to_string());
        let decay_validated = common::validate_metric_exprs(&[(metric_name.clone(), decay_fn)])?;
        let decay_expr = decay_validated[0].1.clone();
        let max_walk_seconds = *seconds.iter().max().expect("Seconds vector should not be empty");
        let tolerance = validate_tolerance(tolerance)?;

        let n = self.node_bound();
        // Aggregate weights by node so duplicate-snapped points are summed rather than discarded,
        // and each node is only visited once.
        let mut origin_weights: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for (node_idx, w) in origins {
            if node_idx >= n {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Origin node index {} is out of bounds (max {})",
                    node_idx,
                    n - 1
                )));
            }
            if w > 0.0 {
                *origin_weights.entry(node_idx).or_insert(0.0) += w;
            }
        }
        let mut dest_weights: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for (node_idx, w) in destinations {
            if node_idx >= n {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Destination node index {} is out of bounds (max {})",
                    node_idx,
                    n - 1
                )));
            }
            if w > 0.0 {
                *dest_weights.entry(node_idx).or_insert(0.0) += w;
            }
        }
        let origin_list: Vec<(usize, f64)> = origin_weights.into_iter().collect();

        let node_keys_py = self.node_keys_py(py);
        let node_indices = self.node_indices();
        let res = CentralityResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
            &[],
            std::slice::from_ref(&metric_name),
            false,
            n,
            0.0,
        );

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        let result = py.detach(move || {
            origin_list.par_iter().for_each(|&(src_idx, o_weight)| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !self.is_node_live_unchecked(src_idx) {
                    return;
                }
                let traversal =
                    self.dijkstra_brandes_shortest(src_idx, max_walk_seconds, speed_m_s, tolerance, false);
                let decay = common::parse_metric_expr(&decay_expr);
                let sorted_visited = Self::sorted_brandes_state_indices(&traversal);
                let mut seed = vec![0.0f64; traversal.state.len()];

                for d_idx in 0..distances.len() {
                    let dist_threshold = distances[d_idx] as f32;
                    seed.fill(0.0);

                    if closest_destination {
                        // route the full origin weight to the single nearest reachable destination
                        let mut nearest: Option<(usize, f32)> = None;
                        for &dest in dest_weights.keys() {
                            let cost = traversal.best_route_cost[dest];
                            if cost > dist_threshold {
                                continue;
                            }
                            if nearest.is_none_or(|(_, c)| cost < c) {
                                nearest = Some((dest, cost));
                            }
                        }
                        match nearest {
                            Some((dest, _)) => seed[dest] = o_weight,
                            None => continue,
                        }
                    } else {
                        // single-constrained gravity: distribute o_weight in proportion to
                        // W_d * decay(c), normalised over reachable destinations.
                        let mut denom = 0.0f64;
                        for (&dest, &w_d) in &dest_weights {
                            let cost = traversal.best_route_cost[dest];
                            if cost > dist_threshold {
                                continue;
                            }
                            let p = cost / dist_threshold;
                            denom += w_d * decay(cost, p) as f64;
                        }
                        if denom <= 0.0 {
                            continue;
                        }
                        for (&dest, &w_d) in &dest_weights {
                            let cost = traversal.best_route_cost[dest];
                            if cost > dist_threshold {
                                continue;
                            }
                            let p = cost / dist_threshold;
                            seed[dest] = o_weight * (w_d * decay(cost, p) as f64) / denom;
                        }
                    }

                    let seed_refs: [&[f64]; 1] = [seed.as_slice()];
                    Self::brandes_backprop_multi(
                        &traversal,
                        &sorted_visited,
                        src_idx,
                        &seed_refs,
                        |state| state.route_cost <= dist_threshold,
                        |inter_node_idx, credits| {
                            let credit = credits[0];
                            if credit > 0.0 {
                                res.betweenness_metrics[0].1.metric[d_idx][inter_node_idx]
                                    .fetch_add(credit, AtomicOrdering::Relaxed);
                            }
                        },
                    );
                }
            });

            res
        });

        Ok(result)
    }
}
