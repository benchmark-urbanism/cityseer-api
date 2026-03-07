use crate::common;

use crate::common::MetricResult;
use crate::common::WALKING_SPEED;
use crate::graph::{EdgeVisit, NetworkStructure, NodeVisit};
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
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering as AtomicOrdering;

const ANGULAR_ROUTE_TIE_BREAK_FACTOR: f32 = 1e-6;
const ANGULAR_TIE_EPSILON: f32 = 1e-4;

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
// Betweenness result types (used by betweenness_od_shortest)
// =========================================================================

#[pyclass]
pub struct BetweennessShortestResult {
    #[pyo3(get)]
    distances: Vec<u32>,
    #[pyo3(get)]
    node_keys_py: Vec<Py<PyAny>>,
    #[pyo3(get)]
    node_indices: Vec<usize>,

    node_betweenness_vec: MetricResult,
    node_betweenness_beta_vec: MetricResult,

    #[pyo3(get)]
    pub reachability_totals: Vec<u32>,
    #[pyo3(get)]
    pub sampled_source_count: u32,
}

impl BetweennessShortestResult {
    pub fn new(
        distances: Vec<u32>,
        node_keys_py: Vec<Py<PyAny>>,
        node_indices: Vec<usize>,
        capacity: usize,
        init_val: f32,
    ) -> Self {
        BetweennessShortestResult {
            distances: distances.clone(),
            node_keys_py,
            node_indices: node_indices.clone(),
            node_betweenness_vec: MetricResult::new(&distances, capacity, init_val),
            node_betweenness_beta_vec: MetricResult::new(&distances, capacity, init_val),
            reachability_totals: Vec::new(),
            sampled_source_count: 0,
        }
    }
}

#[pymethods]
impl BetweennessShortestResult {
    #[getter]
    pub fn node_betweenness(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_betweenness_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn node_betweenness_beta(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_betweenness_beta_vec
            .load_compact(&self.node_indices)
    }
}

// =========================================================================
// Combined centrality result types (closeness + betweenness from single Dijkstra)
// =========================================================================

#[pyclass]
pub struct CentralityShortestResult {
    #[pyo3(get)]
    distances: Vec<u32>,
    #[pyo3(get)]
    node_keys_py: Vec<Py<PyAny>>,
    #[pyo3(get)]
    node_indices: Vec<usize>,

    // Closeness fields
    node_density_vec: MetricResult,
    node_farness_vec: MetricResult,
    node_cycles_vec: MetricResult,
    node_harmonic_vec: MetricResult,
    node_beta_vec: MetricResult,

    // Betweenness fields
    node_betweenness_vec: MetricResult,
    node_betweenness_beta_vec: MetricResult,

    #[pyo3(get)]
    pub reachability_totals: Vec<u32>,
    #[pyo3(get)]
    pub sampled_source_count: u32,
}

impl CentralityShortestResult {
    pub fn new(
        distances: Vec<u32>,
        node_keys_py: Vec<Py<PyAny>>,
        node_indices: Vec<usize>,
        capacity: usize,
        init_val: f32,
    ) -> Self {
        CentralityShortestResult {
            distances: distances.clone(),
            node_keys_py,
            node_indices: node_indices.clone(),
            node_density_vec: MetricResult::new(&distances, capacity, init_val),
            node_farness_vec: MetricResult::new(&distances, capacity, init_val),
            node_cycles_vec: MetricResult::new(&distances, capacity, init_val),
            node_harmonic_vec: MetricResult::new(&distances, capacity, init_val),
            node_beta_vec: MetricResult::new(&distances, capacity, init_val),
            node_betweenness_vec: MetricResult::new(&distances, capacity, init_val),
            node_betweenness_beta_vec: MetricResult::new(&distances, capacity, init_val),
            reachability_totals: Vec::new(),
            sampled_source_count: 0,
        }
    }
}

#[pymethods]
impl CentralityShortestResult {
    #[getter]
    pub fn node_density(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_density_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn node_farness(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_farness_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn node_cycles(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_cycles_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn node_harmonic(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_harmonic_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn node_beta(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_beta_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn node_betweenness(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_betweenness_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn node_betweenness_beta(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_betweenness_beta_vec
            .load_compact(&self.node_indices)
    }
}

#[pyclass]
pub struct CentralitySimplestResult {
    #[pyo3(get)]
    distances: Vec<u32>,
    #[pyo3(get)]
    node_keys_py: Vec<Py<PyAny>>,
    #[pyo3(get)]
    node_indices: Vec<usize>,

    // Closeness fields (no cycles or beta for simplest)
    node_density_vec: MetricResult,
    node_farness_vec: MetricResult,
    node_harmonic_vec: MetricResult,

    // Betweenness fields
    node_betweenness_vec: MetricResult,

    #[pyo3(get)]
    pub reachability_totals: Vec<u32>,
    #[pyo3(get)]
    pub sampled_source_count: u32,
}

impl CentralitySimplestResult {
    pub fn new(
        distances: Vec<u32>,
        node_keys_py: Vec<Py<PyAny>>,
        node_indices: Vec<usize>,
        capacity: usize,
        init_val: f32,
    ) -> Self {
        CentralitySimplestResult {
            distances: distances.clone(),
            node_keys_py,
            node_indices: node_indices.clone(),
            node_density_vec: MetricResult::new(&distances, capacity, init_val),
            node_farness_vec: MetricResult::new(&distances, capacity, init_val),
            node_harmonic_vec: MetricResult::new(&distances, capacity, init_val),
            node_betweenness_vec: MetricResult::new(&distances, capacity, init_val),
            reachability_totals: Vec::new(),
            sampled_source_count: 0,
        }
    }
}

#[pymethods]
impl CentralitySimplestResult {
    #[getter]
    pub fn node_density(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_density_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn node_farness(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_farness_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn node_harmonic(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_harmonic_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn node_betweenness(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.node_betweenness_vec.load_compact(&self.node_indices)
    }
}

#[pyclass]
pub struct CentralitySegmentResult {
    #[pyo3(get)]
    distances: Vec<u32>,
    #[pyo3(get)]
    node_keys_py: Vec<Py<PyAny>>,
    #[pyo3(get)]
    node_indices: Vec<usize>,

    segment_density_vec: MetricResult,
    segment_harmonic_vec: MetricResult,
    segment_beta_vec: MetricResult,
    segment_betweenness_vec: MetricResult,
}

impl CentralitySegmentResult {
    pub fn new(
        distances: Vec<u32>,
        node_keys_py: Vec<Py<PyAny>>,
        node_indices: Vec<usize>,
        capacity: usize,
        init_val: f32,
    ) -> Self {
        CentralitySegmentResult {
            distances: distances.clone(),
            node_keys_py,
            node_indices: node_indices.clone(),
            segment_density_vec: MetricResult::new(&distances, capacity, init_val),
            segment_harmonic_vec: MetricResult::new(&distances, capacity, init_val),
            segment_beta_vec: MetricResult::new(&distances, capacity, init_val),
            segment_betweenness_vec: MetricResult::new(&distances, capacity, init_val),
        }
    }
}

#[pymethods]
impl CentralitySegmentResult {
    #[getter]
    pub fn segment_density(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.segment_density_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn segment_harmonic(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.segment_harmonic_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn segment_beta(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.segment_beta_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn segment_betweenness(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.segment_betweenness_vec
            .load_compact(&self.node_indices)
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
    source_eligible: Vec<bool>,
    is_source_indexed: bool,
    n_sources: usize,
    n_live: usize,
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

    fn dijkstra_brandes_angular(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        endpoint_slots: &[SmallVec<[String; 2]>],
    ) -> BrandesTraversal {
        let node_count = self.node_bound();
        let state_count = node_count * 2;
        let mut states = (0..state_count)
            .map(|state_idx| BrandesTraversalState::new(state_idx / 2))
            .collect::<Vec<_>>();
        let mut visited_state_indices = Vec::new();
        let mut best_route_cost = vec![f32::INFINITY; node_count];
        let mut best_agg_seconds = vec![f32::INFINITY; node_count];

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
                let candidate_route = states[state_idx].route_cost
                    + edge_payload.angle_sum
                    + (ANGULAR_ROUTE_TIE_BREAK_FACTOR * edge_payload.length);

                let improved =
                    candidate_route + ANGULAR_TIE_EPSILON < states[next_state_idx].route_cost;
                let tied = (candidate_route - states[next_state_idx].route_cost).abs()
                    <= ANGULAR_TIE_EPSILON;

                if improved {
                    states[next_state_idx].preds.clear();
                    states[next_state_idx].sigma = states[state_idx].sigma;
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
                } else {
                    continue;
                }

                let next_node_idx = states[next_state_idx].node_idx;
                if candidate_route + ANGULAR_TIE_EPSILON < best_route_cost[next_node_idx] {
                    best_route_cost[next_node_idx] = candidate_route;
                    best_agg_seconds[next_node_idx] = candidate_seconds;
                } else if (candidate_route - best_route_cost[next_node_idx]).abs()
                    <= ANGULAR_TIE_EPSILON
                {
                    best_agg_seconds[next_node_idx] =
                        best_agg_seconds[next_node_idx].min(candidate_seconds);
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
            if (state.route_cost - best_route_cost).abs() <= ANGULAR_TIE_EPSILON {
                best_state_indices.push(state_idx);
            }
        }

        best_state_indices
    }

    fn brandes_backprop_with_beta<FInclude, FCredit>(
        traversal: &BrandesTraversal,
        sorted_state_indices: &[usize],
        src_node_idx: usize,
        target_seed: &[f64],
        target_seed_beta: &[f64],
        include_state: FInclude,
        mut on_credit: FCredit,
    ) where
        FInclude: Fn(&BrandesTraversalState) -> bool,
        FCredit: FnMut(usize, f64, f64),
    {
        let mut delta = vec![0.0f64; traversal.state.len()];
        let mut delta_beta = vec![0.0f64; traversal.state.len()];

        for &state_idx in sorted_state_indices {
            let state = &traversal.state[state_idx];
            if !include_state(state) {
                continue;
            }
            let sigma_w = state.sigma;
            if sigma_w == 0.0 {
                continue;
            }

            let dependency = target_seed[state_idx] + delta[state_idx];
            let dependency_beta = target_seed_beta[state_idx] + delta_beta[state_idx];
            if dependency == 0.0 && dependency_beta == 0.0 {
                continue;
            }

            for &pred_state_idx in &state.preds {
                let sigma_v = traversal.state[pred_state_idx].sigma;
                if sigma_v == 0.0 {
                    continue;
                }
                let factor = sigma_v / sigma_w;
                delta[pred_state_idx] += factor * dependency;
                delta_beta[pred_state_idx] += factor * dependency_beta;
            }

            if state.node_idx == src_node_idx {
                continue;
            }
            let credit = dependency - target_seed[state_idx];
            let credit_beta = dependency_beta - target_seed_beta[state_idx];
            if credit > 0.0 || credit_beta > 0.0 {
                on_credit(state.node_idx, credit.max(0.0), credit_beta.max(0.0));
            }
        }
    }

    fn scale_metric_results(
        metric_results: &[&MetricResult],
        threshold_count: usize,
        node_indices: &[usize],
        scale: f64,
    ) {
        if scale == 1.0 {
            return;
        }
        for d_idx in 0..threshold_count {
            for &node_idx in node_indices {
                for metric_result in metric_results {
                    let raw = metric_result.metric[d_idx][node_idx].load(AtomicOrdering::Relaxed);
                    metric_result.metric[d_idx][node_idx]
                        .store(raw * scale, AtomicOrdering::Relaxed);
                }
            }
        }
    }

    #[inline]
    fn validate_node_exists(&self, node_idx: usize) -> PyResult<()> {
        if node_idx >= self.node_bound()
            || self.graph.node_weight(NodeIndex::new(node_idx)).is_none()
        {
            return Err(exceptions::PyValueError::new_err(format!(
                "node index {} does not exist in the graph",
                node_idx
            )));
        }
        Ok(())
    }

    #[inline]
    fn validate_source_indices_exist(&self, source_indices: &[usize]) -> PyResult<()> {
        for &src_idx in source_indices {
            self.validate_node_exists(src_idx)?;
        }
        Ok(())
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
        source_indices: Option<Vec<usize>>,
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
        if source_indices.is_some() && sampling_weights.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "source_indices and sampling_weights are mutually exclusive",
            ));
        }
        if let Some(ref indices) = source_indices {
            self.validate_source_indices_exist(indices)?;
        }

        let n = self.node_bound();
        let n_live = node_indices
            .iter()
            .filter(|&&idx| self.is_node_live_unchecked(idx))
            .count();
        let is_source_indexed = source_indices.is_some();
        let sources = source_indices.unwrap_or_else(|| node_indices.to_vec());
        let n_sources = sources.len();
        let source_eligible = if is_source_indexed {
            let mut eligible = vec![false; n];
            for &idx in &sources {
                eligible[idx] = true;
            }
            eligible
        } else {
            let mut eligible = vec![false; n];
            for &idx in node_indices {
                if self.is_node_live_unchecked(idx) {
                    eligible[idx] = true;
                }
            }
            eligible
        };
        let sample_randoms = if sample_probability.is_some() && !is_source_indexed {
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
            source_eligible,
            is_source_indexed,
            n_sources,
            n_live,
        })
    }

    #[inline]
    fn sample_source_weight(
        &self,
        src_idx: usize,
        sample_probability: Option<f32>,
        sampling_weights: Option<&[f32]>,
        sample_randoms: &[f32],
        is_source_indexed: bool,
        sampled_source_count: &AtomicU32,
    ) -> Option<f32> {
        let mut wt = self.get_node_weight_unchecked(src_idx);
        if is_source_indexed {
            if let Some(prob) = sample_probability {
                wt /= prob;
            }
            sampled_source_count.fetch_add(1, AtomicOrdering::Relaxed);
            return Some(wt);
        }
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
            wt /= p;
        }
        Some(wt)
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
            for edge_ref in self
                .graph
                .edges_directed(current_node_index, Direction::Incoming)
            {
                let nb_nd_idx = edge_ref.source();
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
                    self.edge_travel_seconds(nb_idx, node_idx, edge_payload, speed_m_s, true);
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

                let improved =
                    candidate_metric + ANGULAR_TIE_EPSILON < states[next_state_idx].route_metric;
                let tied = (candidate_metric - states[next_state_idx].route_metric).abs()
                    <= ANGULAR_TIE_EPSILON;

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
                    let node_improved =
                        candidate_simpl + ANGULAR_TIE_EPSILON < next_visit.simpl_dist;
                    let node_tied =
                        (candidate_simpl - next_visit.simpl_dist).abs() <= ANGULAR_TIE_EPSILON;
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

    fn dijkstra_brandes_shortest(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        tolerance: f32,
    ) -> BrandesTraversal {
        let node_count = self.node_bound();
        let mut states = (0..node_count)
            .map(BrandesTraversalState::new)
            .collect::<Vec<_>>();
        let mut visited_state_indices = Vec::new();
        let mut best_route_cost = vec![f32::INFINITY; node_count];
        let mut best_agg_seconds = vec![f32::INFINITY; node_count];

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
            for edge_ref in self
                .graph
                .edges_directed(current_node_index, Direction::Incoming)
            {
                let nb_nd_idx = edge_ref.source();
                let nb_idx = nb_nd_idx.index();
                let edge_payload = edge_ref.weight();
                if nb_idx == current_node_idx {
                    continue;
                }
                let edge_seconds = self.edge_travel_seconds(
                    nb_idx,
                    current_node_idx,
                    edge_payload,
                    speed_m_s,
                    true,
                );
                let candidate_seconds = states[state_idx].agg_seconds + edge_seconds;
                if candidate_seconds > max_seconds as f32 {
                    continue;
                }
                if states[nb_idx].visited {
                    continue;
                }
                let candidate_route = candidate_seconds * speed_m_s;
                let improved = candidate_seconds < states[nb_idx].agg_seconds;
                let tied = candidate_seconds <= states[nb_idx].agg_seconds * (1.0 + tolerance);

                if improved {
                    if candidate_seconds < states[nb_idx].agg_seconds * (1.0 - tolerance) {
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
                } else {
                    continue;
                }

                best_route_cost[nb_idx] = states[nb_idx].route_cost;
                best_agg_seconds[nb_idx] = states[nb_idx].agg_seconds;
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

    #[pyo3(signature = (src_idx, max_seconds, speed_m_s))]
    pub fn dijkstra_tree_segment(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
    ) -> PyResult<(Vec<usize>, Vec<usize>, Vec<NodeVisit>, Vec<EdgeVisit>)> {
        self.validate_dijkstra_inputs(src_idx, speed_m_s)?;
        let mut tree_map = vec![NodeVisit::new(); self.node_bound()];
        let mut edge_map = vec![EdgeVisit::new(); self.edge_bound()];
        let mut visited_nodes = Vec::new();
        let mut visited_edges = Vec::new();
        tree_map[src_idx].short_dist = 0.0;
        tree_map[src_idx].agg_seconds = 0.0;
        tree_map[src_idx].discovered = true;
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
            // Use Incoming direction to discover neighbors via edges pointing INTO current node.
            // Edge Y→X (neighbor→current) gives us the distance FROM Y TO X, which is what we
            // want for reversed/flipped aggregation where we accumulate TO the target node.
            for edge_ref in self
                .graph
                .edges_directed(current_node_index, Direction::Incoming)
            {
                let nb_nd_idx = edge_ref.source();
                let edge_idx = edge_ref.id();
                // Use incoming edge directly - it has the correct distance for Y→X
                let edge_payload = edge_ref.weight();
                if nb_nd_idx.index() == node_idx {
                    visited_edges.push(edge_idx.index());
                    edge_map[edge_idx.index()].visited = true;
                    edge_map[edge_idx.index()].start_nd_idx = Some(node_idx);
                    edge_map[edge_idx.index()].end_nd_idx = Some(nb_nd_idx.index());
                    edge_map[edge_idx.index()].edge_idx = Some(edge_payload.edge_idx);
                    continue;
                }
                if tree_map[nb_nd_idx.index()].visited {
                    continue;
                }
                visited_edges.push(edge_idx.index());
                edge_map[edge_idx.index()].visited = true;
                edge_map[edge_idx.index()].start_nd_idx = Some(node_idx);
                edge_map[edge_idx.index()].end_nd_idx = Some(nb_nd_idx.index());
                edge_map[edge_idx.index()].edge_idx = Some(edge_payload.edge_idx);

                let edge_seconds = self.edge_travel_seconds(
                    nb_nd_idx.index(),
                    node_idx,
                    edge_payload,
                    speed_m_s,
                    true,
                );
                let total_seconds = tree_map[node_idx].agg_seconds + edge_seconds;
                if total_seconds > max_seconds as f32 {
                    continue;
                }
                if total_seconds < tree_map[nb_nd_idx.index()].agg_seconds {
                    let origin_seg = if node_idx == src_idx {
                        edge_idx.index()
                    } else {
                        tree_map[node_idx].origin_seg.expect(
                            "Origin segment must exist for non-source node in segment path update",
                        )
                    };
                    tree_map[nb_nd_idx.index()].short_dist = total_seconds * speed_m_s;
                    tree_map[nb_nd_idx.index()].agg_seconds = total_seconds;
                    tree_map[nb_nd_idx.index()].pred = Some(node_idx);
                    tree_map[nb_nd_idx.index()].origin_seg = Some(origin_seg);
                    tree_map[nb_nd_idx.index()].last_seg = Some(edge_idx.index());
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        metric: total_seconds,
                    });
                }
            }
        }
        Ok((visited_nodes, visited_edges, tree_map, edge_map))
    }

    // =========================================================================
    // Combined centrality (closeness + betweenness from single Dijkstra)
    // =========================================================================

    /// Compute node centrality using shortest paths with a single Dijkstra per source.
    ///
    /// When both `compute_closeness` and `compute_betweenness` are true, a single
    /// Brandes-style Dijkstra traversal per source produces the data for both
    /// closeness accumulation and betweenness backpropagation, halving computation
    /// time compared to calling `closeness_shortest` and `betweenness_shortest`
    /// separately.
    #[pyo3(signature = (
        distances=None,
        betas=None,
        minutes=None,
        compute_closeness=None,
        compute_betweenness=None,
        min_threshold_wt=None,
        speed_m_s=None,
        tolerance=None,
        sample_probability=None,
        sampling_weights=None,
        random_seed=None,
        source_indices=None,
        pbar_disabled=None
    ))]
    pub fn centrality_shortest(
        &self,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        minutes: Option<Vec<f32>>,
        compute_closeness: Option<bool>,
        compute_betweenness: Option<bool>,
        min_threshold_wt: Option<f32>,
        speed_m_s: Option<f32>,
        tolerance: Option<f32>,
        sample_probability: Option<f32>,
        sampling_weights: Option<Vec<f32>>,
        random_seed: Option<u64>,
        source_indices: Option<Vec<usize>>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralityShortestResult> {
        let compute_closeness = compute_closeness.unwrap_or(true);
        let compute_betweenness = compute_betweenness.unwrap_or(true);
        if !compute_closeness && !compute_betweenness {
            return Err(exceptions::PyValueError::new_err(
                "Either or both closeness and betweenness flags is required, but both parameters are False.",
            ));
        }
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let tolerance = tolerance.unwrap_or(0.0);
        let (distances, betas, seconds) = common::pair_distances_betas_time(
            speed_m_s,
            distances,
            betas,
            minutes,
            min_threshold_wt,
        )?;
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
            source_indices,
            &node_indices,
        )?;
        let n = self.node_bound();
        let mut res = CentralityShortestResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
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
                if !sampling_plan.source_eligible[*src_idx] {
                    return;
                }

                let Some(wt) = self.sample_source_weight(
                    *src_idx,
                    sampling_plan.sample_probability,
                    sampling_plan.sampling_weights.as_deref(),
                    &sampling_plan.sample_randoms,
                    sampling_plan.is_source_indexed,
                    &sampled_source_count,
                ) else {
                    return;
                };

                let traversal = self.dijkstra_brandes_shortest(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    tolerance,
                );

                // --- Closeness accumulation ---
                if compute_closeness {
                    for &to_idx in &traversal.reached_node_indices {
                        if to_idx == *src_idx {
                            continue;
                        }
                        if !traversal.best_agg_seconds[to_idx].is_finite() {
                            continue;
                        }
                        let node_state = &traversal.state[to_idx];
                        let cycle_score = node_state.preds.len().saturating_sub(1) as f32;
                        // Track reachability per distance when sampling
                        if sampling_plan.sample_probability.is_some()
                            || sampling_plan.is_source_indexed
                        {
                            for i in 0..distances.len() {
                                if traversal.best_route_cost[to_idx] <= distances[i] as f32 {
                                    source_reachability_totals[i]
                                        .fetch_add(1, AtomicOrdering::Relaxed);
                                }
                            }
                        }
                        // Flipped aggregation: accumulate to target (to_idx) not source.
                        for i in 0..distances.len() {
                            let distance = distances[i];
                            let beta = betas[i];
                            if traversal.best_route_cost[to_idx] <= distance as f32 {
                                res.node_density_vec.metric[i][to_idx]
                                    .fetch_add(wt as f64, AtomicOrdering::Relaxed);
                                res.node_farness_vec.metric[i][to_idx].fetch_add(
                                    (traversal.best_route_cost[to_idx] * wt) as f64,
                                    AtomicOrdering::Relaxed,
                                );
                                res.node_cycles_vec.metric[i][to_idx]
                                    .fetch_add((cycle_score * wt) as f64, AtomicOrdering::Relaxed);
                                res.node_harmonic_vec.metric[i][to_idx].fetch_add(
                                    ((1.0 / traversal.best_route_cost[to_idx]) * wt) as f64,
                                    AtomicOrdering::Relaxed,
                                );
                                res.node_beta_vec.metric[i][to_idx].fetch_add(
                                    ((-beta * traversal.best_route_cost[to_idx]).exp() * wt) as f64,
                                    AtomicOrdering::Relaxed,
                                );
                            }
                        }
                    }
                }

                // --- Betweenness backpropagation ---
                if compute_betweenness {
                    let sorted_state_indices = Self::sorted_brandes_state_indices(&traversal);
                    let mut target_seed = vec![0.0f64; traversal.state.len()];
                    let mut target_seed_beta = vec![0.0f64; traversal.state.len()];

                    for d_idx in 0..distances.len() {
                        let dist_threshold = distances[d_idx] as f32;
                        let beta = betas[d_idx] as f64;
                        target_seed.fill(0.0);
                        target_seed_beta.fill(0.0);

                        for &to_idx in &traversal.reached_node_indices {
                            if to_idx == *src_idx {
                                continue;
                            }
                            if traversal.best_route_cost[to_idx] > dist_threshold {
                                continue;
                            }

                            let pair_count = if sampling_plan.source_eligible[to_idx] {
                                0.5
                            } else {
                                1.0
                            };
                            let pair_beta = pair_count
                                * (-beta * traversal.best_route_cost[to_idx] as f64).exp();
                            target_seed[to_idx] += pair_count;
                            target_seed_beta[to_idx] += pair_beta;
                        }

                        Self::brandes_backprop_with_beta(
                            &traversal,
                            &sorted_state_indices,
                            *src_idx,
                            &target_seed,
                            &target_seed_beta,
                            |state| state.route_cost <= dist_threshold,
                            |inter_node_idx, credit, credit_beta| {
                                if credit > 0.0 {
                                    res.node_betweenness_vec.metric[d_idx][inter_node_idx]
                                        .fetch_add(credit * wt as f64, AtomicOrdering::Relaxed);
                                }
                                if credit_beta > 0.0 {
                                    res.node_betweenness_beta_vec.metric[d_idx][inter_node_idx]
                                        .fetch_add(
                                            credit_beta * wt as f64,
                                            AtomicOrdering::Relaxed,
                                        );
                                }
                            },
                        );
                    }
                }
            });

            // Closeness sampling metadata
            if sampling_plan.sample_probability.is_some() || sampling_plan.is_source_indexed {
                res.sampled_source_count = sampled_source_count.load(AtomicOrdering::Relaxed);
                res.reachability_totals = source_reachability_totals
                    .iter()
                    .map(|a| a.load(AtomicOrdering::Relaxed))
                    .collect();
            }

            // Betweenness post-hoc scaling.
            // Pair weighting is already handled in-loop (0.5 for source-eligible
            // targets, 1.0 otherwise), so no /2 is required here. For source-indexed
            // without sampling, scale by n_live / n_sources to extrapolate from
            // the subset.
            if compute_betweenness {
                let scale = if sampling_plan.is_source_indexed {
                    if sampling_plan.sample_probability.is_some() {
                        1.0
                    } else {
                        sampling_plan.n_live as f64 / sampling_plan.n_sources as f64
                    }
                } else {
                    1.0
                };
                Self::scale_metric_results(
                    &[&res.node_betweenness_vec, &res.node_betweenness_beta_vec],
                    distances.len(),
                    &node_indices,
                    scale,
                );
            }

            res
        });

        Ok(result)
    }

    /// Compute node centrality using simplest (angular) paths on the dual graph.
    ///
    /// Angular routing is evaluated on two directed states per segment. Each
    /// source segment seeds both orientations into a single Brandes traversal.
    #[pyo3(signature = (
        distances=None,
        betas=None,
        minutes=None,
        compute_closeness=None,
        compute_betweenness=None,
        min_threshold_wt=None,
        speed_m_s=None,
        angular_scaling_unit=None,
        farness_scaling_offset=None,
        sample_probability=None,
        sampling_weights=None,
        random_seed=None,
        source_indices=None,
        pbar_disabled=None
    ))]
    pub fn centrality_simplest(
        &self,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        minutes: Option<Vec<f32>>,
        compute_closeness: Option<bool>,
        compute_betweenness: Option<bool>,
        min_threshold_wt: Option<f32>,
        speed_m_s: Option<f32>,
        angular_scaling_unit: Option<f32>,
        farness_scaling_offset: Option<f32>,
        sample_probability: Option<f32>,
        sampling_weights: Option<Vec<f32>>,
        random_seed: Option<u64>,
        source_indices: Option<Vec<usize>>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralitySimplestResult> {
        self.validate_dual_for_angular("centrality_simplest")?;
        let compute_closeness = compute_closeness.unwrap_or(true);
        let compute_betweenness = compute_betweenness.unwrap_or(true);
        if !compute_closeness && !compute_betweenness {
            return Err(exceptions::PyValueError::new_err(
                "Either or both closeness and betweenness flags is required, but both parameters are False.",
            ));
        }
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let angular_scaling_unit = angular_scaling_unit.unwrap_or(180.0);
        let farness_scaling_offset = farness_scaling_offset.unwrap_or(1.0);
        let (distances, _betas, seconds) = common::pair_distances_betas_time(
            speed_m_s,
            distances,
            betas,
            minutes,
            min_threshold_wt,
        )?;
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
            source_indices,
            &node_indices,
        )?;
        let n = self.node_bound();
        let mut res = CentralitySimplestResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
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
                if !sampling_plan.source_eligible[*src_idx] {
                    return;
                }

                let Some(wt) = self.sample_source_weight(
                    *src_idx,
                    sampling_plan.sample_probability,
                    sampling_plan.sampling_weights.as_deref(),
                    &sampling_plan.sample_randoms,
                    sampling_plan.is_source_indexed,
                    &sampled_source_count,
                ) else {
                    return;
                };

                let traversal = self.dijkstra_brandes_angular(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    &angular_endpoint_slots,
                );

                // --- Closeness accumulation ---
                if compute_closeness {
                    for &to_idx in &traversal.reached_node_indices {
                        if to_idx == *src_idx {
                            continue;
                        }
                        let best_simpl_dist = traversal.best_route_cost[to_idx];
                        let best_agg_seconds = traversal.best_agg_seconds[to_idx];
                        if !best_simpl_dist.is_finite() || !best_agg_seconds.is_finite() {
                            continue;
                        }
                        // Track reachability per time threshold when sampling
                        if sampling_plan.sample_probability.is_some()
                            || sampling_plan.is_source_indexed
                        {
                            for i in 0..seconds.len() {
                                if best_agg_seconds <= seconds[i] as f32 {
                                    source_reachability_totals[i]
                                        .fetch_add(1, AtomicOrdering::Relaxed);
                                }
                            }
                        }
                        // Flipped aggregation: accumulate to target (to_idx) not source.
                        for i in 0..seconds.len() {
                            let sec = seconds[i];
                            if best_agg_seconds <= sec as f32 {
                                res.node_density_vec.metric[i][to_idx]
                                    .fetch_add(wt as f64, AtomicOrdering::Relaxed);
                                let far_ang = farness_scaling_offset
                                    + (best_simpl_dist / angular_scaling_unit);
                                res.node_farness_vec.metric[i][to_idx]
                                    .fetch_add((far_ang * wt) as f64, AtomicOrdering::Relaxed);
                                let harm_ang = 1.0 + (best_simpl_dist / angular_scaling_unit);
                                res.node_harmonic_vec.metric[i][to_idx].fetch_add(
                                    ((1.0 / harm_ang) * wt) as f64,
                                    AtomicOrdering::Relaxed,
                                );
                            }
                        }
                    }
                }

                // --- Betweenness backpropagation ---
                if compute_betweenness {
                    let sorted_state_indices = Self::sorted_brandes_state_indices(&traversal);
                    let mut target_seed = vec![0.0f64; traversal.state.len()];
                    let target_seed_beta = vec![0.0f64; traversal.state.len()];

                    for d_idx in 0..seconds.len() {
                        let sec_threshold = seconds[d_idx] as f32;
                        target_seed.fill(0.0);

                        // Seed one target per reachable segment, split across the
                        // best terminal orientation states by sigma. This matches the
                        // shortest-path half-pair weighting for source-eligible
                        // targets while avoiding double-counting both states.
                        for &to_idx in &traversal.reached_node_indices {
                            if to_idx == *src_idx {
                                continue;
                            }
                            let best_state_indices =
                                Self::best_angular_target_states(&traversal, to_idx, sec_threshold);
                            if best_state_indices.is_empty() {
                                continue;
                            }
                            let pair_count = if sampling_plan.source_eligible[to_idx] {
                                0.5
                            } else {
                                1.0
                            };
                            let sigma_total: f64 = best_state_indices
                                .iter()
                                .map(|&state_idx| traversal.state[state_idx].sigma)
                                .sum();
                            if sigma_total == 0.0 {
                                continue;
                            }
                            for &state_idx in &best_state_indices {
                                target_seed[state_idx] +=
                                    pair_count * (traversal.state[state_idx].sigma / sigma_total);
                            }
                        }

                        Self::brandes_backprop_with_beta(
                            &traversal,
                            &sorted_state_indices,
                            *src_idx,
                            &target_seed,
                            &target_seed_beta,
                            |state| state.agg_seconds <= sec_threshold,
                            |inter_node_idx, credit, _credit_beta| {
                                if credit > 0.0 {
                                    res.node_betweenness_vec.metric[d_idx][inter_node_idx]
                                        .fetch_add(credit * wt as f64, AtomicOrdering::Relaxed);
                                }
                            },
                        );
                    }
                }
            });

            // Closeness sampling metadata
            if sampling_plan.sample_probability.is_some() || sampling_plan.is_source_indexed {
                res.sampled_source_count = sampled_source_count.load(AtomicOrdering::Relaxed);
                res.reachability_totals = source_reachability_totals
                    .iter()
                    .map(|a| a.load(AtomicOrdering::Relaxed))
                    .collect();
            }

            // Betweenness post-hoc scaling (pair weighting handled in-loop, see
            // centrality_shortest).
            if compute_betweenness {
                let scale = if sampling_plan.is_source_indexed {
                    if sampling_plan.sample_probability.is_some() {
                        1.0
                    } else {
                        sampling_plan.n_live as f64 / sampling_plan.n_sources as f64
                    }
                } else {
                    1.0
                };
                Self::scale_metric_results(
                    &[&res.node_betweenness_vec],
                    seconds.len(),
                    &node_indices,
                    scale,
                );
            }

            res
        });

        Ok(result)
    }

    #[pyo3(signature = (
        distances=None,
        betas=None,
        minutes=None,
        compute_closeness=None,
        compute_betweenness=None,
        min_threshold_wt=None,
        speed_m_s=None,
        pbar_disabled=None
    ))]
    pub fn segment_centrality(
        &self,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        minutes: Option<Vec<f32>>,
        compute_closeness: Option<bool>,
        compute_betweenness: Option<bool>,
        min_threshold_wt: Option<f32>,
        speed_m_s: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralitySegmentResult> {
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, betas, seconds) = common::pair_distances_betas_time(
            speed_m_s,
            distances,
            betas,
            minutes,
            min_threshold_wt,
        )?;
        let max_walk_seconds = *seconds
            .iter()
            .max()
            .expect("Seconds vector should not be empty");
        let compute_closeness = compute_closeness.unwrap_or(true);
        let compute_betweenness = compute_betweenness.unwrap_or(true);
        if !compute_closeness && !compute_betweenness {
            return Err(exceptions::PyValueError::new_err(
            "Either or both closeness and betweenness flags is required, but both parameters are False.",
        ));
        }

        let node_keys_py = self.node_keys_py(py);
        let node_indices = self.node_indices();
        let res = CentralitySegmentResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
            self.node_bound(),
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

                let (visited_nodes, visited_edges, tree_map, edge_map) = self
                    .dijkstra_tree_segment(*src_idx, max_walk_seconds, speed_m_s)
                    .expect("pre-validated Dijkstra inputs");
                for edge_idx in visited_edges.iter() {
                    let edge_visit = &edge_map[*edge_idx];
                    let start_node_idx = edge_visit
                        .start_nd_idx
                        .expect("Visited edge must have start node index");
                    let end_node_idx = edge_visit
                        .end_nd_idx
                        .expect("Visited edge must have end node index");
                    let edge_payload_idx = edge_visit
                        .edge_idx
                        .expect("Visited edge must have original edge index");

                    let node_visit_n = &tree_map[start_node_idx];
                    let node_visit_m = &tree_map[end_node_idx];

                    if !node_visit_n.short_dist.is_finite() && !node_visit_m.short_dist.is_finite()
                    {
                        continue;
                    }

                    if compute_closeness {
                        let n_nearer = node_visit_n.short_dist <= node_visit_m.short_dist;
                        let (a, a_imp) = if n_nearer {
                            (node_visit_n.short_dist, node_visit_n.short_dist)
                        } else {
                            (node_visit_m.short_dist, node_visit_m.short_dist)
                        };
                        let (b, b_imp) = if n_nearer {
                            (node_visit_m.short_dist, node_visit_m.short_dist)
                        } else {
                            (node_visit_n.short_dist, node_visit_n.short_dist)
                        };

                        let edge_length = self.get_edge_length_unchecked(
                            start_node_idx,
                            end_node_idx,
                            edge_payload_idx,
                        );
                        let imp_factor = self.get_edge_impedance_unchecked(
                            start_node_idx,
                            end_node_idx,
                            edge_payload_idx,
                        );

                        let c = (edge_length + a + b) / 2.0;
                        let d = c;
                        let c_imp = a_imp + (c - a) * imp_factor;
                        let d_imp = c_imp;

                        for i in (0..distances.len()).rev() {
                            let distance_f32 = distances[i] as f32;
                            let beta = betas[i];
                            let neg_beta = -beta;
                            let inv_neg_beta = if beta != 0.0 { 1.0 / neg_beta } else { 0.0 };

                            if a < distance_f32 {
                                let mut current_c = c;
                                let mut current_c_imp = c_imp;
                                if current_c > distance_f32 {
                                    current_c = distance_f32;
                                    current_c_imp = a_imp + (distance_f32 - a) * imp_factor;
                                }
                                res.segment_density_vec.metric[i][*src_idx]
                                    .fetch_add((current_c - a) as f64, AtomicOrdering::Relaxed);

                                let seg_harm = if a_imp < 1.0 {
                                    current_c_imp.ln()
                                } else {
                                    (current_c_imp / a_imp).max(f32::EPSILON).ln()
                                };
                                res.segment_harmonic_vec.metric[i][*src_idx]
                                    .fetch_add(seg_harm as f64, AtomicOrdering::Relaxed);

                                let bet = if beta == 0.0 {
                                    current_c_imp - a_imp
                                } else {
                                    ((neg_beta * current_c_imp).exp() - (neg_beta * a_imp).exp())
                                        * inv_neg_beta
                                };
                                res.segment_beta_vec.metric[i][*src_idx]
                                    .fetch_add(bet as f64, AtomicOrdering::Relaxed);
                            }

                            if b == d {
                                continue;
                            }

                            if b <= distance_f32 {
                                let mut current_d = d;
                                let mut current_d_imp = d_imp;
                                if current_d > distance_f32 {
                                    current_d = distance_f32;
                                    current_d_imp = b_imp + (distance_f32 - b) * imp_factor;
                                }
                                res.segment_density_vec.metric[i][*src_idx]
                                    .fetch_add((current_d - b) as f64, AtomicOrdering::Relaxed);

                                let seg_harm = if b_imp < 1.0 {
                                    current_d_imp.ln()
                                } else {
                                    (current_d_imp / b_imp).max(f32::EPSILON).ln()
                                };
                                res.segment_harmonic_vec.metric[i][*src_idx]
                                    .fetch_add(seg_harm as f64, AtomicOrdering::Relaxed);

                                let bet = if beta == 0.0 {
                                    current_d_imp - b_imp
                                } else {
                                    ((neg_beta * current_d_imp).exp() - (neg_beta * b_imp).exp())
                                        * inv_neg_beta
                                };
                                res.segment_beta_vec.metric[i][*src_idx]
                                    .fetch_add(bet as f64, AtomicOrdering::Relaxed);
                            }
                        }
                    }
                }

                if compute_betweenness {
                    for to_idx in visited_nodes.iter() {
                        if to_idx <= src_idx {
                            continue;
                        }

                        let to_node_visit = &tree_map[*to_idx];
                        if !to_node_visit.short_dist.is_finite() {
                            continue;
                        }

                        let o_seg_idx = to_node_visit.origin_seg.expect(
                            "Reachable 'to' node in segment betweenness must have origin segment",
                        );
                        let l_seg_idx = to_node_visit.last_seg.expect(
                            "Reachable 'to' node in segment betweenness must have last segment",
                        );

                        let o_edge_visit = &edge_map[o_seg_idx];
                        let l_edge_visit = &edge_map[l_seg_idx];

                        let o_seg_len = self.get_edge_length_unchecked(
                            o_edge_visit
                                .start_nd_idx
                                .expect("Origin edge visit must have start node"),
                            o_edge_visit
                                .end_nd_idx
                                .expect("Origin edge visit must have end node"),
                            o_edge_visit
                                .edge_idx
                                .expect("Origin edge visit must have edge index"),
                        );
                        let l_seg_len = self.get_edge_length_unchecked(
                            l_edge_visit
                                .start_nd_idx
                                .expect("Last edge visit must have start node"),
                            l_edge_visit
                                .end_nd_idx
                                .expect("Last edge visit must have end node"),
                            l_edge_visit
                                .edge_idx
                                .expect("Last edge visit must have edge index"),
                        );

                        let min_span = to_node_visit.short_dist - o_seg_len - l_seg_len;
                        let o_1 = min_span;
                        let o_2 = min_span + o_seg_len;
                        let l_1 = min_span;
                        let l_2 = min_span + l_seg_len;

                        let mut current_pred = to_node_visit.pred;
                        while let Some(inter_idx) = current_pred {
                            if inter_idx == *src_idx {
                                break;
                            }
                            for i in (0..distances.len()).rev() {
                                let distance = distances[i];
                                let beta = betas[i];
                                if min_span <= distance as f32 {
                                    let mut o_2_snip = o_2;
                                    let mut l_2_snip = l_2;
                                    o_2_snip = o_2_snip.min(distance as f32);
                                    l_2_snip = l_2_snip.min(distance as f32);
                                    let auc = if beta == 0.0 {
                                        (o_2_snip - o_1) + (l_2_snip - l_1)
                                    } else {
                                        let neg_beta = -beta;
                                        let inv_neg_beta = 1.0 / neg_beta;
                                        ((neg_beta * o_2_snip).exp() - (neg_beta * o_1).exp())
                                            * inv_neg_beta
                                            + ((neg_beta * l_2_snip).exp() - (neg_beta * l_1).exp())
                                                * inv_neg_beta
                                    };

                                    if auc.is_finite() && auc >= 0.0 {
                                        res.segment_betweenness_vec.metric[i][inter_idx]
                                            .fetch_add(auc as f64, AtomicOrdering::Relaxed);
                                    }
                                }
                            }
                            current_pred = tree_map[inter_idx].pred;
                        }
                    }
                }
            });
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
    /// outbound OD trips. For each OD destination, backpropagates credit
    /// through all equal shortest paths, weighted by the OD flow weight
    /// and split by sigma (path count).
    #[pyo3(signature = (
        od_matrix,
        distances=None,
        betas=None,
        minutes=None,
        min_threshold_wt=None,
        speed_m_s=None,
        tolerance=None,
        pbar_disabled=None
    ))]
    pub fn betweenness_od_shortest(
        &self,
        od_matrix: &OdMatrix,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        minutes: Option<Vec<f32>>,
        min_threshold_wt: Option<f32>,
        speed_m_s: Option<f32>,
        tolerance: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<BetweennessShortestResult> {
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, betas, seconds) = common::pair_distances_betas_time(
            speed_m_s,
            distances,
            betas,
            minutes,
            min_threshold_wt,
        )?;
        let max_walk_seconds = *seconds
            .iter()
            .max()
            .expect("Seconds vector should not be empty");
        let tolerance = tolerance.unwrap_or(0.0);

        let od_map = &od_matrix.map;
        let n = self.node_bound();

        let node_keys_py = self.node_keys_py(py);
        let node_indices = self.node_indices();
        let res = BetweennessShortestResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
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
                // Skip sources with no outbound OD trips.
                let src_dests = match od_map.get(src_idx) {
                    Some(dests) => dests,
                    None => return,
                };

                let traversal = self.dijkstra_brandes_shortest(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    tolerance,
                );

                // Sort visited by distance (farthest first) for backpropagation
                let sorted_visited = Self::sorted_brandes_state_indices(&traversal);

                // Brandes backpropagation per distance threshold, weighted by OD flows
                let mut target_seed = vec![0.0f64; traversal.state.len()];
                let mut target_seed_beta = vec![0.0f64; traversal.state.len()];
                for d_idx in 0..distances.len() {
                    let dist_threshold = distances[d_idx] as f32;
                    let beta = betas[d_idx] as f64;
                    target_seed.fill(0.0);
                    target_seed_beta.fill(0.0);

                    // Seed delta at OD destinations
                    for (&dest, &od_w) in src_dests {
                        if traversal.best_route_cost[dest] > dist_threshold {
                            continue;
                        }
                        let od_beta =
                            od_w as f64 * (-beta * traversal.best_route_cost[dest] as f64).exp();
                        target_seed[dest] += od_w as f64;
                        target_seed_beta[dest] += od_beta;
                    }

                    Self::brandes_backprop_with_beta(
                        &traversal,
                        &sorted_visited,
                        *src_idx,
                        &target_seed,
                        &target_seed_beta,
                        |state| state.route_cost <= dist_threshold,
                        |inter_node_idx, credit, credit_beta| {
                            if credit > 0.0 {
                                res.node_betweenness_vec.metric[d_idx][inter_node_idx]
                                    .fetch_add(credit, AtomicOrdering::Relaxed);
                            }
                            if credit_beta > 0.0 {
                                res.node_betweenness_beta_vec.metric[d_idx][inter_node_idx]
                                    .fetch_add(credit_beta, AtomicOrdering::Relaxed);
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
