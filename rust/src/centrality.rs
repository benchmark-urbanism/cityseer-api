use crate::common;

use crate::common::MetricResult;
use crate::common::WALKING_SPEED;
use crate::graph::{EdgeVisit, NetworkStructure, NodeVisit};
use numpy::PyArray1;
use petgraph::prelude::*;
use petgraph::stable_graph::NodeIndex;
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
struct BrandesNodeState {
    visited: bool,
    discovered: bool,
    preds: SmallVec<[usize; 2]>,
    sigma: f64,
    short_dist: f32,
    simpl_dist: f32,
    cycles: f32,
    prev_in_bearing: f32,
    agg_seconds: f32,
}

impl BrandesNodeState {
    fn new() -> Self {
        Self {
            visited: false,
            discovered: false,
            preds: SmallVec::new(),
            sigma: 0.0,
            short_dist: f32::INFINITY,
            simpl_dist: f32::INFINITY,
            cycles: 0.0,
            prev_in_bearing: f32::NAN,
            agg_seconds: f32::INFINITY,
        }
    }
}

/// Lightweight node state for simplest (angular) Dijkstra with single-predecessor
/// path tracing. Unlike BrandesNodeState, this uses Option<usize> instead of
/// SmallVec for predecessors and omits sigma/cycles (unused for angular paths).
#[derive(Clone)]
struct SimplestNodeState {
    visited: bool,
    discovered: bool,
    pred: Option<usize>,
    simpl_dist: f32,
    prev_in_bearing: f32,
    agg_seconds: f32,
}

impl SimplestNodeState {
    fn new() -> Self {
        Self {
            visited: false,
            discovered: false,
            pred: None,
            simpl_dist: f32::INFINITY,
            prev_in_bearing: f32::NAN,
            agg_seconds: f32::INFINITY,
        }
    }
}

impl NetworkStructure {
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

    /// Dijkstra with multi-predecessor tracking for Brandes betweenness (shortest paths).
    ///
    /// Unlike `dijkstra_tree_shortest` which stores a single predecessor, this variant
    /// tracks ALL predecessors on shortest paths and counts the number of shortest paths
    /// (sigma) from the source to each node. This is required for standard Brandes
    /// betweenness centrality where tie-breaking must be proportional to path counts.
    ///
    /// When `compute_closeness` is true, also performs cycle detection (matching
    /// `dijkstra_tree_shortest` semantics) so that the returned state can serve both
    /// closeness accumulation and betweenness backpropagation from a single traversal.
    fn dijkstra_brandes_shortest(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        tolerance: f32,
        compute_closeness: bool,
    ) -> (Vec<usize>, Vec<BrandesNodeState>) {
        let n = self.node_bound();
        let mut state = vec![BrandesNodeState::new(); n];
        let mut visited_nodes = Vec::new();
        state[src_idx].short_dist = 0.0;
        state[src_idx].agg_seconds = 0.0;
        state[src_idx].sigma = 1.0;
        state[src_idx].discovered = true;
        let mut active = BinaryHeap::new();
        active.push(NodeDistance {
            node_idx: src_idx,
            metric: 0.0,
        });
        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
            if state[node_idx].visited {
                continue;
            }
            state[node_idx].visited = true;
            visited_nodes.push(node_idx);
            let current_node_index = NodeIndex::new(node_idx);
            for edge_ref in self
                .graph
                .edges_directed(current_node_index, Direction::Incoming)
            {
                let nb_nd_idx = edge_ref.source();
                let edge_payload = edge_ref.weight();
                let nb = nb_nd_idx.index();
                if nb == node_idx {
                    continue;
                }
                if compute_closeness {
                    // Skip back-edge to the primary (first) predecessor only, matching
                    // dijkstra_tree_shortest's single-pred semantics. Using preds.contains()
                    // would skip ALL equal-distance predecessors, preventing cycle detection
                    // when the current node is explored from nodes that are also predecessors
                    // via the tolerance branch.
                    if let Some(&primary_pred) = state[node_idx].preds.first() {
                        if nb == primary_pred {
                            continue;
                        }
                    }
                    // Cycle detection: if neighbor was already discovered via a strictly
                    // shorter path, attribute half a cycle to the nearer node.
                    // Use `discovered` (not `!preds.is_empty()`) to match dijkstra_tree_shortest
                    // semantics — preds can also be populated by equal-distance paths via the
                    // tolerance branch, which would trigger false cycle detections.
                    if state[nb].discovered {
                        if state[node_idx].agg_seconds <= state[nb].agg_seconds {
                            state[nb].cycles += 0.5;
                        } else {
                            state[node_idx].cycles += 0.5;
                        }
                    }
                }
                if state[nb].visited {
                    continue;
                }
                let edge_seconds = self.edge_travel_seconds(nb, node_idx, edge_payload, speed_m_s, true);
                let candidate = state[node_idx].agg_seconds + edge_seconds;
                if candidate > max_seconds as f32 {
                    continue;
                }
                if candidate < state[nb].agg_seconds {
                    // Shorter path found: always update distance to keep
                    // Dijkstra exploration accurate.
                    if candidate < state[nb].agg_seconds * (1.0 - tolerance) {
                        // Much shorter — old preds are outside new tolerance
                        // window, so clear them.
                        state[nb].preds.clear();
                        state[nb].sigma = state[node_idx].sigma;
                    } else {
                        // Slightly shorter — old preds are likely still within
                        // tolerance of the new distance, so keep them.
                        state[nb].sigma += state[node_idx].sigma;
                    }
                    state[nb].short_dist = candidate * speed_m_s;
                    state[nb].agg_seconds = candidate;
                    state[nb].preds.push(node_idx);
                    state[nb].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb,
                        metric: candidate,
                    });
                } else if candidate <= state[nb].agg_seconds * (1.0 + tolerance)
                    && !state[nb].visited
                {
                    // Longer but within tolerance: add predecessor only.
                    state[nb].preds.push(node_idx);
                    state[nb].sigma += state[node_idx].sigma;
                }
            }
        }
        (visited_nodes, state)
    }

    /// Single-predecessor Dijkstra for simplest (angular) paths.
    /// Uses SimplestNodeState with Option<usize> pred instead of SmallVec,
    /// since angular turn cost is path-dependent (depends on incoming bearing).
    fn dijkstra_simplest(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
    ) -> (Vec<usize>, Vec<SimplestNodeState>) {
        let n = self.node_bound();
        let mut state = vec![SimplestNodeState::new(); n];
        let mut visited_nodes = Vec::new();
        state[src_idx].simpl_dist = 0.0;
        state[src_idx].agg_seconds = 0.0;
        state[src_idx].discovered = true;
        let mut active = BinaryHeap::new();
        active.push(NodeDistance {
            node_idx: src_idx,
            metric: 0.0,
        });
        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
            if state[node_idx].visited {
                continue;
            }
            state[node_idx].visited = true;
            visited_nodes.push(node_idx);
            let current_node_index = NodeIndex::new(node_idx);
            for edge_ref in self
                .graph
                .edges_directed(current_node_index, Direction::Incoming)
            {
                let nb_nd_idx = edge_ref.source();
                let edge_payload = edge_ref.weight();
                let nb = nb_nd_idx.index();
                if nb == node_idx {
                    continue;
                }
                if state[nb].visited {
                    continue;
                }
                // Sidestepping prevention: skip if current node and neighbor
                // share a predecessor (path doubling back through sibling edge).
                if let (Some(p1), Some(p2)) = (state[node_idx].pred, state[nb].pred) {
                    if p1 == p2 {
                        continue;
                    }
                }
                // Turn angle calculation
                let mut turn = 0.0;
                if node_idx != src_idx
                    && edge_payload.out_bearing.is_finite()
                    && state[node_idx].prev_in_bearing.is_finite()
                {
                    turn = ((edge_payload.out_bearing - state[node_idx].prev_in_bearing + 180.0)
                        .rem_euclid(360.0)
                        - 180.0)
                        .abs();
                }
                let simpl_preceding_dist = turn + edge_payload.angle_sum;
                let candidate = state[node_idx].simpl_dist + simpl_preceding_dist;
                let edge_seconds = self.edge_travel_seconds(nb, node_idx, edge_payload, speed_m_s, false);
                let total_seconds = state[node_idx].agg_seconds + edge_seconds;
                if total_seconds > max_seconds as f32 {
                    continue;
                }
                // Single-predecessor: strict < only, since angular turn cost
                // depends on incoming bearing (path-dependent).
                if candidate < state[nb].simpl_dist {
                    state[nb].pred = Some(node_idx);
                    state[nb].simpl_dist = candidate;
                    state[nb].agg_seconds = total_seconds;
                    state[nb].prev_in_bearing = edge_payload.in_bearing;
                    state[nb].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb,
                        metric: candidate,
                    });
                }
            }
        }
        (visited_nodes, state)
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
        let mut tree_map = vec![NodeVisit::new(); self.node_bound()];
        let mut visited_nodes = Vec::new();
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
                // With Incoming, source() gives the neighbor node (edge goes: neighbor → current)
                let nb_nd_idx = edge_ref.source();
                // Use incoming edge directly - it has the correct distance for Y→X
                let edge_payload = edge_ref.weight();
                if nb_nd_idx.index() == node_idx {
                    continue;
                }
                if let Some(pred_idx) = tree_map[node_idx].pred {
                    if nb_nd_idx.index() == pred_idx {
                        continue;
                    }
                }
                if tree_map[nb_nd_idx.index()].pred.is_some() {
                    // Cycle detection: another path exists to this neighbor
                    // Attribution based on which node was discovered first (smaller agg_seconds)
                    if tree_map[node_idx].agg_seconds <= tree_map[nb_nd_idx.index()].agg_seconds {
                        tree_map[nb_nd_idx.index()].cycles += 0.5;
                    } else {
                        tree_map[node_idx].cycles += 0.5;
                    }
                }
                // Skip already-visited nodes to prevent stale distance updates
                // that cannot propagate (the node has already been explored).
                if tree_map[nb_nd_idx.index()].visited {
                    continue;
                }
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
                    tree_map[nb_nd_idx.index()].short_dist = total_seconds * speed_m_s;
                    tree_map[nb_nd_idx.index()].agg_seconds = total_seconds;
                    tree_map[nb_nd_idx.index()].pred = Some(node_idx);
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        metric: total_seconds,
                    });
                }
            }
        }
        Ok((visited_nodes, tree_map))
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
        let mut tree_map = vec![NodeVisit::new(); self.node_bound()];
        let mut visited_nodes = Vec::new();
        tree_map[src_idx].simpl_dist = 0.0;
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
            // For angular paths, the incoming edge has the correct bearings for Y→X travel.
            for edge_ref in self
                .graph
                .edges_directed(current_node_index, Direction::Incoming)
            {
                // With Incoming, source() gives the neighbor node (edge goes: neighbor → current)
                let nb_nd_idx = edge_ref.source();
                // Use incoming edge directly - it has correct distance and bearings for Y→X
                let edge_payload = edge_ref.weight();
                if nb_nd_idx.index() == node_idx {
                    continue;
                }
                if tree_map[nb_nd_idx.index()].visited {
                    continue;
                }
                let current_pred = tree_map[node_idx].pred;
                let neighbor_pred = tree_map[nb_nd_idx.index()].pred;
                if current_pred.is_some()
                    && neighbor_pred.is_some()
                    && current_pred == neighbor_pred
                {
                    continue;
                }
                // Turn angle at node_idx when path goes: predecessor → node_idx → nb_nd_idx
                // Using Direction::Incoming, we get edge Y→X (neighbor→current) but travel is X→Y.
                // Both bearings are 180° offset from the actual travel direction, but the turn
                // calculation still works if we use rem_euclid for proper modular arithmetic.
                let mut turn = 0.0;
                if node_idx != src_idx
                    && edge_payload.out_bearing.is_finite()
                    && tree_map[node_idx].prev_in_bearing.is_finite()
                {
                    turn = ((edge_payload.out_bearing - tree_map[node_idx].prev_in_bearing
                        + 180.0)
                        .rem_euclid(360.0)
                        - 180.0)
                        .abs();
                }
                let simpl_preceding_dist = turn + edge_payload.angle_sum;
                let simpl_total_dist = tree_map[node_idx].simpl_dist + simpl_preceding_dist;
                let edge_seconds = self.edge_travel_seconds(
                    nb_nd_idx.index(),
                    node_idx,
                    edge_payload,
                    speed_m_s,
                    false,
                );
                let total_seconds = tree_map[node_idx].agg_seconds + edge_seconds;
                if total_seconds > max_seconds as f32 {
                    continue;
                }
                if simpl_total_dist < tree_map[nb_nd_idx.index()].simpl_dist {
                    tree_map[nb_nd_idx.index()].simpl_dist = simpl_total_dist;
                    tree_map[nb_nd_idx.index()].agg_seconds = total_seconds;
                    tree_map[nb_nd_idx.index()].pred = Some(node_idx);
                    // Store in_bearing for next turn calculation
                    // in_bearing on edge Y→X = inward bearing from future Z to Y
                    tree_map[nb_nd_idx.index()].prev_in_bearing = edge_payload.in_bearing;
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        metric: simpl_total_dist,
                    });
                }
            }
        }
        Ok((visited_nodes, tree_map))
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

        // Count live nodes for betweenness scaling (needed for backwards-compat path)
        let n_live: usize = self
            .node_indices()
            .iter()
            .filter(|&&idx| self.is_node_live_unchecked(idx))
            .count();

        let node_keys_py = self.node_keys_py(py);
        let node_indices = self.node_indices();
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

        // Pre-generate random samples from a single RNG to ensure uniform distribution.
        // Only needed for stochastic path (no source_indices).
        let sample_randoms: Vec<f32> = if sample_probability.is_some() && source_indices.is_none() {
            let mut rng = if let Some(seed) = random_seed {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_rng(&mut rand::rng())
            };
            (0..self.node_bound()).map(|_| rng.random()).collect()
        } else {
            Vec::new()
        };

        // Determine which sources to iterate
        let is_source_indexed = source_indices.is_some();
        let sources: Vec<usize> = if let Some(ref indices) = source_indices {
            indices.clone()
        } else {
            node_indices.clone()
        };
        let n_sources = sources.len();
        // Marks nodes that can contribute as sources under this run configuration.
        // Used for direction-free pair weighting in undirected betweenness:
        // - target can also be a source: 0.5 contribution from each direction
        // - target cannot be a source: full 1.0 from the only possible direction
        let source_eligible: Vec<bool> = if is_source_indexed {
            let mut eligible = vec![false; n];
            for &idx in &sources {
                eligible[idx] = true;
            }
            eligible
        } else {
            let mut eligible = vec![false; n];
            for &idx in &node_indices {
                if self.is_node_live_unchecked(idx) {
                    eligible[idx] = true;
                }
            }
            eligible
        };

        // Atomic counters for tracking source reachability when sampling
        let source_reachability_totals: Vec<AtomicU32> =
            distances.iter().map(|_| AtomicU32::new(0)).collect();
        let sampled_source_count = AtomicU32::new(0);

        let result = py.detach(move || {
            sources.par_iter().for_each(|src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !source_eligible[*src_idx] {
                    return;
                }

                // Source sampling: Horvitz-Thompson (IPW) estimator scales by 1/p_src.
                // This weight is used for closeness accumulation directly.
                let mut wt = self.get_node_weight_unchecked(*src_idx);
                if is_source_indexed {
                    if let Some(prob) = sample_probability {
                        wt /= prob;
                    }
                    sampled_source_count.fetch_add(1, AtomicOrdering::Relaxed);
                } else if let Some(prob) = sample_probability {
                    let mut p = prob;
                    if let Some(ref weights) = sampling_weights {
                        p *= weights[*src_idx];
                    }
                    if p <= 0.0 {
                        return;
                    }
                    if sample_randoms[*src_idx] >= p {
                        return;
                    }
                    sampled_source_count.fetch_add(1, AtomicOrdering::Relaxed);
                    wt /= p;
                }

                // Single Dijkstra traversal serves both closeness and betweenness
                let (visited_nodes, state) = self.dijkstra_brandes_shortest(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    tolerance,
                    compute_closeness,
                );

                // --- Closeness accumulation ---
                if compute_closeness {
                    for to_idx in visited_nodes.iter() {
                        let node_state = &state[*to_idx];
                        if to_idx == src_idx {
                            continue;
                        }
                        if !node_state.agg_seconds.is_finite() {
                            continue;
                        }
                        // Track reachability per distance when sampling
                        if sample_probability.is_some() || is_source_indexed {
                            for i in 0..distances.len() {
                                if node_state.short_dist <= distances[i] as f32 {
                                    source_reachability_totals[i]
                                        .fetch_add(1, AtomicOrdering::Relaxed);
                                }
                            }
                        }
                        // Flipped aggregation: accumulate to target (to_idx) not source.
                        for i in 0..distances.len() {
                            let distance = distances[i];
                            let beta = betas[i];
                            if node_state.short_dist <= distance as f32 {
                                res.node_density_vec.metric[i][*to_idx]
                                    .fetch_add(wt as f64, AtomicOrdering::Relaxed);
                                res.node_farness_vec.metric[i][*to_idx].fetch_add(
                                    (node_state.short_dist * wt) as f64,
                                    AtomicOrdering::Relaxed,
                                );
                                res.node_cycles_vec.metric[i][*to_idx].fetch_add(
                                    (node_state.cycles * wt) as f64,
                                    AtomicOrdering::Relaxed,
                                );
                                res.node_harmonic_vec.metric[i][*to_idx].fetch_add(
                                    ((1.0 / node_state.short_dist) * wt) as f64,
                                    AtomicOrdering::Relaxed,
                                );
                                res.node_beta_vec.metric[i][*to_idx].fetch_add(
                                    ((-beta * node_state.short_dist).exp() * wt) as f64,
                                    AtomicOrdering::Relaxed,
                                );
                            }
                        }
                    }
                }

                // --- Betweenness backpropagation ---
                if compute_betweenness {
                    // Sort visited by distance from source (farthest first)
                    let mut sorted_visited: Vec<usize> = visited_nodes
                        .iter()
                        .filter(|&&v| v != *src_idx && state[v].sigma > 0.0)
                        .copied()
                        .collect();
                    sorted_visited.sort_by(|a, b| {
                        state[*b]
                            .short_dist
                            .partial_cmp(&state[*a].short_dist)
                            .unwrap_or(Ordering::Equal)
                    });

                    for d_idx in 0..distances.len() {
                        let dist_threshold = distances[d_idx] as f32;
                        let beta = betas[d_idx] as f64;

                        let mut delta = vec![0.0f64; n];
                        let mut delta_beta = vec![0.0f64; n];

                        for &w in &sorted_visited {
                            if state[w].short_dist > dist_threshold {
                                continue;
                            }
                            let sigma_w = state[w].sigma;
                            if sigma_w == 0.0 {
                                continue;
                            }
                            // Direction-free pair weighting for undirected graphs:
                            // if target can also appear as a source, each direction
                            // contributes 0.5; otherwise this is the only direction
                            // and contributes 1.0.
                            let pair_count = if source_eligible[w] { 0.5 } else { 1.0 };
                            let pair_beta = pair_count * (-beta * state[w].short_dist as f64).exp();
                            for &v in &state[w].preds {
                                let factor = state[v].sigma / sigma_w;
                                delta[v] += factor * (pair_count + delta[w]);
                                delta_beta[v] += factor * (pair_beta + delta_beta[w]);
                            }
                            res.node_betweenness_vec.metric[d_idx][w]
                                .fetch_add(delta[w] * wt as f64, AtomicOrdering::Relaxed);
                            res.node_betweenness_beta_vec.metric[d_idx][w]
                                .fetch_add(delta_beta[w] * wt as f64, AtomicOrdering::Relaxed);
                        }
                    }
                }
            });

            // Closeness sampling metadata
            if sample_probability.is_some() || is_source_indexed {
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
                let scale = if is_source_indexed {
                    if sample_probability.is_some() {
                        1.0
                    } else {
                        n_live as f64 / n_sources as f64
                    }
                } else {
                    1.0
                };
                if scale != 1.0 {
                    for d_idx in 0..distances.len() {
                        for &node_idx in &node_indices {
                            let raw = res.node_betweenness_vec.metric[d_idx][node_idx]
                                .load(AtomicOrdering::Relaxed);
                            res.node_betweenness_vec.metric[d_idx][node_idx]
                                .store(raw * scale, AtomicOrdering::Relaxed);
                            let raw_beta = res.node_betweenness_beta_vec.metric[d_idx][node_idx]
                                .load(AtomicOrdering::Relaxed);
                            res.node_betweenness_beta_vec.metric[d_idx][node_idx]
                                .store(raw_beta * scale, AtomicOrdering::Relaxed);
                        }
                    }
                }
            }

            res
        });

        Ok(result)
    }

    /// Compute node centrality using simplest (angular) paths with a single Dijkstra per source.
    ///
    /// When both `compute_closeness` and `compute_betweenness` are true, a single
    /// Brandes-style Dijkstra traversal per source produces the data for both
    /// closeness accumulation and betweenness backpropagation.
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

        // Count live nodes for betweenness scaling (needed for backwards-compat path)
        let n_live: usize = self
            .node_indices()
            .iter()
            .filter(|&&idx| self.is_node_live_unchecked(idx))
            .count();

        let node_keys_py = self.node_keys_py(py);
        let node_indices = self.node_indices();
        let n = self.node_bound();
        let mut res = CentralitySimplestResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
            n,
            0.0,
        );

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        // Pre-generate random samples from a single RNG to ensure uniform distribution.
        let sample_randoms: Vec<f32> = if sample_probability.is_some() && source_indices.is_none() {
            let mut rng = if let Some(seed) = random_seed {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_rng(&mut rand::rng())
            };
            (0..self.node_bound()).map(|_| rng.random()).collect()
        } else {
            Vec::new()
        };

        // Determine which sources to iterate
        let is_source_indexed = source_indices.is_some();
        let sources: Vec<usize> = if let Some(ref indices) = source_indices {
            indices.clone()
        } else {
            node_indices.clone()
        };
        let n_sources = sources.len();
        // Marks nodes that can contribute as sources under this run configuration.
        // Used for direction-free pair weighting in undirected betweenness:
        // - target can also be a source: 0.5 contribution from each direction
        // - target cannot be a source: full 1.0 from the only possible direction
        let source_eligible: Vec<bool> = if is_source_indexed {
            let mut eligible = vec![false; n];
            for &idx in &sources {
                eligible[idx] = true;
            }
            eligible
        } else {
            let mut eligible = vec![false; n];
            for &idx in &node_indices {
                if self.is_node_live_unchecked(idx) {
                    eligible[idx] = true;
                }
            }
            eligible
        };

        // Atomic counters for tracking source reachability when sampling
        let source_reachability_totals: Vec<AtomicU32> =
            seconds.iter().map(|_| AtomicU32::new(0)).collect();
        let sampled_source_count = AtomicU32::new(0);

        let result = py.detach(move || {
            sources.par_iter().for_each(|src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !source_eligible[*src_idx] {
                    return;
                }

                // Source sampling: Horvitz-Thompson (IPW) estimator
                let mut wt = self.get_node_weight_unchecked(*src_idx);
                if is_source_indexed {
                    if let Some(prob) = sample_probability {
                        wt /= prob;
                    }
                    sampled_source_count.fetch_add(1, AtomicOrdering::Relaxed);
                } else if let Some(prob) = sample_probability {
                    let mut p = prob;
                    if let Some(ref weights) = sampling_weights {
                        p *= weights[*src_idx];
                    }
                    if p <= 0.0 {
                        return;
                    }
                    if sample_randoms[*src_idx] >= p {
                        return;
                    }
                    sampled_source_count.fetch_add(1, AtomicOrdering::Relaxed);
                    wt /= p;
                }

                // Single Dijkstra traversal serves both closeness and betweenness
                let (visited_nodes, state) =
                    self.dijkstra_simplest(*src_idx, max_walk_seconds, speed_m_s);

                // --- Closeness accumulation ---
                if compute_closeness {
                    for to_idx in visited_nodes.iter() {
                        let node_state = &state[*to_idx];
                        if to_idx == src_idx {
                            continue;
                        }
                        if !node_state.agg_seconds.is_finite() {
                            continue;
                        }
                        // Track reachability per time threshold when sampling
                        if sample_probability.is_some() || is_source_indexed {
                            for i in 0..seconds.len() {
                                if node_state.agg_seconds <= seconds[i] as f32 {
                                    source_reachability_totals[i]
                                        .fetch_add(1, AtomicOrdering::Relaxed);
                                }
                            }
                        }
                        // Flipped aggregation: accumulate to target (to_idx) not source.
                        for i in 0..seconds.len() {
                            let sec = seconds[i];
                            if node_state.agg_seconds <= sec as f32 {
                                res.node_density_vec.metric[i][*to_idx]
                                    .fetch_add(wt as f64, AtomicOrdering::Relaxed);
                                let far_ang = farness_scaling_offset
                                    + (node_state.simpl_dist / angular_scaling_unit);
                                res.node_farness_vec.metric[i][*to_idx]
                                    .fetch_add((far_ang * wt) as f64, AtomicOrdering::Relaxed);
                                let harm_ang = 1.0 + (node_state.simpl_dist / angular_scaling_unit);
                                res.node_harmonic_vec.metric[i][*to_idx].fetch_add(
                                    ((1.0 / harm_ang) * wt) as f64,
                                    AtomicOrdering::Relaxed,
                                );
                            }
                        }
                    }
                }

                // --- Betweenness via predecessor chain tracing ---
                // Angular paths have single-predecessor semantics (turn cost
                // depends on incoming bearing), so we walk each target's
                // predecessor chain back to source, incrementing betweenness
                // for each intermediate node. This matches the pre-v4.23
                // approach and avoids Brandes backpropagation, which requires
                // multi-predecessor semantics that are invalid for angular paths.
                if compute_betweenness {
                    for &to_idx in &visited_nodes {
                        if to_idx == *src_idx {
                            continue;
                        }
                        let pair_count = if source_eligible[to_idx] { 0.5 } else { 1.0 };
                        // Walk predecessor chain from target back to source.
                        let mut current_pred = state[to_idx].pred;
                        while let Some(inter_idx) = current_pred {
                            if inter_idx == *src_idx {
                                break;
                            }
                            for d_idx in 0..seconds.len() {
                                if state[to_idx].agg_seconds <= seconds[d_idx] as f32 {
                                    res.node_betweenness_vec.metric[d_idx][inter_idx]
                                        .fetch_add(pair_count * wt as f64, AtomicOrdering::Relaxed);
                                }
                            }
                            current_pred = state[inter_idx].pred;
                        }
                    }
                }
            });

            // Closeness sampling metadata
            if sample_probability.is_some() || is_source_indexed {
                res.sampled_source_count = sampled_source_count.load(AtomicOrdering::Relaxed);
                res.reachability_totals = source_reachability_totals
                    .iter()
                    .map(|a| a.load(AtomicOrdering::Relaxed))
                    .collect();
            }

            // Betweenness post-hoc scaling (pair weighting handled in-loop, see
            // centrality_shortest).
            if compute_betweenness {
                let scale = if is_source_indexed {
                    if sample_probability.is_some() {
                        1.0
                    } else {
                        n_live as f64 / n_sources as f64
                    }
                } else {
                    1.0
                };
                if scale != 1.0 {
                    for d_idx in 0..seconds.len() {
                        for &node_idx in &node_indices {
                            let raw = res.node_betweenness_vec.metric[d_idx][node_idx]
                                .load(AtomicOrdering::Relaxed);
                            res.node_betweenness_vec.metric[d_idx][node_idx]
                                .store(raw * scale, AtomicOrdering::Relaxed);
                        }
                    }
                }
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
    // OD-weighted betweenness (single-predecessor Dijkstra)
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

                let (visited_nodes, state) = self.dijkstra_brandes_shortest(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    tolerance,
                    false,
                );

                // Sort visited by distance (farthest first) for backpropagation
                let mut sorted_visited: Vec<usize> = visited_nodes
                    .iter()
                    .filter(|&&v| v != *src_idx && state[v].sigma > 0.0)
                    .copied()
                    .collect();
                sorted_visited.sort_by(|a, b| {
                    state[*b]
                        .short_dist
                        .partial_cmp(&state[*a].short_dist)
                        .unwrap_or(Ordering::Equal)
                });

                // Brandes backpropagation per distance threshold, weighted by OD flows
                for d_idx in 0..distances.len() {
                    let dist_threshold = distances[d_idx] as f32;
                    let beta = betas[d_idx] as f64;

                    // delta[v] = OD-weighted pair-dependency of this source on v
                    let mut delta = vec![0.0f64; n];
                    let mut delta_beta = vec![0.0f64; n];

                    // Seed delta at OD destinations
                    for &dest in sorted_visited.iter() {
                        if state[dest].short_dist > dist_threshold {
                            continue;
                        }
                        if let Some(&od_w) = src_dests.get(&dest) {
                            delta[dest] += od_w as f64;
                            delta_beta[dest] +=
                                od_w as f64 * (-beta * state[dest].short_dist as f64).exp();
                        }
                    }

                    // Process in reverse order of distance (farthest first)
                    for &w in &sorted_visited {
                        if state[w].short_dist > dist_threshold {
                            continue;
                        }
                        let sigma_w = state[w].sigma;
                        if sigma_w == 0.0 || delta[w] == 0.0 {
                            continue;
                        }

                        // Propagate to predecessors
                        for &v in &state[w].preds {
                            let factor = state[v].sigma / sigma_w;
                            delta[v] += factor * delta[w];
                            delta_beta[v] += factor * delta_beta[w];
                        }

                        // Credit betweenness for w (only intermediates, not source)
                        // Subtract the direct OD seed so we only count pass-through
                        let direct_od = if let Some(&od_w) = src_dests.get(&w) {
                            od_w as f64
                        } else {
                            0.0
                        };
                        let direct_od_beta = if let Some(&od_w) = src_dests.get(&w) {
                            od_w as f64 * (-beta * state[w].short_dist as f64).exp()
                        } else {
                            0.0
                        };
                        let credit = delta[w] - direct_od;
                        let credit_beta = delta_beta[w] - direct_od_beta;
                        if credit > 0.0 {
                            res.node_betweenness_vec.metric[d_idx][w]
                                .fetch_add(credit, AtomicOrdering::Relaxed);
                        }
                        if credit_beta > 0.0 {
                            res.node_betweenness_beta_vec.metric[d_idx][w]
                                .fetch_add(credit_beta, AtomicOrdering::Relaxed);
                        }
                    }
                }
            });

            res
        });

        Ok(result)
    }
}
