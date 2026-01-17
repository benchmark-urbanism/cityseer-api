use crate::common;

use crate::common::MetricResult;
use crate::common::WALKING_SPEED;
use crate::graph::{EdgeVisit, NetworkStructure, NodeVisit};
use numpy::PyArray1;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::Direction;
use pyo3::exceptions;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::sync::atomic::Ordering as AtomicOrdering;

#[pyclass]
pub struct CentralityShortestResult {
    #[pyo3(get)]
    distances: Vec<u32>,
    #[pyo3(get)]
    node_keys_py: Vec<Py<PyAny>>,
    #[pyo3(get)]
    node_indices: Vec<usize>,

    node_density_vec: MetricResult,
    node_farness_vec: MetricResult,
    node_cycles_vec: MetricResult,
    node_harmonic_vec: MetricResult,
    node_beta_vec: MetricResult,
    node_betweenness_vec: MetricResult,
    node_betweenness_beta_vec: MetricResult,
}

impl CentralityShortestResult {
    pub fn new(
        distances: Vec<u32>,
        node_keys_py: Vec<Py<PyAny>>,
        node_indices: Vec<usize>,
        init_val: f32,
    ) -> Self {
        let len = node_indices.len();
        CentralityShortestResult {
            distances: distances.clone(),
            node_keys_py: node_keys_py,
            node_indices: node_indices.clone(),
            node_density_vec: MetricResult::new(&distances, len, init_val),
            node_farness_vec: MetricResult::new(&distances, len, init_val),
            node_cycles_vec: MetricResult::new(&distances, len, init_val),
            node_harmonic_vec: MetricResult::new(&distances, len, init_val),
            node_beta_vec: MetricResult::new(&distances, len, init_val),
            node_betweenness_vec: MetricResult::new(&distances, len, init_val),
            node_betweenness_beta_vec: MetricResult::new(&distances, len, init_val),
        }
    }
}

#[pymethods]
impl CentralityShortestResult {
    #[getter]
    pub fn node_density(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.node_density_vec.load()
    }
    #[getter]
    pub fn node_farness(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.node_farness_vec.load()
    }
    #[getter]
    pub fn node_cycles(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.node_cycles_vec.load()
    }
    #[getter]
    pub fn node_harmonic(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.node_harmonic_vec.load()
    }
    #[getter]
    pub fn node_beta(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.node_beta_vec.load()
    }
    #[getter]
    pub fn node_betweenness(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.node_betweenness_vec.load()
    }
    #[getter]
    pub fn node_betweenness_beta(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.node_betweenness_beta_vec.load()
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

    node_density_vec: MetricResult,
    node_farness_vec: MetricResult,
    node_harmonic_vec: MetricResult,
    node_betweenness_vec: MetricResult,
}

impl CentralitySimplestResult {
    pub fn new(
        distances: Vec<u32>,
        node_keys_py: Vec<Py<PyAny>>,
        node_indices: Vec<usize>,
        init_val: f32,
    ) -> Self {
        let len = node_indices.len();
        CentralitySimplestResult {
            distances: distances.clone(),
            node_keys_py: node_keys_py,
            node_indices: node_indices.clone(),
            node_density_vec: MetricResult::new(&distances, len, init_val),
            node_farness_vec: MetricResult::new(&distances, len, init_val),
            node_harmonic_vec: MetricResult::new(&distances, len, init_val),
            node_betweenness_vec: MetricResult::new(&distances, len, init_val),
        }
    }
}

#[pymethods]
impl CentralitySimplestResult {
    #[getter]
    pub fn node_density(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.node_density_vec.load()
    }
    #[getter]
    pub fn node_farness(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.node_farness_vec.load()
    }
    #[getter]
    pub fn node_harmonic(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.node_harmonic_vec.load()
    }
    #[getter]
    pub fn node_betweenness(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.node_betweenness_vec.load()
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
        init_val: f32,
    ) -> Self {
        let len = node_indices.len();
        CentralitySegmentResult {
            distances: distances.clone(),
            node_keys_py: node_keys_py,
            node_indices: node_indices.clone(),
            segment_density_vec: MetricResult::new(&distances, len, init_val),
            segment_harmonic_vec: MetricResult::new(&distances, len, init_val),
            segment_beta_vec: MetricResult::new(&distances, len, init_val),
            segment_betweenness_vec: MetricResult::new(&distances, len, init_val),
        }
    }
}

#[pymethods]
impl CentralitySegmentResult {
    #[getter]
    pub fn segment_density(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.segment_density_vec.load()
    }
    #[getter]
    pub fn segment_harmonic(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.segment_harmonic_vec.load()
    }
    #[getter]
    pub fn segment_beta(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.segment_beta_vec.load()
    }
    #[getter]
    pub fn segment_betweenness(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.segment_betweenness_vec.load()
    }
}

// NodeDistance for heap
struct NodeDistance {
    node_idx: usize,
    metric: f32,
}

// Implement PartialOrd and Ord focusing on distance for comparison
impl PartialOrd for NodeDistance {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.metric.partial_cmp(&self.metric)
    }
}

impl Ord for NodeDistance {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// PartialEq to satisfy BinaryHeap requirements
impl PartialEq for NodeDistance {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.node_idx == other.node_idx && (self.metric - other.metric).abs() < f32::EPSILON
    }
}

// Implement Eq since we've provided a custom PartialEq
impl Eq for NodeDistance {}

#[pymethods]
impl NetworkStructure {
    #[pyo3(signature = (src_idx, max_seconds, speed_m_s, jitter_scale=None, random_seed=None))]
    pub fn dijkstra_tree_shortest(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        jitter_scale: Option<f32>,
        random_seed: Option<u64>,
    ) -> (Vec<usize>, Vec<NodeVisit>) {
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        let mut tree_map = vec![NodeVisit::new(); self.node_count()];
        let mut visited_nodes = Vec::new();
        tree_map[src_idx].short_dist = 0.0;
        tree_map[src_idx].agg_seconds = 0.0;
        tree_map[src_idx].discovered = true;
        let mut active = BinaryHeap::new();
        active.push(NodeDistance {
            node_idx: src_idx,
            metric: 0.0,
        });
        let mut rng = if let Some(seed) = random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut rand::rng())
        };
        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
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
                let edge_seconds = if edge_payload.seconds.is_nan() {
                    (edge_payload.length * edge_payload.imp_factor) / speed_m_s
                } else {
                    edge_payload.seconds
                };
                let total_seconds = tree_map[node_idx].agg_seconds + edge_seconds;
                if total_seconds > max_seconds as f32 {
                    continue;
                }
                if !tree_map[nb_nd_idx.index()].discovered {
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        metric: total_seconds,
                    });
                }
                let mut jitter = 0.0;
                if jitter_scale > 0.0 {
                    jitter = rng.random::<f32>() * jitter_scale;
                }
                if total_seconds + jitter < tree_map[nb_nd_idx.index()].agg_seconds {
                    tree_map[nb_nd_idx.index()].short_dist = total_seconds * speed_m_s;
                    tree_map[nb_nd_idx.index()].agg_seconds = total_seconds + jitter;
                    tree_map[nb_nd_idx.index()].pred = Some(node_idx);
                }
            }
        }
        (visited_nodes, tree_map)
    }

    #[pyo3(signature = (src_idx, max_seconds, speed_m_s, jitter_scale=None, random_seed=None))]
    pub fn dijkstra_tree_simplest(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        jitter_scale: Option<f32>,
        random_seed: Option<u64>,
    ) -> (Vec<usize>, Vec<NodeVisit>) {
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        let mut tree_map = vec![NodeVisit::new(); self.node_count()];
        let mut visited_nodes = Vec::new();
        tree_map[src_idx].simpl_dist = 0.0;
        tree_map[src_idx].agg_seconds = 0.0;
        tree_map[src_idx].discovered = true;
        let mut active = BinaryHeap::new();
        active.push(NodeDistance {
            node_idx: src_idx,
            metric: 0.0,
        });
        let mut rng = if let Some(seed) = random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut rand::rng())
        };
        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
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
                let edge_seconds = if edge_payload.seconds.is_nan() {
                    edge_payload.length / speed_m_s
                } else {
                    edge_payload.seconds
                };
                let total_seconds = tree_map[node_idx].agg_seconds + edge_seconds;
                if total_seconds > max_seconds as f32 {
                    continue;
                }
                if !tree_map[nb_nd_idx.index()].discovered {
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        metric: simpl_total_dist,
                    });
                }
                let mut jitter = 0.0;
                if jitter_scale > 0.0 {
                    jitter = rng.random::<f32>() * jitter_scale;
                }
                if simpl_total_dist + jitter < tree_map[nb_nd_idx.index()].simpl_dist {
                    tree_map[nb_nd_idx.index()].simpl_dist = simpl_total_dist + jitter;
                    tree_map[nb_nd_idx.index()].agg_seconds = total_seconds;
                    tree_map[nb_nd_idx.index()].pred = Some(node_idx);
                    // Store in_bearing for next turn calculation
                    // in_bearing on edge Y→X = inward bearing from future Z to Y
                    tree_map[nb_nd_idx.index()].prev_in_bearing = edge_payload.in_bearing;
                }
            }
        }
        (visited_nodes, tree_map)
    }

    #[pyo3(signature = (src_idx, max_seconds, speed_m_s, jitter_scale=None, random_seed=None))]
    pub fn dijkstra_tree_segment(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        jitter_scale: Option<f32>,
        random_seed: Option<u64>,
    ) -> (Vec<usize>, Vec<usize>, Vec<NodeVisit>, Vec<EdgeVisit>) {
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        let mut tree_map = vec![NodeVisit::new(); self.node_count()];
        let mut edge_map = vec![EdgeVisit::new(); self.graph.edge_count()];
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
        let mut rng = if let Some(seed) = random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut rand::rng())
        };
        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
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

                let edge_seconds = if edge_payload.seconds.is_nan() {
                    (edge_payload.length * edge_payload.imp_factor) / speed_m_s
                } else {
                    edge_payload.seconds
                };
                let total_seconds = tree_map[node_idx].agg_seconds + edge_seconds;
                if total_seconds > max_seconds as f32 {
                    continue;
                }
                if !tree_map[nb_nd_idx.index()].discovered {
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        metric: total_seconds,
                    });
                }
                let mut jitter = 0.0;
                if jitter_scale > 0.0 {
                    jitter = rng.random::<f32>() * jitter_scale;
                }
                if total_seconds + jitter < tree_map[nb_nd_idx.index()].agg_seconds {
                    let origin_seg = if node_idx == src_idx {
                        edge_idx.index()
                    } else {
                        tree_map[node_idx].origin_seg.expect(
                            "Origin segment must exist for non-source node in segment path update",
                        )
                    };
                    tree_map[nb_nd_idx.index()].short_dist = total_seconds * speed_m_s;
                    tree_map[nb_nd_idx.index()].agg_seconds = total_seconds + jitter;
                    tree_map[nb_nd_idx.index()].pred = Some(node_idx);
                    tree_map[nb_nd_idx.index()].origin_seg = Some(origin_seg);
                    tree_map[nb_nd_idx.index()].last_seg = Some(edge_idx.index());
                }
            }
        }
        (visited_nodes, visited_edges, tree_map, edge_map)
    }

    #[pyo3(signature = (
        distances=None,
        betas=None,
        minutes=None,
        compute_closeness=None,
        compute_betweenness=None,
        min_threshold_wt=None,
        speed_m_s=None,
        jitter_scale=None,
        sample_probability=None,
        sampling_weights=None,
        random_seed=None,
        pbar_disabled=None
    ))]
    pub fn local_node_centrality_shortest(
        &self,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        minutes: Option<Vec<f32>>,
        compute_closeness: Option<bool>,
        compute_betweenness: Option<bool>,
        min_threshold_wt: Option<f32>,
        speed_m_s: Option<f32>,
        jitter_scale: Option<f32>,
        sample_probability: Option<f32>,
        sampling_weights: Option<Vec<f32>>,
        random_seed: Option<u64>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralityShortestResult> {
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
        if let Some(ref weights) = sampling_weights {
            if weights.len() != self.node_count() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "sampling_weights length ({}) must match node count ({})",
                    weights.len(),
                    self.node_count()
                )));
            }
            for (i, &w) in weights.iter().enumerate() {
                if w < 0.0 || w > 1.0 {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "sampling_weights[{}] = {} is out of range [0.0, 1.0]",
                        i, w
                    )));
                }
            }
        }

        let node_keys_py = self.node_keys_py(py);
        let node_indices = self.node_indices();
        let res = CentralityShortestResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
            0.0,
        );

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        // Closeness uses flipped aggregation: accumulate to TARGET nodes (to_idx) not source
        // When sampling: all nodes within range of ANY sampled source get contributions
        self.progress_init();

        // Pre-generate random samples from a single RNG to ensure uniform distribution.
        // Using consecutive seeds (seed + src_idx) causes biased first draws from PRNGs.
        let sample_randoms: Vec<f32> = if sample_probability.is_some() {
            let mut rng = if let Some(seed) = random_seed {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_rng(&mut rand::rng())
            };
            (0..self.node_count()).map(|_| rng.random()).collect()
        } else {
            Vec::new()
        };

        let result = py.detach(move || {
            node_indices.par_iter().for_each(|src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !self.is_node_live(*src_idx) {
                    return;
                }

                // Source sampling: skip Dijkstra for unsampled sources
                // IPW scale applies to all contributions from this source
                let ipw_scale = if let Some(prob) = sample_probability {
                    let mut p = prob;
                    if let Some(ref weights) = sampling_weights {
                        p *= weights[*src_idx];
                    }
                    if sample_randoms[*src_idx] >= p {
                        return; // Skip this source entirely
                    }
                    1.0 / p // Inverse probability weight
                } else {
                    1.0
                };

                let (visited_nodes, tree_map) = self.dijkstra_tree_shortest(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    jitter_scale,
                    random_seed.map(|s| s.wrapping_add(*src_idx as u64)),
                );
                for to_idx in visited_nodes.iter() {
                    let node_visit = &tree_map[*to_idx];
                    if to_idx == src_idx {
                        continue;
                    }
                    if !node_visit.agg_seconds.is_finite() {
                        continue;
                    }
                    // Flipped aggregation: accumulate to target (to_idx) not source
                    // Weight comes from the source node (the node contributing to others' closeness)
                    let wt = self.get_node_weight(*src_idx) * ipw_scale;
                    if compute_closeness {
                        for i in 0..distances.len() {
                            let distance = distances[i];
                            let beta = betas[i];
                            if node_visit.short_dist <= distance as f32 {
                                res.node_density_vec.metric[i][*to_idx]
                                    .fetch_add(wt, AtomicOrdering::Relaxed);
                                res.node_farness_vec.metric[i][*to_idx]
                                    .fetch_add(node_visit.short_dist * wt, AtomicOrdering::Relaxed);
                                // Cycles: accumulate to target (to_idx) for consistency with other metrics
                                // and to ensure all nodes get cycle values with sampling.
                                // node_visit.cycles represents cycles detected at the target node during traversal.
                                res.node_cycles_vec.metric[i][*to_idx]
                                    .fetch_add(node_visit.cycles * wt, AtomicOrdering::Relaxed);
                                res.node_harmonic_vec.metric[i][*to_idx].fetch_add(
                                    (1.0 / node_visit.short_dist) * wt,
                                    AtomicOrdering::Relaxed,
                                );
                                res.node_beta_vec.metric[i][*to_idx].fetch_add(
                                    (-beta * node_visit.short_dist).exp() * wt,
                                    AtomicOrdering::Relaxed,
                                );
                            }
                        }
                    }
                    if compute_betweenness {
                        if to_idx < src_idx {
                            continue;
                        }
                        let mut current_pred = node_visit.pred;
                        while let Some(inter_idx) = current_pred {
                            if inter_idx == *src_idx {
                                break;
                            }
                            let node_visit_short_dist = node_visit.short_dist;
                            for i in 0..distances.len() {
                                let distance = distances[i];
                                let beta = betas[i];
                                if node_visit_short_dist <= distance as f32 {
                                    res.node_betweenness_vec.metric[i][inter_idx]
                                        .fetch_add(wt, AtomicOrdering::Acquire);
                                    let exp_val = (-beta * node_visit_short_dist).exp();
                                    res.node_betweenness_beta_vec.metric[i][inter_idx]
                                        .fetch_add(exp_val * wt, AtomicOrdering::Acquire);
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
        jitter_scale=None,
        sample_probability=None,
        sampling_weights=None,
        random_seed=None,
        pbar_disabled=None
    ))]
    pub fn local_node_centrality_simplest(
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
        jitter_scale: Option<f32>,
        sample_probability: Option<f32>,
        sampling_weights: Option<Vec<f32>>,
        random_seed: Option<u64>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralitySimplestResult> {
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
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
        let compute_closeness = compute_closeness.unwrap_or(true);
        let compute_betweenness = compute_betweenness.unwrap_or(true);
        if !compute_closeness && !compute_betweenness {
            return Err(exceptions::PyValueError::new_err(
            "Either or both closeness and betweenness flags is required, but both parameters are False.",
        ));
        }
        if let Some(ref weights) = sampling_weights {
            if weights.len() != self.node_count() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "sampling_weights length ({}) must match node count ({})",
                    weights.len(),
                    self.node_count()
                )));
            }
            for (i, &w) in weights.iter().enumerate() {
                if w < 0.0 || w > 1.0 {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "sampling_weights[{}] = {} is out of range [0.0, 1.0]",
                        i, w
                    )));
                }
            }
        }
        let angular_scaling_unit = angular_scaling_unit.unwrap_or(180.0);
        let farness_scaling_offset = farness_scaling_offset.unwrap_or(1.0);

        let node_keys_py = self.node_keys_py(py);
        let node_indices = self.node_indices();
        let res = CentralitySimplestResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
            0.0,
        );

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        // Angular (simplest) centrality uses flipped aggregation: accumulate to TARGET nodes (to_idx).
        // dijkstra_tree_simplest uses Direction::Incoming to discover neighbors, then looks up the
        // When sampling: all nodes within range of ANY sampled source get contributions.
        self.progress_init();

        // Pre-generate random samples from a single RNG to ensure uniform distribution.
        // Using consecutive seeds (seed + src_idx) causes biased first draws from PRNGs.
        let sample_randoms: Vec<f32> = if sample_probability.is_some() {
            let mut rng = if let Some(seed) = random_seed {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_rng(&mut rand::rng())
            };
            (0..self.node_count()).map(|_| rng.random()).collect()
        } else {
            Vec::new()
        };

        let result = py.detach(move || {
            node_indices.par_iter().for_each(|src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !self.is_node_live(*src_idx) {
                    return;
                }

                // Source sampling: skip Dijkstra for unsampled sources
                // IPW scale applies to all contributions from this source
                let ipw_scale = if let Some(prob) = sample_probability {
                    let mut p = prob;
                    if let Some(ref weights) = sampling_weights {
                        p *= weights[*src_idx];
                    }
                    if sample_randoms[*src_idx] >= p {
                        return; // Skip this source entirely
                    }
                    1.0 / p // Inverse probability weight
                } else {
                    1.0
                };

                let (visited_nodes, tree_map) = self.dijkstra_tree_simplest(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    jitter_scale,
                    random_seed.map(|s| s.wrapping_add(*src_idx as u64)),
                );
                for to_idx in visited_nodes.iter() {
                    let node_visit = &tree_map[*to_idx];
                    if to_idx == src_idx {
                        continue;
                    }
                    if !node_visit.agg_seconds.is_finite() {
                        continue;
                    }
                    // Flipped aggregation: accumulate to target (to_idx) not source
                    // Weight comes from the source node (the node contributing to others' closeness)
                    let wt = self.get_node_weight(*src_idx) * ipw_scale;
                    if compute_closeness {
                        for i in 0..seconds.len() {
                            let sec = seconds[i];
                            if node_visit.agg_seconds <= sec as f32 {
                                res.node_density_vec.metric[i][*to_idx]
                                    .fetch_add(wt, AtomicOrdering::Relaxed);
                                let far_ang = farness_scaling_offset
                                    + (node_visit.simpl_dist / angular_scaling_unit);
                                res.node_farness_vec.metric[i][*to_idx]
                                    .fetch_add(far_ang * wt, AtomicOrdering::Relaxed);
                                let harm_ang = 1.0 + (node_visit.simpl_dist / angular_scaling_unit);
                                res.node_harmonic_vec.metric[i][*to_idx]
                                    .fetch_add((1.0 / harm_ang) * wt, AtomicOrdering::Relaxed);
                            }
                        }
                    }
                    if compute_betweenness {
                        if to_idx < src_idx {
                            continue;
                        }
                        let mut current_pred = node_visit.pred;
                        while let Some(inter_idx) = current_pred {
                            if inter_idx == *src_idx {
                                break;
                            }
                            for i in 0..seconds.len() {
                                let sec = seconds[i];
                                if node_visit.agg_seconds <= sec as f32 {
                                    res.node_betweenness_vec.metric[i][inter_idx]
                                        .fetch_add(wt, AtomicOrdering::Acquire);
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

    #[pyo3(signature = (
        distances=None,
        betas=None,
        minutes=None,
        compute_closeness=None,
        compute_betweenness=None,
        min_threshold_wt=None,
        speed_m_s=None,
        jitter_scale=None,
        random_seed=None,
        pbar_disabled=None
    ))]
    pub fn local_segment_centrality(
        &self,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        minutes: Option<Vec<f32>>,
        compute_closeness: Option<bool>,
        compute_betweenness: Option<bool>,
        min_threshold_wt: Option<f32>,
        speed_m_s: Option<f32>,
        jitter_scale: Option<f32>,
        random_seed: Option<u64>,
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
            0.0,
        );

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        let result = py.detach(move || {
            node_indices.par_iter().for_each(|src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !self.is_node_live(*src_idx) {
                    return;
                }

                let (visited_nodes, visited_edges, tree_map, edge_map) = self
                    .dijkstra_tree_segment(
                        *src_idx,
                        max_walk_seconds,
                        speed_m_s,
                        jitter_scale,
                        random_seed.map(|s| s.wrapping_add(*src_idx as u64)),
                    );
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

                        let edge_length =
                            self.get_edge_length(start_node_idx, end_node_idx, edge_payload_idx);
                        let imp_factor =
                            self.get_edge_impedance(start_node_idx, end_node_idx, edge_payload_idx);

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
                                    .fetch_add(current_c - a, AtomicOrdering::Relaxed);

                                let seg_harm = if a_imp < 1.0 {
                                    current_c_imp.ln()
                                } else {
                                    (current_c_imp / a_imp).max(f32::EPSILON).ln()
                                };
                                res.segment_harmonic_vec.metric[i][*src_idx]
                                    .fetch_add(seg_harm, AtomicOrdering::Relaxed);

                                let bet = if beta == 0.0 {
                                    current_c_imp - a_imp
                                } else {
                                    ((neg_beta * current_c_imp).exp() - (neg_beta * a_imp).exp())
                                        * inv_neg_beta
                                };
                                res.segment_beta_vec.metric[i][*src_idx]
                                    .fetch_add(bet, AtomicOrdering::Relaxed);
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
                                    .fetch_add(current_d - b, AtomicOrdering::Relaxed);

                                let seg_harm = if b_imp < 1.0 {
                                    current_d_imp.ln()
                                } else {
                                    (current_d_imp / b_imp).max(f32::EPSILON).ln()
                                };
                                res.segment_harmonic_vec.metric[i][*src_idx]
                                    .fetch_add(seg_harm, AtomicOrdering::Relaxed);

                                let bet = if beta == 0.0 {
                                    current_d_imp - b_imp
                                } else {
                                    ((neg_beta * current_d_imp).exp() - (neg_beta * b_imp).exp())
                                        * inv_neg_beta
                                };
                                res.segment_beta_vec.metric[i][*src_idx]
                                    .fetch_add(bet, AtomicOrdering::Relaxed);
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

                        let o_seg_len = self.get_edge_length(
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
                        let l_seg_len = self.get_edge_length(
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
                                            .fetch_add(auc, AtomicOrdering::Acquire);
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
}
