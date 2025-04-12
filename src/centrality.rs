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
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::sync::atomic::Ordering as AtomicOrdering;

// Constants for angular calculations
const HALF_CIRCLE_DEGREES: f32 = 180.0;
const FULL_CIRCLE_DEGREES: f32 = 360.0;

#[pyclass]
pub struct CentralityShortestResult {
    #[pyo3(get)]
    node_density: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    node_farness: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    node_cycles: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    node_harmonic: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    node_beta: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    node_betweenness: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    node_betweenness_beta: Option<HashMap<u32, Py<PyArray1<f32>>>>,
}

#[pyclass]
pub struct CentralitySimplestResult {
    #[pyo3(get)]
    node_density: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    node_farness: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    node_harmonic: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    node_betweenness: Option<HashMap<u32, Py<PyArray1<f32>>>>,
}

#[pyclass]
pub struct CentralitySegmentResult {
    #[pyo3(get)]
    segment_density: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    segment_harmonic: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    segment_beta: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    segment_betweenness: Option<HashMap<u32, Py<PyArray1<f32>>>>,
}

// NodeDistance for heap
struct NodeDistance {
    node_idx: usize,
    metric: f32,
}

// Implement PartialOrd and Ord focusing on distance for comparison
impl PartialOrd for NodeDistance {
    #[inline] // Hint for potential inlining
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.metric.partial_cmp(&self.metric)
    }
}

impl Ord for NodeDistance {
    #[inline] // Hint for potential inlining
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// PartialEq to satisfy BinaryHeap requirements
// can't derive PartialEq for f32, so use a custom approach
impl PartialEq for NodeDistance {
    #[inline] // Hint for potential inlining
    fn eq(&self, other: &Self) -> bool {
        self.node_idx == other.node_idx && (self.metric - other.metric).abs() < f32::EPSILON
    }
}

// Implement Eq since we've provided a custom PartialEq
impl Eq for NodeDistance {}

#[pymethods]
impl NetworkStructure {
    #[pyo3(signature = (src_idx, max_seconds, speed_m_s, jitter_scale=None))]
    pub fn dijkstra_tree_shortest(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        jitter_scale: Option<f32>,
    ) -> (Vec<usize>, Vec<NodeVisit>) {
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        let mut tree_map: Vec<NodeVisit> = vec![NodeVisit::new(); self.graph.node_count()];
        let mut visited_nodes: Vec<usize> = Vec::new();
        tree_map[src_idx].short_dist = 0.0;
        tree_map[src_idx].agg_seconds = 0.0;
        tree_map[src_idx].discovered = true;
        let mut active: BinaryHeap<NodeDistance> = BinaryHeap::new();
        active.push(NodeDistance {
            node_idx: src_idx,
            metric: 0.0,
        });
        let mut rng = rand::rng();
        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
            tree_map[node_idx].visited = true;
            visited_nodes.push(node_idx);
            let current_node_index = NodeIndex::new(node_idx);
            for edge_ref in self
                .graph
                .edges_directed(current_node_index, Direction::Outgoing)
            {
                let nb_nd_idx = edge_ref.target();
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
                } else if !tree_map[nb_nd_idx.index()].discovered {
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        metric: total_seconds,
                    });
                }
                let mut jitter: f32 = 0.0;
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

    #[pyo3(signature = (src_idx, max_seconds, speed_m_s, jitter_scale=None))]
    pub fn dijkstra_tree_simplest(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        jitter_scale: Option<f32>,
    ) -> (Vec<usize>, Vec<NodeVisit>) {
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        let mut tree_map: Vec<NodeVisit> = vec![NodeVisit::new(); self.graph.node_count()];
        let mut visited_nodes: Vec<usize> = Vec::new();
        tree_map[src_idx].simpl_dist = 0.0;
        tree_map[src_idx].agg_seconds = 0.0;
        tree_map[src_idx].discovered = true;
        let mut active: BinaryHeap<NodeDistance> = BinaryHeap::new();
        active.push(NodeDistance {
            node_idx: src_idx,
            metric: 0.0,
        });
        let mut rng = rand::rng();
        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
            tree_map[node_idx].visited = true;
            visited_nodes.push(node_idx);
            let current_node_index = NodeIndex::new(node_idx);
            for edge_ref in self
                .graph
                .edges_directed(current_node_index, Direction::Outgoing)
            {
                let nb_nd_idx = edge_ref.target();
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
                let mut turn: f32 = 0.0;
                if node_idx != src_idx
                    && edge_payload.in_bearing.is_finite()
                    && tree_map[node_idx].out_bearing.is_finite()
                {
                    turn = ((edge_payload.in_bearing - tree_map[node_idx].out_bearing
                        + HALF_CIRCLE_DEGREES)
                        % FULL_CIRCLE_DEGREES
                        - HALF_CIRCLE_DEGREES)
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
                } else if !tree_map[nb_nd_idx.index()].discovered {
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        metric: simpl_total_dist,
                    });
                }
                let mut jitter: f32 = 0.0;
                if jitter_scale > 0.0 {
                    jitter = rng.random::<f32>() * jitter_scale;
                }
                if simpl_total_dist + jitter < tree_map[nb_nd_idx.index()].simpl_dist {
                    tree_map[nb_nd_idx.index()].simpl_dist = simpl_total_dist + jitter;
                    tree_map[nb_nd_idx.index()].agg_seconds = total_seconds;
                    tree_map[nb_nd_idx.index()].pred = Some(node_idx);
                    tree_map[nb_nd_idx.index()].out_bearing = edge_payload.out_bearing;
                }
            }
        }
        (visited_nodes, tree_map)
    }

    #[pyo3(signature = (src_idx, max_seconds, speed_m_s, jitter_scale=None))]
    pub fn dijkstra_tree_segment(
        &self,
        src_idx: usize,
        max_seconds: u32,
        speed_m_s: f32,
        jitter_scale: Option<f32>,
    ) -> (Vec<usize>, Vec<usize>, Vec<NodeVisit>, Vec<EdgeVisit>) {
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        let mut tree_map: Vec<NodeVisit> = vec![NodeVisit::new(); self.graph.node_count()];
        let mut edge_map: Vec<EdgeVisit> = vec![EdgeVisit::new(); self.graph.edge_count()];
        let mut visited_nodes: Vec<usize> = Vec::new();
        let mut visited_edges: Vec<usize> = Vec::new();
        tree_map[src_idx].short_dist = 0.0;
        tree_map[src_idx].agg_seconds = 0.0;
        tree_map[src_idx].discovered = true;
        let mut active: BinaryHeap<NodeDistance> = BinaryHeap::new();
        active.push(NodeDistance {
            node_idx: src_idx,
            metric: 0.0,
        });
        let mut rng = rand::rng();
        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
            tree_map[node_idx].visited = true;
            visited_nodes.push(node_idx);
            let current_node_index = NodeIndex::new(node_idx);
            for edge_ref in self
                .graph
                .edges_directed(current_node_index, Direction::Outgoing)
            {
                let nb_nd_idx = edge_ref.target();
                let edge_idx = edge_ref.id();
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
                } else if !tree_map[nb_nd_idx.index()].discovered {
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        metric: total_seconds,
                    });
                }
                let mut jitter: f32 = 0.0;
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
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralityShortestResult> {
        self.validate()?;
        let (distances, betas, seconds) = common::pair_distances_betas_time(
            distances,
            betas,
            minutes,
            min_threshold_wt,
            speed_m_s,
        )?;
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
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
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        let result = py.allow_threads(move || {
            let node_density = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_farness = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_cycles = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_harmonic = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_beta = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_betweenness =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_betweenness_beta =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_indices: Vec<usize> = self.node_indices();
            node_indices.par_iter().for_each(|src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !self
                    .is_node_live(*src_idx)
                    .expect("Node index must be valid for liveness check")
                {
                    return;
                }
                let (visited_nodes, tree_map) = self.dijkstra_tree_shortest(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    jitter_scale,
                );
                for to_idx in visited_nodes.iter() {
                    let node_visit = &tree_map[*to_idx];
                    if to_idx == src_idx {
                        continue;
                    }
                    if !node_visit.agg_seconds.is_finite() {
                        continue;
                    }
                    let wt = self
                        .get_node_weight(*to_idx)
                        .expect("Visited node index must have weight");
                    if compute_closeness {
                        for i in 0..distances.len() {
                            let distance = distances[i];
                            let beta = betas[i];
                            if node_visit.short_dist <= distance as f32 {
                                node_density.metric[i][*src_idx]
                                    .fetch_add(wt, AtomicOrdering::Relaxed);
                                node_farness.metric[i][*src_idx]
                                    .fetch_add(node_visit.short_dist * wt, AtomicOrdering::Relaxed);
                                node_cycles.metric[i][*src_idx]
                                    .fetch_add(node_visit.cycles * wt, AtomicOrdering::Relaxed);
                                node_harmonic.metric[i][*src_idx].fetch_add(
                                    (1.0 / node_visit.short_dist) * wt,
                                    AtomicOrdering::Relaxed,
                                );
                                node_beta.metric[i][*src_idx].fetch_add(
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
                                    node_betweenness.metric[i][inter_idx]
                                        .fetch_add(wt, AtomicOrdering::Acquire);
                                    let exp_val = (-beta * node_visit_short_dist).exp();
                                    node_betweenness_beta.metric[i][inter_idx]
                                        .fetch_add(exp_val * wt, AtomicOrdering::Acquire);
                                }
                            }
                            current_pred = tree_map[inter_idx].pred;
                        }
                    }
                }
            });
            CentralityShortestResult {
                node_density: if compute_closeness {
                    Some(node_density.load())
                } else {
                    None
                },
                node_farness: if compute_closeness {
                    Some(node_farness.load())
                } else {
                    None
                },
                node_cycles: if compute_closeness {
                    Some(node_cycles.load())
                } else {
                    None
                },
                node_harmonic: if compute_closeness {
                    Some(node_harmonic.load())
                } else {
                    None
                },
                node_beta: if compute_closeness {
                    Some(node_beta.load())
                } else {
                    None
                },
                node_betweenness: if compute_betweenness {
                    Some(node_betweenness.load())
                } else {
                    None
                },
                node_betweenness_beta: if compute_betweenness {
                    Some(node_betweenness_beta.load())
                } else {
                    None
                },
            }
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
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralitySimplestResult> {
        self.validate()?;
        let (distances, _betas, seconds) = common::pair_distances_betas_time(
            distances,
            betas,
            minutes,
            min_threshold_wt,
            speed_m_s,
        )?;
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
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
        let angular_scaling_unit = angular_scaling_unit.unwrap_or(HALF_CIRCLE_DEGREES);
        let farness_scaling_offset = farness_scaling_offset.unwrap_or(1.0);
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        let result = py.allow_threads(move || {
            let node_density = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_farness = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_harmonic = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_betweenness =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_indices: Vec<usize> = self.node_indices();
            node_indices.par_iter().for_each(|src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !self
                    .is_node_live(*src_idx)
                    .expect("Node index must be valid for liveness check")
                {
                    return;
                }
                let (visited_nodes, tree_map) = self.dijkstra_tree_simplest(
                    *src_idx,
                    max_walk_seconds,
                    speed_m_s,
                    jitter_scale,
                );
                for to_idx in visited_nodes.iter() {
                    let node_visit = &tree_map[*to_idx];
                    if to_idx == src_idx {
                        continue;
                    }
                    if !node_visit.agg_seconds.is_finite() {
                        continue;
                    }
                    let wt = self
                        .get_node_weight(*to_idx)
                        .expect("Visited node index must have weight");
                    if compute_closeness {
                        for i in 0..seconds.len() {
                            let sec = seconds[i];
                            if node_visit.agg_seconds <= sec as f32 {
                                node_density.metric[i][*src_idx]
                                    .fetch_add(wt, AtomicOrdering::Relaxed);
                                let far_ang = farness_scaling_offset
                                    + (node_visit.simpl_dist / angular_scaling_unit);
                                node_farness.metric[i][*src_idx]
                                    .fetch_add(far_ang * wt, AtomicOrdering::Relaxed);
                                let harm_ang = 1.0 + (node_visit.simpl_dist / angular_scaling_unit);
                                node_harmonic.metric[i][*src_idx]
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
                                    node_betweenness.metric[i][inter_idx]
                                        .fetch_add(wt, AtomicOrdering::Acquire);
                                }
                            }
                            current_pred = tree_map[inter_idx].pred;
                        }
                    }
                }
            });
            CentralitySimplestResult {
                node_density: if compute_closeness {
                    Some(node_density.load())
                } else {
                    None
                },
                node_farness: if compute_closeness {
                    Some(node_farness.load())
                } else {
                    None
                },
                node_harmonic: if compute_closeness {
                    Some(node_harmonic.load())
                } else {
                    None
                },
                node_betweenness: if compute_betweenness {
                    Some(node_betweenness.load())
                } else {
                    None
                },
            }
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
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralitySegmentResult> {
        self.validate()?;
        let (distances, betas, seconds) = common::pair_distances_betas_time(
            distances,
            betas,
            minutes,
            min_threshold_wt,
            speed_m_s,
        )?;
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
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
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        let result = py.allow_threads(move || {
            let segment_density =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let segment_harmonic =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let segment_beta = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let segment_betweenness =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_indices: Vec<usize> = self.node_indices();
            node_indices.par_iter().for_each(|src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !self
                    .is_node_live(*src_idx)
                    .expect("Node index must be valid for liveness check")
                {
                    return;
                }
                let (visited_nodes, visited_edges, tree_map, edge_map) =
                    self.dijkstra_tree_segment(*src_idx, max_walk_seconds, speed_m_s, jitter_scale);
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

                        let edge_payload = self
                            .get_edge_payload(start_node_idx, end_node_idx, edge_payload_idx)
                            .expect("Edge payload must exist for visited edge index");

                        let imp_factor = edge_payload.imp_factor;

                        let c = (edge_payload.length + a + b) / 2.0;
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
                                segment_density.metric[i][*src_idx]
                                    .fetch_add(current_c - a, AtomicOrdering::Relaxed);

                                let seg_harm = if a_imp < 1.0 {
                                    current_c_imp.ln()
                                } else {
                                    (current_c_imp / a_imp).max(f32::EPSILON).ln()
                                };
                                segment_harmonic.metric[i][*src_idx]
                                    .fetch_add(seg_harm, AtomicOrdering::Relaxed);

                                let bet = if beta == 0.0 {
                                    current_c_imp - a_imp
                                } else {
                                    ((neg_beta * current_c_imp).exp() - (neg_beta * a_imp).exp())
                                        * inv_neg_beta
                                };
                                segment_beta.metric[i][*src_idx]
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
                                segment_density.metric[i][*src_idx]
                                    .fetch_add(current_d - b, AtomicOrdering::Relaxed);

                                let seg_harm = if b_imp < 1.0 {
                                    current_d_imp.ln()
                                } else {
                                    (current_d_imp / b_imp).max(f32::EPSILON).ln()
                                };
                                segment_harmonic.metric[i][*src_idx]
                                    .fetch_add(seg_harm, AtomicOrdering::Relaxed);

                                let bet = if beta == 0.0 {
                                    current_d_imp - b_imp
                                } else {
                                    ((neg_beta * current_d_imp).exp() - (neg_beta * b_imp).exp())
                                        * inv_neg_beta
                                };
                                segment_beta.metric[i][*src_idx]
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

                        let o_seg_len = self
                            .get_edge_payload(
                                o_edge_visit
                                    .start_nd_idx
                                    .expect("Origin edge visit must have start node"),
                                o_edge_visit
                                    .end_nd_idx
                                    .expect("Origin edge visit must have end node"),
                                o_edge_visit
                                    .edge_idx
                                    .expect("Origin edge visit must have edge index"),
                            )
                            .expect("Origin segment payload must exist")
                            .length;
                        let l_seg_len = self
                            .get_edge_payload(
                                l_edge_visit
                                    .start_nd_idx
                                    .expect("Last edge visit must have start node"),
                                l_edge_visit
                                    .end_nd_idx
                                    .expect("Last edge visit must have end node"),
                                l_edge_visit
                                    .edge_idx
                                    .expect("Last edge visit must have edge index"),
                            )
                            .expect("Last segment payload must exist")
                            .length;

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
                                        segment_betweenness.metric[i][inter_idx]
                                            .fetch_add(auc, AtomicOrdering::Acquire);
                                    }
                                }
                            }
                            current_pred = tree_map[inter_idx].pred;
                        }
                    }
                }
            });
            CentralitySegmentResult {
                segment_density: if compute_closeness {
                    Some(segment_density.load())
                } else {
                    None
                },
                segment_harmonic: if compute_closeness {
                    Some(segment_harmonic.load())
                } else {
                    None
                },
                segment_beta: if compute_closeness {
                    Some(segment_beta.load())
                } else {
                    None
                },
                segment_betweenness: if compute_betweenness {
                    Some(segment_betweenness.load())
                } else {
                    None
                },
            }
        });
        Ok(result)
    }
}
