use crate::common;
use crate::common::MetricResult;
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
    distance: f32,
}
// Implement PartialOrd and Ord focusing on distance for comparison
impl PartialOrd for NodeDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}
impl Ord for NodeDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
// PartialEq to satisfy BinaryHeap requirements
// can't derive PartialEq for f32, so use a custom approach
impl PartialEq for NodeDistance {
    fn eq(&self, other: &Self) -> bool {
        self.node_idx == other.node_idx && (self.distance - other.distance).abs() < f32::EPSILON
    }
}
// Implement Eq since we've provided a custom PartialEq
impl Eq for NodeDistance {}

#[pymethods]
impl NetworkStructure {
    /*
    All shortest paths to max network distance from source node.

    Returns impedances and predecessors for shortest paths from a source node to all other nodes within max
    distance. Angular flag triggers check for sidestepping / cheating with angular impedances (sharp turns).

    Prepares a shortest path tree map - loosely based on dijkstra's shortest path algo. Predecessor map is based on
    impedance heuristic - which can be different from metres. Distance map in metres is used for defining max
    distances and computing equivalent distance measures.
    */
    pub fn dijkstra_tree_shortest(
        &self,
        src_idx: usize,
        max_dist: u32,
        jitter_scale: Option<f32>,
    ) -> (Vec<usize>, Vec<NodeVisit>) {
        /* shortest path implementation */
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        // hashmap of visited nodes
        let mut tree_map: Vec<NodeVisit> = vec![NodeVisit::new(); self.graph.node_count()];
        // vecs of visited nodes and edges
        let mut visited_nodes: Vec<usize> = Vec::new();
        // the starting node's impedance and distance will be zero
        tree_map[src_idx].short_dist = 0.0;
        tree_map[src_idx].discovered = true;
        // min heap
        let mut active: BinaryHeap<NodeDistance> = BinaryHeap::new();
        // prime the min heap with the src node
        active.push(NodeDistance {
            node_idx: src_idx,
            distance: 0.0,
        });
        // random number generator
        let mut rng = rand::thread_rng();
        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
            tree_map[node_idx].visited = true;
            visited_nodes.push(node_idx);
            // visit all edges between the node and its neighbour
            for edge_ref in self
                .graph
                .edges_directed(NodeIndex::new(node_idx), Direction::Outgoing)
            {
                let nb_nd_idx = edge_ref.target();
                let edge_payload = edge_ref.weight();
                // ignore self loops
                if nb_nd_idx.index() == node_idx {
                    continue;
                }
                /*
                don't visit predecessor node
                otherwise successive nodes revisit out-edges to previous (neighbour) nodes
                */
                if !tree_map[node_idx].pred.is_none()
                    && nb_nd_idx.index() == tree_map[node_idx].pred.unwrap()
                {
                    continue;
                }
                /*
                if the neighbouring node has already been discovered then it is a cycle
                do before checking if visited and before checking distance cut-off because this node and the neighbour
                can respectively be within max distance even if cumulative distance across this edge
                (via non-shortest path) exceeds distance. In some cases all distances are run at once,
                so keep behaviour consistent by designating the farthest node (but via the shortest distance)
                as the cycle node
                */
                if !tree_map[nb_nd_idx.index()].pred.is_none() {
                    // bump farther location
                    // prevents mismatching if cycle exceeds threshold in one direction or another
                    if tree_map[node_idx].short_dist <= tree_map[nb_nd_idx.index()].short_dist {
                        tree_map[nb_nd_idx.index()].cycles += 0.5;
                    } else {
                        tree_map[node_idx].cycles += 0.5;
                    }
                }
                // impedance and distance is previous plus new
                let short_preceding_dist = edge_payload.length * edge_payload.imp_factor;
                let short_total_dist = tree_map[node_idx].short_dist + short_preceding_dist;
                // bail if distance is greater than threshold
                // otherwise, add the next node to the heap
                // use unjittered distance for this step
                if short_total_dist > max_dist as f32 {
                    continue;
                } else if !tree_map[nb_nd_idx.index()].discovered {
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        distance: short_total_dist,
                    });
                }
                // inject jitter
                let mut jitter: f32 = 0.0;
                if jitter_scale > 0.0 {
                    jitter = rng.gen::<f32>() * jitter_scale;
                }
                /*
                if impedance less than prior distances for this node then update shortest path
                */
                if short_total_dist + jitter < tree_map[nb_nd_idx.index()].short_dist {
                    tree_map[nb_nd_idx.index()].short_dist = short_total_dist + jitter;
                    tree_map[nb_nd_idx.index()].pred = Some(node_idx);
                }
            }
        }
        (visited_nodes, tree_map)
    }
    pub fn dijkstra_tree_simplest(
        &self,
        src_idx: usize,
        max_dist: u32,
        jitter_scale: Option<f32>,
    ) -> (Vec<usize>, Vec<NodeVisit>) {
        /* simplest path implementation */
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        // hashmap of visited nodes
        let mut tree_map: Vec<NodeVisit> = vec![NodeVisit::new(); self.graph.node_count()];
        // vecs of visited nodes and edges
        let mut visited_nodes: Vec<usize> = Vec::new();
        // the starting node's impedance and distance will be zero
        // tracks shortest path distance for search cut-off
        tree_map[src_idx].short_dist = 0.0;
        tree_map[src_idx].simpl_dist = 0.0;
        tree_map[src_idx].discovered = true;
        // prime the active heap with the src node
        let mut active: BinaryHeap<NodeDistance> = BinaryHeap::new();
        active.push(NodeDistance {
            node_idx: src_idx,
            distance: 0.0,
        });
        // random number generator
        let mut rng = rand::thread_rng();
        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
            tree_map[node_idx].visited = true;
            visited_nodes.push(node_idx);
            // visit all edges between the node and its neighbour
            for edge_ref in self
                .graph
                .edges_directed(NodeIndex::new(node_idx), Direction::Outgoing)
            {
                let nb_nd_idx = edge_ref.target();
                let edge_payload = edge_ref.weight();
                // ignore self loops
                if nb_nd_idx.index() == node_idx {
                    continue;
                }
                // ignore if visited
                if tree_map[nb_nd_idx.index()].visited {
                    continue;
                }
                // ignore if predecessors match
                // this prevents short-cutting sharp turns through combination of two shallower turns
                if !tree_map[node_idx].pred.is_none()
                    && !tree_map[nb_nd_idx.index()].pred.is_none()
                    && tree_map[node_idx].pred == tree_map[nb_nd_idx.index()].pred
                {
                    continue;
                }
                // impedance and distance is previous plus new
                let short_preceding_dist = edge_payload.length * edge_payload.imp_factor;
                let short_total_dist = tree_map[node_idx].short_dist + short_preceding_dist;
                /*
                angular impedance includes two parts:
                A - turn from prior simplest-path route segment
                B - angular change across current segment
                */
                let mut turn: f32 = 0.0;
                if node_idx != src_idx {
                    turn = ((edge_payload.in_bearing - tree_map[node_idx].out_bearing + 180.0)
                        % 360.0
                        - 180.0)
                        .abs();
                }
                let simpl_preceding_dist = turn + edge_payload.angle_sum;
                let simpl_total_dist = tree_map[node_idx].simpl_dist + simpl_preceding_dist;
                // bail if distance is greater than threshold
                // otherwise, add the next node to the heap
                // use unjittered distance for this step
                if short_total_dist > max_dist as f32 {
                    continue;
                } else if !tree_map[nb_nd_idx.index()].discovered {
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        distance: simpl_total_dist,
                    });
                }
                // inject jitter
                let mut jitter: f32 = 0.0;
                if jitter_scale > 0.0 {
                    jitter = rng.gen::<f32>() * jitter_scale;
                }
                /*
                if impedance less than prior distances for this node then update shortest path
                */
                if simpl_total_dist + jitter < tree_map[nb_nd_idx.index()].simpl_dist {
                    tree_map[nb_nd_idx.index()].simpl_dist = simpl_total_dist + jitter;
                    tree_map[nb_nd_idx.index()].short_dist = short_total_dist;
                    tree_map[nb_nd_idx.index()].pred = Some(node_idx);
                    tree_map[nb_nd_idx.index()].out_bearing = edge_payload.out_bearing;
                }
            }
        }
        (visited_nodes, tree_map)
    }
    pub fn dijkstra_tree_segment(
        &self,
        src_idx: usize,
        max_dist: u32,
        jitter_scale: Option<f32>,
    ) -> (Vec<usize>, Vec<usize>, Vec<NodeVisit>, Vec<EdgeVisit>) {
        /* shortest path segment implementation */
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        // hashmap of visited nodes
        let mut tree_map: Vec<NodeVisit> = vec![NodeVisit::new(); self.graph.node_count()];
        // hashmap of visited edges
        let mut edge_map: Vec<EdgeVisit> = vec![EdgeVisit::new(); self.graph.edge_count()];
        // vecs of visited nodes and edges
        let mut visited_nodes: Vec<usize> = Vec::new();
        let mut visited_edges: Vec<usize> = Vec::new();
        // the starting node's impedance and distance will be zero
        tree_map[src_idx].short_dist = 0.0;
        tree_map[src_idx].discovered = true;
        // prime the active heap with the src node
        let mut active: BinaryHeap<NodeDistance> = BinaryHeap::new();
        active.push(NodeDistance {
            node_idx: src_idx,
            distance: 0.0,
        });
        // random number generator
        let mut rng = rand::thread_rng();
        while let Some(NodeDistance { node_idx, .. }) = active.pop() {
            tree_map[node_idx].visited = true;
            visited_nodes.push(node_idx);
            // visit all edges between the node and its neighbour
            for edge_ref in self
                .graph
                .edges_directed(NodeIndex::new(node_idx), Direction::Outgoing)
            {
                let nb_nd_idx = edge_ref.target();
                let edge_idx = edge_ref.id();
                let edge_payload = edge_ref.weight();
                // don't follow self-loops
                if nb_nd_idx.index() == node_idx {
                    // before continuing, add edge to active for segment methods
                    visited_edges.push(edge_idx.index());
                    edge_map[edge_idx.index()].visited = true;
                    edge_map[edge_idx.index()].start_nd_idx = Some(node_idx);
                    edge_map[edge_idx.index()].end_nd_idx = Some(nb_nd_idx.index());
                    edge_map[edge_idx.index()].edge_idx = Some(edge_payload.edge_idx);
                    continue;
                }
                /*
                only do visited check after checking for loops
                only add edge to active if the neighbour node has not been processed previously
                i.e. single direction only - if a neighbour node has been processed it has already been explored
                */
                if tree_map[nb_nd_idx.index()].visited {
                    continue;
                } else {
                    visited_edges.push(edge_idx.index());
                    edge_map[edge_idx.index()].visited = true;
                    edge_map[edge_idx.index()].start_nd_idx = Some(node_idx);
                    edge_map[edge_idx.index()].end_nd_idx = Some(nb_nd_idx.index());
                    edge_map[edge_idx.index()].edge_idx = Some(edge_payload.edge_idx);
                }
                // impedance and distance is previous plus new
                let short_preceding_dist = edge_payload.length * edge_payload.imp_factor;
                let short_total_dist = tree_map[node_idx].short_dist + short_preceding_dist;
                // bail if distance is greater than threshold
                // otherwise, add the next node to the heap
                // use unjittered distance for this step
                if short_total_dist > max_dist as f32 {
                    continue;
                } else if !tree_map[nb_nd_idx.index()].discovered {
                    tree_map[nb_nd_idx.index()].discovered = true;
                    active.push(NodeDistance {
                        node_idx: nb_nd_idx.index(),
                        distance: short_total_dist,
                    });
                }
                // inject jitter
                let mut jitter: f32 = 0.0;
                if jitter_scale > 0.0 {
                    jitter = rng.gen::<f32>() * jitter_scale;
                }
                /*
                if impedance less than prior distances for this node then update shortest path
                */
                if short_total_dist + jitter < tree_map[nb_nd_idx.index()].short_dist {
                    let origin_seg = if node_idx == src_idx {
                        edge_idx.index()
                    } else {
                        tree_map[node_idx].origin_seg.unwrap()
                    };
                    tree_map[nb_nd_idx.index()].short_dist = short_total_dist + jitter;
                    tree_map[nb_nd_idx.index()].pred = Some(node_idx);
                    tree_map[nb_nd_idx.index()].out_bearing = edge_payload.out_bearing;
                    tree_map[nb_nd_idx.index()].origin_seg = Some(origin_seg);
                    tree_map[nb_nd_idx.index()].last_seg = Some(edge_idx.index());
                }
            }
        }
        (visited_nodes, visited_edges, tree_map, edge_map)
    }
    pub fn local_node_centrality_shortest(
        &self,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        compute_closeness: Option<bool>,
        compute_betweenness: Option<bool>,
        min_threshold_wt: Option<f32>,
        jitter_scale: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralityShortestResult> {
        // setup
        self.validate()?;
        let (distances, betas) =
            common::pair_distances_and_betas(distances, betas, min_threshold_wt)?;
        let max_dist: u32 = distances.iter().max().unwrap().clone();
        let compute_closeness = compute_closeness.unwrap_or(true);
        let compute_betweenness = compute_betweenness.unwrap_or(true);
        if !compute_closeness && !compute_betweenness {
            return Err(exceptions::PyValueError::new_err(
            "Either or both closeness and betweenness flags is required, but both parameters are False.",
        ));
        }
        // track progress
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        let result = py.allow_threads(move || {
            // metrics
            let node_density = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_farness = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_cycles = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_harmonic = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_beta = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_betweenness =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_betweenness_beta =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            // indices
            let node_indices: Vec<usize> = self.node_indices();
            // iter
            node_indices.par_iter().for_each(|src_idx| {
                // progress
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                // skip if not live
                if !self.is_node_live(*src_idx).unwrap() {
                    return;
                }
                let (visited_nodes, tree_map) =
                    self.dijkstra_tree_shortest(*src_idx, max_dist, jitter_scale);
                for to_idx in visited_nodes.iter() {
                    let node_visit = tree_map[*to_idx].clone();
                    if to_idx == src_idx {
                        continue;
                    }
                    if !node_visit.short_dist.is_finite() {
                        continue;
                    }
                    let wt = self.get_node_weight(*to_idx).unwrap();
                    if compute_closeness {
                        for i in 0..distances.len() {
                            let distance = distances[i];
                            let beta = betas[i];
                            if node_visit.short_dist <= distance as f32 {
                                node_density.metric[i][*src_idx]
                                    .fetch_add(1.0 * wt, AtomicOrdering::Relaxed);
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
                        let mut inter_idx: usize = node_visit.pred.unwrap();
                        loop {
                            if inter_idx == *src_idx {
                                break;
                            }
                            for i in 0..distances.len() {
                                let distance = distances[i];
                                let beta = betas[i];
                                if node_visit.short_dist <= distance as f32 {
                                    node_betweenness.metric[i][inter_idx]
                                        .fetch_add(1.0 * wt, AtomicOrdering::Acquire);
                                    node_betweenness_beta.metric[i][inter_idx].fetch_add(
                                        (-beta * node_visit.short_dist).exp() * wt,
                                        AtomicOrdering::Acquire,
                                    );
                                }
                            }
                            inter_idx = tree_map[inter_idx].pred.unwrap();
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

    pub fn local_node_centrality_simplest(
        &self,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        compute_closeness: Option<bool>,
        compute_betweenness: Option<bool>,
        min_threshold_wt: Option<f32>,
        jitter_scale: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralitySimplestResult> {
        // setup
        self.validate()?;
        let (distances, _betas) =
            common::pair_distances_and_betas(distances, betas, min_threshold_wt)?;
        let max_dist: u32 = distances.iter().max().unwrap().clone();
        let compute_closeness = compute_closeness.unwrap_or(true);
        let compute_betweenness = compute_betweenness.unwrap_or(true);
        if !compute_closeness && !compute_betweenness {
            return Err(exceptions::PyValueError::new_err(
            "Either or both closeness and betweenness flags is required, but both parameters are False.",
        ));
        }
        // track progress
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        // iter
        let result = py.allow_threads(move || {
            // metrics
            let node_density = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_farness = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_harmonic = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let node_betweenness =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            // indices
            let node_indices: Vec<usize> = self.node_indices();
            // iter
            node_indices.par_iter().for_each(|src_idx| {
                // progress
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                // skip if not live
                if !self.is_node_live(*src_idx).unwrap() {
                    return;
                }
                let (visited_nodes, tree_map) =
                    self.dijkstra_tree_simplest(*src_idx, max_dist, jitter_scale);
                for to_idx in visited_nodes.iter() {
                    let node_visit = tree_map[*to_idx].clone();
                    if to_idx == src_idx {
                        continue;
                    }
                    if !node_visit.short_dist.is_finite() {
                        continue;
                    }
                    let wt = self.get_node_weight(*to_idx).unwrap();
                    if compute_closeness {
                        for i in 0..distances.len() {
                            let distance = distances[i];
                            if node_visit.short_dist <= distance as f32 {
                                let ang = 1.0 + (node_visit.simpl_dist / 180.0);
                                node_density.metric[i][*src_idx]
                                    .fetch_add(1.0 * wt, AtomicOrdering::Relaxed);
                                node_farness.metric[i][*src_idx]
                                    .fetch_add(ang * wt, AtomicOrdering::Relaxed);
                                node_harmonic.metric[i][*src_idx]
                                    .fetch_add((1.0 / ang) * wt, AtomicOrdering::Relaxed);
                            }
                        }
                    }
                    if compute_betweenness {
                        if to_idx < src_idx {
                            continue;
                        }
                        let mut inter_idx: usize = node_visit.pred.unwrap();
                        loop {
                            if inter_idx == *src_idx {
                                break;
                            }
                            for i in 0..distances.len() {
                                let distance = distances[i];
                                if node_visit.short_dist <= distance as f32 {
                                    node_betweenness.metric[i][inter_idx]
                                        .fetch_add(1.0 * wt, AtomicOrdering::Acquire);
                                }
                            }
                            inter_idx = tree_map[inter_idx].pred.unwrap();
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

    pub fn local_segment_centrality(
        &self,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        compute_closeness: Option<bool>,
        compute_betweenness: Option<bool>,
        min_threshold_wt: Option<f32>,
        jitter_scale: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<CentralitySegmentResult> {
        /*
        can't do edge processing as part of shortest tree because all shortest paths have to be resolved first
        hence visiting all processed edges and extrapolating information
        NOTES:
        1. the above shortest tree algorithm only tracks edges in one direction - i.e. no duplication
        2. dijkstra sorts all active nodes by distance: explores from near to far: edges discovered accordingly
        */
        self.validate()?;
        let (distances, betas) =
            common::pair_distances_and_betas(distances, betas, min_threshold_wt)?;
        let max_dist: u32 = distances.iter().max().unwrap().clone();
        let compute_closeness = compute_closeness.unwrap_or(true);
        let compute_betweenness = compute_betweenness.unwrap_or(true);
        if !compute_closeness && !compute_betweenness {
            return Err(exceptions::PyValueError::new_err(
            "Either or both closeness and betweenness flags is required, but both parameters are False.",
        ));
        }
        // track progress
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        // iter
        let result = py.allow_threads(move || {
            // metrics
            let segment_density =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let segment_harmonic =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let segment_beta = MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            let segment_betweenness =
                MetricResult::new(distances.clone(), self.graph.node_count(), 0.0);
            // indices
            let node_indices: Vec<usize> = self.node_indices();
            // iter
            /*
            can't do edge processing as part of shortest tree because all shortest paths have to be resolved first
            hence visiting all processed edges and extrapolating information
            NOTES:
            1. the above shortest tree algorithm only tracks edges in one direction - i.e. no duplication
            2. dijkstra sorts all active nodes by distance: explores from near to far: edges discovered accordingly
            */
            node_indices.par_iter().for_each(|src_idx| {
                // progress
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                // skip if not live
                if !self.is_node_live(*src_idx).unwrap() {
                    return;
                }
                let (visited_nodes, visited_edges, tree_map, edge_map) =
                    self.dijkstra_tree_segment(*src_idx, max_dist, jitter_scale);
                for edge_idx in visited_edges.iter() {
                    let edge_visit = edge_map[*edge_idx].clone();
                    let node_visit_n = tree_map[edge_visit.start_nd_idx.unwrap()].clone();
                    let node_visit_m = tree_map[edge_visit.end_nd_idx.unwrap()].clone();
                    // don't process unreachable segments
                    if !node_visit_n.short_dist.is_finite() && !node_visit_m.short_dist.is_finite()
                    {
                        continue;
                    }
                    /*
                    shortest path (non-angular) uses a split segment workflow
                    the split workflow allows for non-shortest-path edges to be approached from either direction
                    i.e. the shortest path to node "b" isn't necessarily via node "a"
                    the edge is then split at the farthest point from either direction and apportioned either way
                    if the segment is on the shortest path then the second segment will squash down to naught
                    */
                    if compute_closeness {
                        /*
                        dijkstra discovers edges from near to far (sorts before popping next node)
                        i.e. this sort may be unnecessary?
                        */
                        // sort where a < b
                        let n_nearer = node_visit_n.short_dist <= node_visit_m.short_dist;
                        let a = if n_nearer {
                            node_visit_n.short_dist
                        } else {
                            node_visit_m.short_dist
                        };
                        let a_imp = if n_nearer {
                            node_visit_n.short_dist
                        } else {
                            node_visit_m.short_dist
                        };
                        let b = if n_nearer {
                            node_visit_m.short_dist
                        } else {
                            node_visit_n.short_dist
                        };
                        let b_imp = if n_nearer {
                            node_visit_m.short_dist
                        } else {
                            node_visit_n.short_dist
                        };
                        // get the max distance along the segment: seg_len = (m - start_len) + (m - end_len)
                        let edge_payload = self
                            .get_edge_payload(
                                edge_visit.start_nd_idx.unwrap(),
                                edge_visit.end_nd_idx.unwrap(),
                                edge_visit.edge_idx.unwrap(),
                            )
                            .unwrap();
                        // c and d variables can diverge per beneath
                        let mut c = (edge_payload.length + a + b) / 2.0;
                        let mut d = c.clone();
                        // c | d impedance should technically be the same if computed from either side
                        let mut c_imp = a_imp + (c - a) * edge_payload.imp_factor;
                        let mut d_imp = c_imp.clone();
                        // iterate the distance and beta thresholds - from large to small for threshold snipping
                        for i in (0..distances.len()).rev() {
                            let distance = distances[i] as f32;
                            let beta = betas[i];
                            // if c or d are greater than the distance threshold, then the segments are "snipped"
                            // a to c segment
                            if a < distance {
                                if c > distance {
                                    c = distance;
                                    c_imp = a_imp + (distance - a) * edge_payload.imp_factor;
                                }
                                segment_density.metric[i][*src_idx]
                                    .fetch_add(c - a, AtomicOrdering::Relaxed);
                                let seg_harm = if a_imp < 1.0 {
                                    c_imp.ln()
                                } else {
                                    c_imp.ln() - a_imp.ln()
                                };
                                segment_harmonic.metric[i][*src_idx]
                                    .fetch_add(seg_harm, AtomicOrdering::Relaxed);
                                let bet = if beta == 0.0 {
                                    c_imp - a_imp
                                } else {
                                    ((-beta * c_imp).exp() - (-beta * a_imp).exp()) / -beta
                                };
                                segment_beta.metric[i][*src_idx]
                                    .fetch_add(bet, AtomicOrdering::Relaxed);
                            }
                            if b == d {
                                continue;
                            }
                            if b <= distance {
                                if d > distance {
                                    d = distance;
                                    d_imp = b_imp + (distance - b) * edge_payload.imp_factor;
                                }
                                segment_density.metric[i][*src_idx]
                                    .fetch_add(d - b, AtomicOrdering::Relaxed);
                                let seg_harm = if b_imp < 1.0 {
                                    d_imp.ln()
                                } else {
                                    d_imp.ln() - b_imp.ln()
                                };
                                segment_harmonic.metric[i][*src_idx]
                                    .fetch_add(seg_harm, AtomicOrdering::Relaxed);
                                let bet = if beta == 0.0 {
                                    d_imp - b_imp
                                } else {
                                    ((-beta * d_imp).exp() - (-beta * b_imp).exp()) / -beta
                                };
                                segment_beta.metric[i][*src_idx]
                                    .fetch_add(bet, AtomicOrdering::Relaxed);
                            }
                        }
                    }
                }
                if compute_betweenness {
                    // prepare a list of neighbouring nodes relative to the src node
                    let mut nb_nodes: Vec<usize> = Vec::new();
                    for nb_nd_idx in self
                        .graph
                        .neighbors_directed(NodeIndex::new(*src_idx), Direction::Outgoing)
                    {
                        nb_nodes.push(nb_nd_idx.index());
                    }
                    // betweenness is computed per to_idx
                    for to_idx in visited_nodes.iter() {
                        // only process in one direction
                        if to_idx < src_idx {
                            continue;
                        }
                        // skip self node
                        if to_idx == src_idx {
                            continue;
                        }
                        // skip direct neighbours (no nodes between)
                        if nb_nodes.contains(&to_idx) {
                            continue;
                        }
                        // distance - do not proceed if no route available
                        let to_node_visit = tree_map[*to_idx].clone();
                        if !to_node_visit.short_dist.is_finite() {
                            continue;
                        }
                        /*
                        BETWEENNESS
                        segment versions only agg first and last segments
                        the distance decay is based on the distance between the src segment and to segment
                        i.e. willingness of people to walk between src and to segments

                        betweenness is aggregated to intervening nodes based on above distances and decays
                        other sections (in between current first and last) are respectively processed from other to nodes

                        distance thresholds are computed using the inner as opposed to outer edges of the segments
                        */
                        // get the origin and last segment lengths for to_idx
                        let o_seg_idx = to_node_visit.origin_seg.unwrap();
                        let o_seg_len = self
                            .get_edge_payload(
                                edge_map[o_seg_idx].start_nd_idx.unwrap(),
                                edge_map[o_seg_idx].end_nd_idx.unwrap(),
                                edge_map[o_seg_idx].edge_idx.unwrap(),
                            )
                            .unwrap()
                            .length;
                        let l_seg_idx = to_node_visit.last_seg.unwrap();
                        let l_seg_len = self
                            .get_edge_payload(
                                edge_map[l_seg_idx].start_nd_idx.unwrap(),
                                edge_map[l_seg_idx].end_nd_idx.unwrap(),
                                edge_map[l_seg_idx].edge_idx.unwrap(),
                            )
                            .unwrap()
                            .length;
                        // calculate traversal distances from opposing segments
                        let min_span = to_node_visit.short_dist - o_seg_len - l_seg_len;
                        let o_1 = min_span;
                        let mut o_2 = min_span + o_seg_len;
                        let l_1 = min_span;
                        let mut l_2 = min_span + l_seg_len;
                        // betweenness - only counting truly between vertices, not starting and ending verts
                        let mut inter_idx: usize = to_node_visit.pred.unwrap();
                        loop {
                            // break out of while loop if the intermediary has reached the source node
                            if inter_idx == *src_idx {
                                break;
                            }
                            // iterate the distance thresholds - from large to small for threshold snipping
                            for i in (0..distances.len()).rev() {
                                let distance = distances[i];
                                let beta = betas[i];
                                if min_span <= distance as f32 {
                                    // prune if necessary
                                    o_2 = o_2.min(distance as f32);
                                    l_2 = l_2.min(distance as f32);
                                    // catch division by zero
                                    let auc = if beta == 0.0 {
                                        o_2 - o_1 + l_2 - l_1
                                    } else {
                                        ((-beta * o_2).exp() - (-beta * o_1).exp()) / -beta
                                            + ((-beta * l_2).exp() - (-beta * l_1).exp()) / -beta
                                    };
                                    segment_betweenness.metric[i][inter_idx]
                                        .fetch_add(auc, AtomicOrdering::Acquire);
                                }
                            }
                            inter_idx = tree_map[inter_idx].pred.unwrap();
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
/*
Earlier versions of the segment centrality methods had a version for simplest path centralities

CLOSENESS
"""
there is a different workflow for angular - uses single segment (no segment splitting)
this is because the simplest path onto the entire length of segment is from the lower impedance end
this assumes segments are relatively straight, overly complex to subdivide segments for spliting...
"""
# only a single case existing for angular version so no need for abstracted functions
# there are three scenarios:
# 1) e is the predecessor for f
if n_nd_idx == src_idx or preds[m_nd_idx] == n_nd_idx:  # pylint: disable=consider-using-in
e = short_dist[n_nd_idx]
f = short_dist[m_nd_idx]
# if travelling via n, then m = n_imp + seg_ang
# calculations are based on segment length / angle
# i.e. need to decide whether to base angular change on entry vs exit impedance
# else take midpoint of segment as ballpark for average, which is the course taken here
# i.e. exit impedance minus half segment impedance
ang = m_simpl_dist - seg_ang / 2
# 2) f is the predecessor for e
elif m_nd_idx == src_idx or preds[n_nd_idx] == m_nd_idx:  # pylint: disable=consider-using-in
e = short_dist[m_nd_idx]
f = short_dist[n_nd_idx]
ang = n_simpl_dist - seg_ang / 2  # per above
# 3) neither of the above
# get the approach angles for either side and compare to find the least inwards impedance
# this involves impedance up to entrypoint either side plus respective turns onto the segment
else:
# get the out bearing from the predecessor and calculate the turn onto current seg's in bearing
# find n's predecessor
n_pred_idx = int(preds[n_nd_idx])
# find the edge from n's predecessor to n
e_i = _find_edge_idx(node_edge_map, edges_end_arr, n_pred_idx, n_nd_idx)
# get the predecessor edge's outwards bearing at index 6
n_pred_out_bear = edges_out_bearing_arr[e_i]
# calculating the turn into this segment from the predecessor's out bearing
n_turn_in = np.abs((seg_in_bear - n_pred_out_bear + 180) % 360 - 180)
# then add the turn-in to the aggregated impedance at n
# i.e. total angular impedance onto this segment
# as above two scenarios, adding half of angular impedance for segment as avg between in / out
n_ang = n_simpl_dist + n_turn_in + seg_ang / 2
# repeat for the other side other side
# per original n -> m edge destructuring: m is the node in the outwards bound direction
# i.e. need to first find the corresponding edge in the opposite m -> n direction of travel
# this gives the correct inwards bearing as if m were the entry point
opp_i = _find_edge_idx(node_edge_map, edges_end_arr, m_nd_idx, n_nd_idx)
# now that the opposing edge is known, we can fetch the inwards bearing at index 5 (not 6)
opp_in_bear = edges_in_bearing_arr[opp_i]
# find m's predecessor
m_pred_idx = int(preds[m_nd_idx])
# we can now go ahead and find m's predecessor edge
e_i = _find_edge_idx(node_edge_map, edges_end_arr, m_pred_idx, m_nd_idx)
# get the predecessor edge's outwards bearing at index 6
m_pred_out_bear = edges_out_bearing_arr[e_i]
# and calculate the turn-in from m's predecessor onto the m inwards bearing
m_turn_in = np.abs((opp_in_bear - m_pred_out_bear + 180) % 360 - 180)
# then add to aggregated impedance at m
m_ang = m_simpl_dist + m_turn_in + seg_ang / 2
# the distance and angle are based on the smallest angular impedance onto the segment
# select by shortest distance in event angular impedances are identical from either direction
if n_ang == m_ang:
    if n_short_dist <= m_short_dist:
        e = short_dist[n_nd_idx]
        ang = n_ang
    else:
        e = short_dist[m_nd_idx]
        ang = m_ang
elif n_ang < m_ang:
    e = short_dist[n_nd_idx]
    ang = n_ang
else:
    e = short_dist[m_nd_idx]
    ang = m_ang
# f is the entry distance plus segment length
f = e + seg_len
# iterate the distance thresholds - from large to small for threshold snipping
for d_idx in range(len(distances) - 1, -1, -1):
dist_cutoff = distances[d_idx]
if e <= dist_cutoff:
    f = min(f, dist_cutoff)
    # uses segment length as base (in this sense hybrid)
    # intentionally not using integral because conflates harmonic shortest-path w. simplest
    # there is only one case for angular - no need to abstract to func
    for m_idx in close_simpl_idxs:
        # transform - prevents division by zero
        agg_ang = 1 + (ang / 180)
        # then aggregate - angular uses distances explicitly
        shadow_arr[m_idx, d_idx, src_idx] += (f - e) / agg_ang
BETWEENNESS
bt_ang = 1 + simpl_dist[to_idx] / 180
pt_a = o_2 - o_1
pt_b = l_2 - l_1
shadow_arr[m_idx, d_idx, inter_idx] += (pt_a + pt_b) / bt_ang
*/
