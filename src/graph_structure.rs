use atomic_float::AtomicF32;
use core::panic;
use indicatif::{ProgressBar, ProgressDrawTarget};
use numpy::{IntoPyArray, PyArray1};
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::prelude::*;
use petgraph::Direction;
use pyo3::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

#[pyclass]
#[derive(Clone)]
pub struct NodePayload {
    #[pyo3(get)]
    node_key: String,
    #[pyo3(get)]
    x: f32,
    #[pyo3(get)]
    y: f32,
    #[pyo3(get)]
    live: bool,
}
#[pymethods]
impl NodePayload {
    fn xy(&self) -> (f32, f32) {
        (self.x, self.y)
    }
    fn validate(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}
#[pyclass]
#[derive(Clone)]
pub struct EdgePayload {
    #[pyo3(get)]
    start_nd_key: String,
    #[pyo3(get)]
    end_nd_key: String,
    #[pyo3(get)]
    edge_idx: usize,
    #[pyo3(get)]
    length: f32,
    #[pyo3(get)]
    angle_sum: f32,
    #[pyo3(get)]
    imp_factor: f32,
    #[pyo3(get)]
    in_bearing: f32,
    #[pyo3(get)]
    out_bearing: f32,
}
#[pymethods]
impl EdgePayload {
    fn validate(&self) -> bool {
        self.length.is_finite()
            && self.angle_sum.is_finite()
            && self.imp_factor.is_finite()
            && self.in_bearing.is_finite()
            && self.out_bearing.is_finite()
    }
}
#[pyclass]
#[derive(Clone)]
pub struct NodeVisit {
    #[pyo3(get)]
    visited: bool,
    #[pyo3(get)]
    pred: Option<usize>,
    #[pyo3(get)]
    short_dist: f32,
    #[pyo3(get)]
    simpl_dist: f32,
    #[pyo3(get)]
    cycles: f32,
    #[pyo3(get)]
    origin_seg: Option<usize>,
    #[pyo3(get)]
    last_seg: Option<usize>,
    #[pyo3(get)]
    out_bearing: f32,
}
#[pymethods]
impl NodeVisit {
    #[new]
    fn new() -> Self {
        Self {
            visited: false,
            pred: None,
            short_dist: f32::INFINITY,
            simpl_dist: f32::INFINITY,
            cycles: 0.0,
            origin_seg: None,
            last_seg: None,
            out_bearing: f32::NAN,
        }
    }
}
#[pyclass]
#[derive(Clone)]
pub struct EdgeVisit {
    #[pyo3(get)]
    visited: bool,
    #[pyo3(get)]
    start_nd_idx: Option<usize>,
    #[pyo3(get)]
    end_nd_idx: Option<usize>,
    #[pyo3(get)]
    edge_idx: Option<usize>,
}
#[pymethods]
impl EdgeVisit {
    #[new]
    fn new() -> Self {
        Self {
            visited: false,
            start_nd_idx: None,
            end_nd_idx: None,
            edge_idx: None,
        }
    }
}
#[pyclass]
pub struct NetworkStructure {
    graph: DiGraph<NodePayload, EdgePayload>,
}
#[pymethods]
impl NetworkStructure {
    #[new]
    fn new() -> Self {
        Self {
            graph: DiGraph::<NodePayload, EdgePayload>::default(),
        }
    }
    fn add_node(&mut self, node_key: String, x: f32, y: f32, live: bool) -> usize {
        let new_node_idx = self.graph.add_node(NodePayload {
            node_key,
            x,
            y,
            live,
        });
        new_node_idx.index().try_into().unwrap()
    }
    fn get_node_payload(&self, node_idx: usize) -> Option<NodePayload> {
        Some(
            self.graph
                .node_weight(NodeIndex::new(node_idx.try_into().unwrap()))?
                .clone(),
        )
    }
    fn add_edge(
        &mut self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
        start_nd_key: String,
        end_nd_key: String,
        length: f32,
        angle_sum: f32,
        imp_factor: f32,
        in_bearing: f32,
        out_bearing: f32,
    ) -> usize {
        let _node_idx_a = NodeIndex::new(start_nd_idx.try_into().unwrap());
        let _node_idx_b = NodeIndex::new(end_nd_idx.try_into().unwrap());
        let new_edge_idx = self.graph.add_edge(
            _node_idx_a,
            _node_idx_b,
            EdgePayload {
                start_nd_key,
                end_nd_key,
                edge_idx,
                length,
                angle_sum,
                imp_factor,
                in_bearing,
                out_bearing,
            },
        );
        new_edge_idx.index().try_into().unwrap()
    }
    fn get_edge_payload(&self, start_nd_idx: usize, end_nd_idx: usize) -> Option<EdgePayload> {
        let edge_idx = self.graph.find_edge(
            NodeIndex::new(start_nd_idx.try_into().unwrap()),
            NodeIndex::new(end_nd_idx.try_into().unwrap()),
        );
        if edge_idx.is_some() {
            Some(self.graph.edge_weight(edge_idx.unwrap())?.clone())
        } else {
            None
        }
    }
    #[getter]
    fn node_count(&self) -> usize {
        self.graph.node_count().try_into().unwrap()
    }
    #[getter]
    fn edge_count(&self) -> usize {
        self.graph.edge_count().try_into().unwrap()
    }
    fn node_indices(&self) -> Vec<usize> {
        self.graph
            .node_indices()
            .map(|node| node.index() as usize)
            .collect()
    }
    fn edge_indices(&self) -> Vec<(usize, usize)> {
        self.graph
            .edge_indices()
            .map(|edge| {
                let source = self.graph.edge_endpoints(edge).unwrap().0.index() as usize;
                let target = self.graph.edge_endpoints(edge).unwrap().1.index() as usize;
                (source, target)
            })
            .collect()
    }
    fn validate(&self) -> bool {
        if self.node_count() == 0 {
            panic!("NetworkStructure contains no nodes.")
        };
        if self.edge_count() == 0 {
            panic!("NetworkStructure contains no edges.")
        };
        for node_idx in self.graph.node_indices() {
            let node_payload = self.graph.node_weight(node_idx).unwrap();
            if !node_payload.validate() {
                panic!("Invalid node for node idx {:?}.", node_idx)
            }
        }
        for edge_idx in self.graph.edge_indices() {
            let edge_payload = self.graph.edge_weight(edge_idx).unwrap();
            if !edge_payload.validate() {
                panic!("Invalid edge for edge idx {:?}.", edge_idx)
            }
        }
        true
    }
    fn shortest_path_tree(
        &self,
        src_idx: usize,
        max_dist: u32,
        jitter_scale: Option<f32>,
        angular: Option<bool>,
    ) -> (Vec<usize>, Vec<NodeVisit>, Vec<EdgeVisit>) {
        // setup
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        let angular = angular.unwrap_or(false);
        // hashmap of visited nodes
        let mut tree_map: Vec<NodeVisit> = vec![NodeVisit::new(); self.graph.node_count()];
        // hashmap of visited edges
        let mut edge_map: Vec<EdgeVisit> = vec![EdgeVisit::new(); self.graph.edge_count()];
        // vec of visited nodes
        let mut visited_nodes: Vec<usize> = Vec::new();
        // vec of active node indices
        let mut active: Vec<usize> = Vec::new();
        // insert src node
        tree_map.insert(src_idx, NodeVisit::new());
        // the starting node's impedance and distance will be zero
        tree_map[src_idx].short_dist = 0.0;
        tree_map[src_idx].simpl_dist = 0.0;
        // prime the active vec with the src node
        active.push(src_idx);
        // keep iterating while adding and removing until done
        while active.len() > 0 {
            // find the next active node with the smallest impedance
            let mut min_nd_idx: Option<usize> = None;
            let mut min_imp: f32 = f32::INFINITY;
            for nd_idx in active.iter() {
                let imp = if angular {
                    tree_map[*nd_idx].simpl_dist
                } else {
                    tree_map[*nd_idx].short_dist
                };
                if imp < min_imp {
                    min_imp = imp;
                    min_nd_idx = Some(*nd_idx);
                }
            }
            // select the nearest node
            let active_nd_idx = NodeIndex::new(min_nd_idx.unwrap());
            // remove from active vec
            active.retain(|&x| x != active_nd_idx.index());
            // mark as visited in tree map
            tree_map[active_nd_idx.index()].visited = true;
            visited_nodes.push(active_nd_idx.index());
            // visit neighbours
            for nb_nd_idx in self
                .graph
                .neighbors_directed(active_nd_idx, Direction::Outgoing)
            {
                // visit all edges between the node and its neighbour
                for edge_ref in self.graph.edges_connecting(active_nd_idx, nb_nd_idx) {
                    let edge_idx = edge_ref.id();
                    let edge_payload = edge_ref.weight();
                    // don't follow self-loops
                    if nb_nd_idx == active_nd_idx {
                        // before continuing, add edge to active for segment methods
                        edge_map[edge_idx.index()].visited = true;
                        edge_map[edge_idx.index()].start_nd_idx = Some(active_nd_idx.index());
                        edge_map[edge_idx.index()].end_nd_idx = Some(nb_nd_idx.index());
                        edge_map[edge_idx.index()].edge_idx = Some(edge_payload.edge_idx);
                        continue;
                    }
                    /*
                    don't visit predecessor nodes
                    otherwise successive nodes revisit out-edges to previous (neighbour) nodes
                    */
                    if !tree_map[active_nd_idx.index()].pred.is_none() {
                        if nb_nd_idx.index() == tree_map[active_nd_idx.index()].pred.unwrap() {
                            continue;
                        }
                    }
                    /*
                    only add edge to active if the neighbour node has not been processed previously
                    i.e. single direction only - if a neighbour node has been processed it has already been explored
                    */
                    if !tree_map[nb_nd_idx.index()].visited {
                        edge_map[edge_idx.index()].visited = true;
                        edge_map[edge_idx.index()].start_nd_idx = Some(active_nd_idx.index());
                        edge_map[edge_idx.index()].end_nd_idx = Some(nb_nd_idx.index());
                        edge_map[edge_idx.index()].edge_idx = Some(edge_payload.edge_idx);
                    }
                    if !angular {
                        /*
                        if edge has not been claimed AND the neighbouring node has already been discovered,
                        then it is a cycle do before distance cutoff because this node and the neighbour can
                        respectively be within max distance even if cumulative distance across this edge
                        (via non-shortest path) exceeds distance in some cases all distances are run at once,
                        so keep behaviour consistent by designating the farthest node (but via the shortest distance)
                        as the cycle node
                        */
                        if !tree_map[nb_nd_idx.index()].pred.is_none() {
                            if tree_map[active_nd_idx.index()].short_dist
                                <= tree_map[nb_nd_idx.index()].short_dist
                            {
                                tree_map[nb_nd_idx.index()].cycles += 0.5;
                            } else {
                                tree_map[active_nd_idx.index()].cycles += 0.5;
                            }
                        }
                    }
                    // impedance and distance is previous plus new
                    let short_dist: f32 = tree_map[active_nd_idx.index()].short_dist
                        + edge_payload.length * edge_payload.imp_factor;
                    /*
                    angular impedance include two parts:
                    A - turn from prior simplest-path route segment
                    B - angular change across current segment
                    */
                    let mut turn: f32 = 0.0;
                    if active_nd_idx.index() != src_idx {
                        turn = (edge_payload.in_bearing
                            - tree_map[active_nd_idx.index()].out_bearing
                            + 180.0)
                            .abs()
                            % 360.0
                            - 180.0
                    }
                    let simpl_dist =
                        tree_map[active_nd_idx.index()].simpl_dist + turn + edge_payload.angle_sum;
                    // add the neighbour to active if undiscovered but only if less than max shortest path threshold
                    if tree_map[nb_nd_idx.index()].pred.is_none() && short_dist <= max_dist as f32 {
                        active.push(nb_nd_idx.index());
                    }
                    // jitter is for injecting a small amount of stochasticity for rectlinear grids
                    let mut rng = thread_rng();
                    let normal = Normal::new(0.0, 1.0).unwrap();
                    let jitter: f32 = normal.sample(&mut rng) * jitter_scale;
                    /*
                    if impedance less than prior, update
                    this will also happen for the first nodes that overshoot the boundary
                    they will not be explored further because they have not been added to active
                    */
                    // shortest path heuristic differs for angular vs. not
                    if (angular && simpl_dist + jitter < tree_map[nb_nd_idx.index()].simpl_dist)
                        || (!angular
                            && short_dist + jitter < tree_map[nb_nd_idx.index()].short_dist)
                    {
                        let origin_seg = if active_nd_idx.index() == src_idx {
                            edge_idx.index()
                        } else {
                            tree_map[active_nd_idx.index()].origin_seg.unwrap()
                        };
                        if let Some(nb_node_ref) = tree_map.get_mut(nb_nd_idx.index()) {
                            nb_node_ref.simpl_dist = simpl_dist;
                            nb_node_ref.short_dist = short_dist;
                            nb_node_ref.pred = Some(active_nd_idx.index());
                            nb_node_ref.out_bearing = edge_payload.out_bearing;
                            nb_node_ref.origin_seg = Some(origin_seg);
                            nb_node_ref.last_seg = Some(edge_idx.index());
                        }
                    }
                }
            }
        }
        (visited_nodes, tree_map, edge_map)
    }

    fn local_node_centrality_shortest(
        &self,
        distances: Vec<u32>,
        betas: Vec<f32>,
        closeness: bool,
        betweenness: bool,
        pbar_disabled: Option<bool>,
    ) -> (Option<CloseShortResult>, Option<BetwShortResult>) {
        let max_dist: u32 = distances.iter().max().unwrap().clone();
        // metrics
        let node_density = MetricResult::new(distances.clone(), self.graph.node_count());
        let node_farness = MetricResult::new(distances.clone(), self.graph.node_count());
        let node_cycles = MetricResult::new(distances.clone(), self.graph.node_count());
        let node_harmonic = MetricResult::new(distances.clone(), self.graph.node_count());
        let node_beta = MetricResult::new(distances.clone(), self.graph.node_count());
        let node_betweenness = MetricResult::new(distances.clone(), self.graph.node_count());
        let node_betweenness_beta = MetricResult::new(distances.clone(), self.graph.node_count());
        // indices
        let node_indices: Vec<usize> = self
            .graph
            .node_indices()
            .map(|node| node.index() as usize)
            .collect();
        let bar = ProgressBar::new(node_indices.len() as u64).with_message("Computing centrality");
        if pbar_disabled.is_some() {
            if pbar_disabled.unwrap() {
                bar.set_draw_target(ProgressDrawTarget::hidden())
            }
        }
        node_indices.par_iter().for_each(|src_idx| {
            bar.inc(1);
            let (visited_nodes, tree_map, _edge_map) =
                self.shortest_path_tree(*src_idx, max_dist, Some(0.0), Some(false));
            for to_idx in visited_nodes.iter() {
                let node_visit = tree_map[*to_idx].clone();
                if to_idx == src_idx {
                    continue;
                }
                if node_visit.short_dist.is_infinite() {
                    continue;
                }
                if closeness {
                    for i in 0..distances.len() {
                        let distance = distances[i];
                        let beta = betas[i];
                        if node_visit.short_dist > distance as f32 {
                            break;
                        }
                        node_density.metric[i][*src_idx].fetch_add(1.0, Ordering::Relaxed);
                        node_farness.metric[i][*src_idx]
                            .fetch_add(node_visit.short_dist, Ordering::Relaxed);
                        node_cycles.metric[i][*src_idx]
                            .fetch_add(node_visit.cycles, Ordering::Relaxed);
                        node_harmonic.metric[i][*src_idx]
                            .fetch_add(1 as f32 / node_visit.short_dist, Ordering::Relaxed);
                        node_beta.metric[i][*src_idx]
                            .fetch_add((-beta * node_visit.short_dist).exp(), Ordering::Relaxed);
                    }
                }
                if betweenness {
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
                                    .fetch_add(1.0, Ordering::Acquire);
                                node_betweenness_beta.metric[i][inter_idx].fetch_add(
                                    (-beta * node_visit.short_dist).exp(),
                                    Ordering::Acquire,
                                );
                            }
                        }
                        inter_idx = tree_map[inter_idx].pred.unwrap();
                    }
                }
            }
        });
        bar.finish();
        let close_result: Option<CloseShortResult> = if closeness {
            Some(CloseShortResult {
                node_density: node_density.load(),
                node_farness: node_farness.load(),
                node_cycles: node_cycles.load(),
                node_harmonic: node_harmonic.load(),
                node_beta: node_beta.load(),
            })
        } else {
            None
        };
        let betw_result: Option<BetwShortResult> = if betweenness {
            Some(BetwShortResult {
                node_betweenness: node_betweenness.load(),
                node_betweenness_beta: node_betweenness_beta.load(),
            })
        } else {
            None
        };
        (close_result, betw_result)
    }
}
struct MetricResult {
    distances: Vec<u32>,
    metric: Vec<Vec<AtomicF32>>,
}
impl MetricResult {
    fn new(distances: Vec<u32>, size: usize) -> Self {
        let mut metric = Vec::new();
        for _d in 0..distances.len() {
            metric.push(
                // tricky to initialise for given size
                std::iter::repeat_with(|| AtomicF32::new(0.0))
                    .take(size)
                    .collect::<Vec<AtomicF32>>(),
            );
        }
        Self { distances, metric }
    }
    fn load(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        let mut loaded: HashMap<u32, Py<PyArray1<f32>>> = HashMap::new();
        for i in 0..self.distances.len() {
            let dist = self.distances[i];
            let vec_f32: Vec<f32> = self.metric[i]
                .iter()
                .map(|a| a.load(Ordering::SeqCst))
                .collect();
            let array = Python::with_gil(|py| vec_f32.into_pyarray(py).to_owned());
            loaded.insert(dist, array);
        }
        loaded
    }
}
#[pyclass]
pub struct CloseShortResult {
    #[pyo3(get)]
    node_density: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    node_farness: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    node_cycles: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    node_harmonic: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    node_beta: HashMap<u32, Py<PyArray1<f32>>>,
}
#[pyclass]
pub struct BetwShortResult {
    #[pyo3(get)]
    node_betweenness: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    node_betweenness_beta: HashMap<u32, Py<PyArray1<f32>>>,
}
// TODO: can remove these
#[pyclass]
#[derive(Clone)]
struct PyNodeIndex(NodeIndex);
#[pymethods]
impl PyNodeIndex {
    #[new]
    fn new(index: usize) -> Self {
        PyNodeIndex(NodeIndex::new(index))
    }
    #[getter]
    fn index(&self) -> usize {
        self.0.index().try_into().unwrap()
    }
}

#[pyclass]
#[derive(Clone)]
struct PyEdgeIndex(EdgeIndex);
#[pymethods]
impl PyEdgeIndex {
    #[new]
    fn new(index: usize) -> Self {
        PyEdgeIndex(EdgeIndex::new(index))
    }
    #[getter]
    fn index(&self) -> usize {
        self.0.index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_network_structure() {
        pyo3::prepare_freethreaded_python();

        let mut ns = NetworkStructure::new();
        let nd_a = ns.add_node("a".to_string(), 0.0, 0.0, true);
        let nd_b = ns.add_node("b".to_string(), 1.0, 0.0, true);
        let nd_c = ns.add_node("c".to_string(), 1.0, 1.0, true);
        let nd_d = ns.add_node("d".to_string(), 0.0, 1.0, true);
        let e_a = ns.add_edge(
            nd_a,
            nd_b,
            0,
            "a".to_string(),
            "b".to_string(),
            1.0,
            0.0,
            1.0,
            90.0,
            90.0,
        );
        let e_b = ns.add_edge(
            nd_b,
            nd_c,
            0,
            "b".to_string(),
            "c".to_string(),
            1.0,
            0.0,
            1.0,
            180.0,
            180.0,
        );
        let e_c = ns.add_edge(
            nd_c,
            nd_d,
            0,
            "c".to_string(),
            "d".to_string(),
            1.0,
            0.0,
            1.0,
            270.0,
            270.0,
        );
        let (visited_nodes, tree_map, edge_map) = ns.shortest_path_tree(0, 5, None, None);
        let close_result =
            ns.local_node_centrality_shortest(vec![200], vec![0.02], true, false, Some(false));
        // assert_eq!(add(2, 2), 4);
    }
}
