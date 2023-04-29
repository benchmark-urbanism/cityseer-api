use core::panic;
use petgraph::prelude::*;
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use petgraph::Direction;
use pyo3::prelude::*;
use std::collections::HashMap;

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
pub struct NetworkStructure {
    graph: StableDiGraph<NodePayload, EdgePayload>,
}
#[pymethods]
impl NetworkStructure {
    #[new]
    fn new() -> Self {
        Self {
            graph: StableDiGraph::<NodePayload, EdgePayload>::default(),
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
        node_idx_a: usize,
        node_idx_b: usize,
        edge_idx: usize,
        start_nd_key: String,
        end_nd_key: String,
        length: f32,
        angle_sum: f32,
        imp_factor: f32,
        in_bearing: f32,
        out_bearing: f32,
    ) -> usize {
        let _node_idx_a = NodeIndex::new(node_idx_a.try_into().unwrap());
        let _node_idx_b = NodeIndex::new(node_idx_b.try_into().unwrap());
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
    fn get_edge_payload(&self, node_idx_a: usize, node_idx_b: usize) -> Option<EdgePayload> {
        let edge_idx = self.graph.find_edge(
            NodeIndex::new(node_idx_a.try_into().unwrap()),
            NodeIndex::new(node_idx_b.try_into().unwrap()),
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
        max_dist: f32,
        jitter_scale: Option<f32>,
        angular: Option<bool>,
    ) {
        // setup
        let jitter = jitter_scale.unwrap_or(0.0);
        let angular = angular.unwrap_or(false);
        // hashmap of visited nodes
        let mut tree_map: HashMap<usize, NodeVisit> = HashMap::new();
        // hashmap of visited edges
        let mut edge_map: HashMap<usize, EdgeVisit> = HashMap::new();
        // vec of active node indices
        let mut active: Vec<usize> = Vec::new();
        // insert src node
        tree_map.insert(src_idx, NodeVisit::new());
        // the starting node's impedance and distance will be zero
        if let Some(entry) = tree_map.get_mut(&src_idx) {
            entry.short_dist = 0.0;
            entry.simpl_dist = 0.0;
        }
        // prime the active vec with the src node
        active.push(src_idx);
        // keep iterating while adding and removing until done
        while active.len() > 0 {
            // find the next active node with the smallest impedance
            let mut min_nd_idx: Option<usize> = None;
            let mut min_imp = f32::INFINITY;
            for nd_idx in active.iter() {
                let nd_visit_state = tree_map.get(&nd_idx).unwrap();
                let mut imp = f32::NAN;
                if angular {
                    imp = nd_visit_state.simpl_dist
                } else {
                    imp = nd_visit_state.short_dist
                }
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
            if let Some(entry) = tree_map.get_mut(&active_nd_idx.index()) {
                entry.visited = true;
            }
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
                        edge_map.insert(
                            edge_idx.index(),
                            EdgeVisit {
                                visited: true,
                                start_nd_idx: active_nd_idx.index(),
                                end_nd_idx: nb_nd_idx.index(),
                                edge_idx: edge_payload.edge_idx,
                            },
                        );
                        continue;
                    }
                    /*
                    don't visit predecessor nodes
                    otherwise successive nodes revisit out-edges to previous (neighbour) nodes
                    */
                    if nb_nd_idx.index() == tree_map[&active_nd_idx.index()].pred.unwrap() {
                        continue;
                    }
                    // insert the neighbour into the tree map if it doesn't exist yet
                    let nb_visit_state = tree_map
                        .entry(nb_nd_idx.index())
                        .or_insert(NodeVisit::new());
                    /*
                    only add edge to active if the neighbour node has not been processed previously
                    i.e. single direction only - if a neighbour node has been processed it has already been explored
                    */
                    if !nb_visit_state.visited {
                        edge_map.insert(
                            edge_idx.index(),
                            EdgeVisit {
                                visited: true,
                                start_nd_idx: active_nd_idx.index(),
                                end_nd_idx: nb_nd_idx.index(),
                                edge_idx: edge_payload.edge_idx,
                            },
                        );
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
                        if nb_visit_state.visited {
                            if !nb_visit_state.preds.is_none() {}
                        }
                    }
                }
            }
        }
    }
}

pub struct NodeVisit {
    visited: bool,
    pred: Option<usize>,
    short_dist: f32,
    simpl_dist: f32,
    cycles: Option<usize>,
    origin_seg: Option<usize>,
    last_seg: Option<usize>,
    out_bearing: f32,
    in_bearing: f32,
}
impl NodeVisit {
    fn new() -> Self {
        Self {
            visited: false,
            pred: None,
            short_dist: f32::INFINITY,
            simpl_dist: f32::INFINITY,
            cycles: Some(0),
            origin_seg: None,
            last_seg: None,
            out_bearing: f32::NAN,
            in_bearing: f32::NAN,
        }
    }
}
pub struct EdgeVisit {
    visited: bool,
    start_nd_idx: usize,
    end_nd_idx: usize,
    edge_idx: usize,
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
