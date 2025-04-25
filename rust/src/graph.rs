use crate::common::Coord;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::prelude::*;
use pyo3::exceptions;
use pyo3::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Payload for a network node.
#[pyclass]
#[derive(Clone)]
pub struct NodePayload {
    #[pyo3(get)]
    pub node_key: String,
    #[pyo3(get)]
    pub coord: Coord,
    #[pyo3(get)]
    pub live: bool,
    #[pyo3(get)]
    pub weight: f32,
}

#[pymethods]
impl NodePayload {
    #[inline]
    pub fn validate(&self) -> bool {
        self.coord.validate()
    }
}

/// Payload for a network edge.
#[pyclass]
#[derive(Clone)]
pub struct EdgePayload {
    #[pyo3(get)]
    pub start_nd_key: String,
    #[pyo3(get)]
    pub end_nd_key: String,
    #[pyo3(get)]
    pub edge_idx: usize,
    #[pyo3(get)]
    pub length: f32,
    #[pyo3(get)]
    pub angle_sum: f32,
    #[pyo3(get)]
    pub imp_factor: f32,
    #[pyo3(get)]
    pub in_bearing: f32,
    #[pyo3(get)]
    pub out_bearing: f32,
    #[pyo3(get)]
    pub seconds: f32,
}

#[pymethods]
impl EdgePayload {
    #[inline]
    pub fn validate(&self) -> bool {
        // If seconds is NaN, all other values must be finite
        if self.seconds.is_nan() {
            self.length.is_finite()
                && self.angle_sum.is_finite()
                && self.imp_factor.is_finite()
                && self.in_bearing.is_finite()
                && self.out_bearing.is_finite()
        } else {
            // If seconds is finite, other values are optional
            self.seconds.is_finite()
                && self.length.is_finite()
                && self.angle_sum.is_finite()
                && self.imp_factor.is_finite()
                && (self.in_bearing.is_finite() || self.in_bearing.is_nan())
                && (self.out_bearing.is_finite() || self.out_bearing.is_nan())
        }
    }
}

/// Visit state for a node during traversal.
#[pyclass]
#[derive(Clone, Copy)]
pub struct NodeVisit {
    #[pyo3(get)]
    pub visited: bool,
    #[pyo3(get)]
    pub discovered: bool,
    #[pyo3(get)]
    pub pred: Option<usize>,
    #[pyo3(get)]
    pub short_dist: f32,
    #[pyo3(get)]
    pub simpl_dist: f32,
    #[pyo3(get)]
    pub cycles: f32,
    #[pyo3(get)]
    pub origin_seg: Option<usize>,
    #[pyo3(get)]
    pub last_seg: Option<usize>,
    #[pyo3(get)]
    pub out_bearing: f32,
    #[pyo3(get)]
    pub agg_seconds: f32,
}

#[pymethods]
impl NodeVisit {
    #[new]
    pub fn new() -> Self {
        Self {
            visited: false,
            discovered: false,
            pred: None,
            short_dist: f32::INFINITY,
            simpl_dist: f32::INFINITY,
            cycles: 0.0,
            origin_seg: None,
            last_seg: None,
            out_bearing: f32::NAN,
            agg_seconds: f32::INFINITY,
        }
    }
}

/// Visit state for an edge during traversal.
#[pyclass]
#[derive(Clone)]
pub struct EdgeVisit {
    #[pyo3(get)]
    pub visited: bool,
    #[pyo3(get)]
    pub start_nd_idx: Option<usize>,
    #[pyo3(get)]
    pub end_nd_idx: Option<usize>,
    #[pyo3(get)]
    pub edge_idx: Option<usize>,
}

#[pymethods]
impl EdgeVisit {
    #[new]
    pub fn new() -> Self {
        Self {
            visited: false,
            start_nd_idx: None,
            end_nd_idx: None,
            edge_idx: None,
        }
    }
}

/// Edge segment for spatial queries.
#[derive(Clone)]
pub struct EdgeSegment {
    pub a_idx: usize,
    pub b_idx: usize,
    pub a: [f32; 2],
    pub b: [f32; 2],
}

impl RTreeObject for EdgeSegment {
    type Envelope = AABB<[f32; 2]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(self.a, self.b)
    }
}

impl PointDistance for EdgeSegment {
    fn distance_2(&self, point: &[f32; 2]) -> f32 {
        // Project point onto segment ab
        let ab = [self.b[0] - self.a[0], self.b[1] - self.a[1]];
        let ap = [point[0] - self.a[0], point[1] - self.a[1]];
        let ab_len2 = ab[0] * ab[0] + ab[1] * ab[1];
        let mut t = 0.0;
        if ab_len2 > 0.0 {
            t = (ap[0] * ab[0] + ap[1] * ab[1]) / ab_len2;
            t = t.clamp(0.0, 1.0);
        }
        let proj = [self.a[0] + ab[0] * t, self.a[1] + ab[1] * t];
        let dx = point[0] - proj[0];
        let dy = point[1] - proj[1];
        dx * dx + dy * dy
    }
}

/// Main network structure.
#[pyclass]
#[derive(Clone)]
pub struct NetworkStructure {
    pub graph: DiGraph<NodePayload, EdgePayload>,
    pub progress: Arc<AtomicUsize>,
    pub edge_rtree: Option<RTree<EdgeSegment>>,
    pub edge_rtree_built: bool,
}

#[pymethods]
impl NetworkStructure {
    #[new]
    pub fn new() -> Self {
        Self {
            graph: DiGraph::<NodePayload, EdgePayload>::default(),
            progress: Arc::new(AtomicUsize::new(0)),
            edge_rtree: None,
            edge_rtree_built: false,
        }
    }

    #[inline]
    pub fn progress_init(&self) {
        self.progress.store(0, Ordering::Relaxed);
    }

    #[inline]
    pub fn progress(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }

    pub fn add_node(&mut self, node_key: String, x: f32, y: f32, live: bool, weight: f32) -> usize {
        let new_node_idx = self.graph.add_node(NodePayload {
            node_key,
            coord: Coord::new(x, y),
            live,
            weight,
        });
        new_node_idx.index()
    }

    pub fn get_node_payload(&self, node_idx: usize) -> PyResult<NodePayload> {
        self.graph
            .node_weight(NodeIndex::new(node_idx))
            .cloned()
            .ok_or_else(|| {
                exceptions::PyValueError::new_err(format!(
                    "No payload for requested node index {}.",
                    node_idx
                ))
            })
    }

    pub fn get_node_weight(&self, node_idx: usize) -> PyResult<f32> {
        self.get_node_payload(node_idx)
            .map(|payload| payload.weight)
    }

    pub fn is_node_live(&self, node_idx: usize) -> PyResult<bool> {
        self.get_node_payload(node_idx).map(|payload| payload.live)
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn node_indices(&self) -> Vec<usize> {
        self.graph.node_indices().map(|node| node.index()).collect()
    }

    #[getter]
    pub fn node_xs(&self) -> Vec<f32> {
        self.graph
            .node_indices()
            .map(|node| self.graph[node].coord.x)
            .collect()
    }

    #[getter]
    pub fn node_ys(&self) -> Vec<f32> {
        self.graph
            .node_indices()
            .map(|node| self.graph[node].coord.y)
            .collect()
    }

    #[getter]
    pub fn node_xys(&self) -> Vec<(f32, f32)> {
        self.graph
            .node_indices()
            .map(|node| self.graph[node].coord.xy())
            .collect()
    }

    #[getter]
    pub fn node_lives(&self) -> Vec<bool> {
        self.graph
            .node_indices()
            .map(|node| self.graph[node].live)
            .collect()
    }

    #[getter]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn add_edge(
        &mut self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
        start_nd_key: String,
        end_nd_key: String,
        length: Option<f32>,
        angle_sum: Option<f32>,
        imp_factor: Option<f32>,
        in_bearing: Option<f32>,
        out_bearing: Option<f32>,
        seconds: Option<f32>,
    ) -> usize {
        let node_idx_a = NodeIndex::new(start_nd_idx);
        let node_idx_b = NodeIndex::new(end_nd_idx);
        let new_edge_idx = self.graph.add_edge(
            node_idx_a,
            node_idx_b,
            EdgePayload {
                start_nd_key,
                end_nd_key,
                edge_idx,
                length: length.unwrap_or(1.0),
                angle_sum: angle_sum.unwrap_or(1.0),
                imp_factor: imp_factor.unwrap_or(1.0),
                in_bearing: in_bearing.unwrap_or(f32::NAN),
                out_bearing: out_bearing.unwrap_or(f32::NAN),
                seconds: seconds.unwrap_or(f32::NAN),
            },
        );
        new_edge_idx.index()
    }

    pub fn edge_references(&self) -> Vec<(usize, usize, usize)> {
        self.graph
            .edge_references()
            .map(|edge_ref| {
                (
                    edge_ref.source().index(),
                    edge_ref.target().index(),
                    edge_ref.weight().edge_idx,
                )
            })
            .collect()
    }

    pub fn get_edge_payload(
        &self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
    ) -> PyResult<EdgePayload> {
        let start_node_index = NodeIndex::new(start_nd_idx);
        let end_node_index = NodeIndex::new(end_nd_idx);
        self.graph
            .edges_connecting(start_node_index, end_node_index)
            .find(|edge_ref| edge_ref.weight().edge_idx == edge_idx)
            .map(|edge_ref| edge_ref.weight().clone())
            .ok_or_else(|| {
                exceptions::PyValueError::new_err(format!(
                    "Edge not found for nodes {}, {}, and idx {}.",
                    start_nd_idx, end_nd_idx, edge_idx
                ))
            })
    }

    pub fn validate(&self) -> PyResult<bool> {
        if self.node_count() == 0 {
            return Err(exceptions::PyValueError::new_err(
                "NetworkStructure contains no nodes.",
            ));
        }
        if self.edge_count() == 0 {
            return Err(exceptions::PyValueError::new_err(
                "NetworkStructure contains no edges.",
            ));
        }
        for node_idx in self.graph.node_indices() {
            let node_payload = self.get_node_payload(node_idx.index())?;
            if !node_payload.validate() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid node payload for node idx {:?}.",
                    node_idx
                )));
            }
        }
        for edge_ref in self.graph.edge_references() {
            let edge_payload = edge_ref.weight();
            if !edge_payload.validate() {
                let start_node_idx = edge_ref.source().index();
                let end_node_idx = edge_ref.target().index();
                let edge_data_idx = edge_payload.edge_idx;
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid edge payload for edge between nodes {} and {} (edge_idx {}).",
                    start_node_idx, end_node_idx, edge_data_idx
                )));
            }
        }
        Ok(true)
    }

    pub fn prep_edge_rtree(&mut self) -> PyResult<()> {
        if self.edge_rtree_built {
            return Ok(());
        }
        let mut segments = Vec::with_capacity(self.graph.edge_count());
        for (a_idx, b_idx, _) in self.edge_references() {
            let a_coord = self.get_node_payload(a_idx)?.coord;
            let b_coord = self.get_node_payload(b_idx)?.coord;
            segments.push(EdgeSegment {
                a_idx,
                b_idx,
                a: [a_coord.x, a_coord.y],
                b: [b_coord.x, b_coord.y],
            });
        }
        self.edge_rtree = Some(RTree::bulk_load(segments));
        self.edge_rtree_built = true;
        Ok(())
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_network_structure() {
        pyo3::prepare_freethreaded_python();
        //     3
        //    / \
        //   /   \
        //  /  a  \
        // 1-------2
        //  \  |  /
        //   \ |b/ c
        //    \|/
        //     0
        // a = 100m = 2 * 50m
        // b = 86.60254m
        // c = 100m
        // all inner angles = 60º
        let mut ns = NetworkStructure::new();
        let nd_a = ns.add_node("a".to_string(), 0.0, -86.60254, true, 1.0);
        let nd_b = ns.add_node("b".to_string(), -50.0, 0.0, true, 1.0);
        let nd_c = ns.add_node("c".to_string(), 50.0, 0.0, true, 1.0);
        let nd_d = ns.add_node("d".to_string(), 0.0, 86.60254, true, 1.0);
        let e_a = ns.add_edge(
            nd_a,
            nd_b,
            0,
            "a".to_string(),
            "b".to_string(),
            100.0,
            0.0,
            1.0,
            120.0,
            120.0,
        );
        let e_b = ns.add_edge(
            nd_a,
            nd_c,
            0,
            "a".to_string(),
            "c".to_string(),
            100.0,
            0.0,
            1.0,
            60.0,
            60.0,
        );
        let e_c = ns.add_edge(
            nd_b,
            nd_c,
            0,
            "b".to_string(),
            "c".to_string(),
            100.0,
            0.0,
            1.0,
            0.0,
            0.0,
        );
        let e_d = ns.add_edge(
            nd_b,
            nd_d,
            0,
            "b".to_string(),
            "d".to_string(),
            100.0,
            0.0,
            1.0,
            60.0,
            60.0,
        );
        let e_e = ns.add_edge(
            nd_c,
            nd_d,
            0,
            "c".to_string(),
            "d".to_string(),
            100.0,
            0.0,
            1.0,
            120.0,
            120.0,
        );
        let (visited_nodes, tree_map) = ns.dijkstra_tree_shortest(0, 5, None);
        // let close_result = ns.local_node_centrality_shortest(
        //     Some(vec![50]),
        //     None,
        //     Some(true),
        //     Some(false),
        //     None,
        //     None,
        //     None,
        // );
        // let betw_result_seg = ns.local_segment_centrality(
        //     Some(vec![50]),
        //     None,
        //     Some(false),
        //     Some(true),
        //     None,
        //     None,
        //     None,
        // );
        // assert_eq!(add(2, 2), 4);
    }
}
*/
