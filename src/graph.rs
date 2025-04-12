use crate::common::{calculate_rotation_smallest, Coord};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::prelude::*;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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
    fn validate(&self) -> bool {
        self.coord.validate()
    }
}
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
    fn validate(&self) -> bool {
        // if seconds is NaN, then all other values must be finite
        if self.seconds.is_nan() {
            return (self.length.is_finite())
                && (self.angle_sum.is_finite())
                && (self.imp_factor.is_finite())
                && (self.in_bearing.is_finite())
                && (self.out_bearing.is_finite());
        }
        // if seconds is finite, then other values are optional
        self.seconds.is_finite()
            && self.length.is_finite()
            && self.angle_sum.is_finite()
            && self.imp_factor.is_finite()
            && (self.in_bearing.is_finite() || self.in_bearing.is_nan())
            && (self.out_bearing.is_finite() || self.out_bearing.is_nan())
    }
}
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
#[pyclass]
#[derive(Clone)]
pub struct NetworkStructure {
    pub graph: DiGraph<NodePayload, EdgePayload>,
    pub progress: Arc<AtomicUsize>,
}
#[pymethods]
impl NetworkStructure {
    #[new]
    fn new() -> Self {
        Self {
            graph: DiGraph::<NodePayload, EdgePayload>::default(),
            progress: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn progress_init(&self) {
        self.progress.store(0, Ordering::Relaxed);
    }
    fn progress(&self) -> usize {
        self.progress.as_ref().load(Ordering::Relaxed)
    }
    fn add_node(&mut self, node_key: String, x: f32, y: f32, live: bool, weight: f32) -> usize {
        let new_node_idx = self.graph.add_node(NodePayload {
            node_key,
            coord: Coord::new(x, y),
            live,
            weight,
        });
        new_node_idx
            .index()
            .try_into()
            .expect("Node index should fit into usize")
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
        let node_payload = self.get_node_payload(node_idx)?;
        Ok(node_payload.weight)
    }
    pub fn is_node_live(&self, node_idx: usize) -> PyResult<bool> {
        let node_payload = self.get_node_payload(node_idx)?;
        Ok(node_payload.live)
    }
    pub fn node_count(&self) -> usize {
        self.graph
            .node_count()
            .try_into()
            .expect("Node count should fit into usize")
    }
    pub fn node_indices(&self) -> Vec<usize> {
        self.graph
            .node_indices()
            .map(|node| node.index() as usize)
            .collect()
    }
    #[getter]
    fn node_xs(&self) -> Vec<f32> {
        self.graph
            .node_indices()
            .map(|node| self.graph[node].coord.x)
            .collect()
    }
    #[getter]
    fn node_ys(&self) -> Vec<f32> {
        self.graph
            .node_indices()
            .map(|node| self.graph[node].coord.y)
            .collect()
    }
    #[getter]
    fn node_xys(&self) -> Vec<(f32, f32)> {
        self.graph
            .node_indices()
            .map(|node| self.graph[node].coord.xy())
            .collect()
    }
    #[getter]
    fn node_lives(&self) -> Vec<bool> {
        self.graph
            .node_indices()
            .map(|node| self.graph[node].live)
            .collect()
    }
    #[getter]
    fn edge_count(&self) -> usize {
        self.graph
            .edge_count()
            .try_into()
            .expect("Edge count should fit into usize")
    }
    fn add_edge(
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
        let node_idx_a = NodeIndex::new(
            start_nd_idx
                .try_into()
                .expect("Start node index should fit into usize"),
        );
        let node_idx_b = NodeIndex::new(
            end_nd_idx
                .try_into()
                .expect("End node index should fit into usize"),
        );
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
        new_edge_idx
            .index()
            .try_into()
            .expect("Edge index should fit into usize")
    }
    fn edge_references(&self) -> Vec<(usize, usize, usize)> {
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
        let start_node_index = NodeIndex::new(
            start_nd_idx
                .try_into()
                .expect("Start node index should fit into usize"),
        );
        let end_node_index = NodeIndex::new(
            end_nd_idx
                .try_into()
                .expect("End node index should fit into usize"),
        );
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
        };
        if self.edge_count() == 0 {
            return Err(exceptions::PyValueError::new_err(
                "NetworkStructure contains no edges.",
            ));
        };
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

    fn assign_to_network(
        &self,
        data_coord: Coord,
        max_dist: f32,
    ) -> PyResult<(Option<usize>, Option<usize>)> {
        let (start_min_idx, start_min_dist, start_next_min_idx) =
            self.find_nearest(data_coord, max_dist)?;
        if start_min_idx.is_none() {
            return Ok((None, None));
        }
        let min_idx = start_min_idx.unwrap();
        if let Some(next_min_idx) = start_next_min_idx {
            if self
                .graph
                .neighbors_directed(NodeIndex::new(min_idx), Direction::Outgoing)
                .any(|nb_nd_idx| nb_nd_idx.index() == next_min_idx)
            {
                return Ok((Some(min_idx), Some(next_min_idx)));
            }
        }
        let mut current_idx = min_idx;
        let mut nearest_idx = min_idx;
        let mut next_nearest_idx: Option<usize> = None;
        let mut prev_idx: Option<usize> = None;
        let mut pred_map: Vec<Option<usize>> = vec![None; self.graph.node_count()];
        let mut min_dist = start_min_dist;
        let mut reversing = false;
        loop {
            let mut best_rotation = f32::INFINITY;
            let mut nb_idx: Option<usize> = None;
            let current_nd_coord = self.get_node_payload(current_idx)?.coord;
            let reference_vec = if let Some(p_idx) = prev_idx {
                self.get_node_payload(p_idx)?
                    .coord
                    .difference(current_nd_coord)
            } else {
                data_coord.difference(current_nd_coord)
            };
            for candidate_nb_idx in self
                .graph
                .neighbors_directed(NodeIndex::new(current_idx), Direction::Outgoing)
            {
                let candidate_idx = candidate_nb_idx.index();
                if candidate_idx == current_idx {
                    continue;
                }
                if prev_idx.map_or(false, |p_idx| candidate_idx == p_idx) {
                    continue;
                }
                let candidate_nb_coord = self.get_node_payload(candidate_idx)?.coord;
                let candidate_vec = candidate_nb_coord.difference(current_nd_coord);
                let mut rot = calculate_rotation_smallest(candidate_vec, reference_vec);

                if !reversing {
                    if rot < 0.0 {
                        rot += 360.0;
                    }
                } else {
                    if rot > 0.0 {
                        rot -= 360.0;
                    }
                    rot = -rot;
                }

                if rot < best_rotation {
                    best_rotation = rot;
                    nb_idx = Some(candidate_idx);
                }
            }
            if nb_idx.is_none() {
                if pred_map.get(current_idx).map_or(true, |opt| opt.is_none()) {
                    if prev_idx.is_none() {
                        break;
                    }
                    if let Some(p_idx) = prev_idx {
                        let (dist, n_opt, n_n_opt) =
                            self.closest_intersections(data_coord, pred_map.clone(), p_idx)?;
                        if dist < min_dist {
                            if let Some(n) = n_opt {
                                nearest_idx = n;
                                next_nearest_idx = n_n_opt;
                            }
                        }
                    } else {
                        return Err(exceptions::PyValueError::new_err(
                            "Internal logic error: prev_idx is None unexpectedly in assign_to_network.",
                        ));
                    }
                    break;
                }
                nb_idx = pred_map.get(current_idx).and_then(|&opt| opt);
                if nb_idx.is_none() {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "Internal logic error: Failed to backtrack, invalid current_idx {} or pred_map state.", current_idx
                    )));
                }
            }
            let selected_nb_idx = nb_idx.unwrap();
            let nb_nd_coord = self.get_node_payload(selected_nb_idx)?.coord;
            let dist_to_nb = nb_nd_coord.hypot(data_coord);
            if dist_to_nb > max_dist {
                if selected_nb_idx < pred_map.len() {
                    pred_map[selected_nb_idx] = Some(current_idx);
                } else {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "Internal logic error: selected_nb_idx {} out of bounds for pred_map.",
                        selected_nb_idx
                    )));
                }
                let (dist, n_opt, n_n_opt) =
                    self.closest_intersections(data_coord, pred_map.clone(), selected_nb_idx)?;
                if dist < min_dist {
                    min_dist = dist;
                    if let Some(n) = n_opt {
                        nearest_idx = n;
                        next_nearest_idx = n_n_opt;
                    }
                }
                if !reversing {
                    reversing = true;
                    pred_map.fill(None);
                    current_idx = min_idx;
                    prev_idx = None;
                    continue;
                }
                break;
            }
            if nb_idx != pred_map.get(current_idx).and_then(|&opt| opt) {
                if pred_map
                    .get(selected_nb_idx)
                    .map_or(false, |opt| opt.is_some())
                    || selected_nb_idx == min_idx
                {
                    if selected_nb_idx == min_idx {
                        if selected_nb_idx < pred_map.len() {
                            pred_map[selected_nb_idx] = Some(current_idx);
                        } else {
                            return Err(exceptions::PyValueError::new_err(format!(
                                "Internal logic error: selected_nb_idx {} out of bounds for pred_map.", selected_nb_idx
                            )));
                        }
                    }
                    let (dist, n_opt, n_n_opt) =
                        self.closest_intersections(data_coord, pred_map.clone(), selected_nb_idx)?;
                    if dist < min_dist {
                        if let Some(n) = n_opt {
                            nearest_idx = n;
                            next_nearest_idx = n_n_opt;
                        }
                    }
                    break;
                }
                if selected_nb_idx < pred_map.len() {
                    pred_map[selected_nb_idx] = Some(current_idx);
                } else {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "Internal logic error: selected_nb_idx {} out of bounds for pred_map.",
                        selected_nb_idx
                    )));
                }
            }
            prev_idx = Some(current_idx);
            current_idx = selected_nb_idx;
        }
        Ok((Some(nearest_idx), next_nearest_idx))
    }

    fn find_nearest(
        &self,
        data_coord: Coord,
        max_dist: f32,
    ) -> PyResult<(Option<usize>, f32, Option<usize>)> {
        let mut min_idx = None;
        let mut min_dist = f32::INFINITY;
        let mut next_min_idx = None;
        let mut next_min_dist = f32::INFINITY;
        for node_index in self.graph.node_indices() {
            let node_coord = self.get_node_payload(node_index.index())?.coord;
            let dist = data_coord.hypot(node_coord);
            if dist <= max_dist && dist < min_dist {
                next_min_idx = min_idx;
                next_min_dist = min_dist;
                min_idx = Some(node_index.index());
                min_dist = dist;
            } else if dist <= max_dist && dist < next_min_dist {
                next_min_idx = Some(node_index.index());
                next_min_dist = dist;
            }
        }
        Ok((min_idx, min_dist, next_min_idx))
    }

    fn road_distance(
        &self,
        data_coord: Coord,
        nd_a_idx: usize,
        nd_b_idx: usize,
    ) -> PyResult<(f32, Option<usize>, Option<usize>)> {
        let coord_a = self.get_node_payload(nd_a_idx)?.coord;
        let coord_b = self.get_node_payload(nd_b_idx)?.coord;
        let ang_a = calculate_rotation_smallest(
            data_coord.difference(coord_a),
            coord_b.difference(coord_a),
        );
        let ang_b = calculate_rotation_smallest(
            data_coord.difference(coord_b),
            coord_a.difference(coord_b),
        );
        if ang_a > 110.0 || ang_b > 110.0 {
            return Ok((f32::INFINITY, None, None));
        }
        let side_a = data_coord.hypot(coord_a);
        let side_b = data_coord.hypot(coord_b);
        let base = coord_a.hypot(coord_b);
        if base == 0.0 {
            return Ok((f32::INFINITY, None, None));
        }
        let half_perim = (side_a + side_b + base) / 2.0;
        let area_squared =
            half_perim * (half_perim - side_a) * (half_perim - side_b) * (half_perim - base);
        if area_squared < 0.0 {
            return Ok((f32::INFINITY, None, None));
        }
        let area = area_squared.sqrt();
        if area.is_nan() {
            return Ok((f32::INFINITY, None, None));
        }
        let height = area / (0.5 * base);
        if side_a < side_b {
            if ang_a > 90.0 {
                return Ok((side_a, Some(nd_a_idx), Some(nd_b_idx)));
            }
            return Ok((height, Some(nd_a_idx), Some(nd_b_idx)));
        }
        if ang_b > 90.0 {
            return Ok((side_b, Some(nd_b_idx), Some(nd_a_idx)));
        }
        Ok((height, Some(nd_b_idx), Some(nd_a_idx)))
    }

    fn closest_intersections(
        &self,
        data_coord: Coord,
        pred_map: Vec<Option<usize>>,
        last_nd_idx: usize,
    ) -> PyResult<(f32, Option<usize>, Option<usize>)> {
        let n_preds = pred_map.iter().filter(|opt| opt.is_some()).count();
        if n_preds <= 1 {
            if let Some(Some(pred_idx)) = pred_map.get(last_nd_idx) {
                return self.road_distance(data_coord, last_nd_idx, *pred_idx);
            } else {
                return Ok((f32::INFINITY, Some(last_nd_idx), None));
            }
        }
        let mut current_idx = last_nd_idx;
        let mut pred_idx = pred_map
            .get(last_nd_idx)
            .and_then(|&opt| opt)
            .ok_or_else(|| {
                exceptions::PyValueError::new_err(format!(
                    "Invalid predecessor map state: No predecessor found for last_nd_idx {} despite n_preds > 1.",
                    last_nd_idx
                ))
            })?;
        let mut nearest_idx: Option<usize> = None;
        let mut next_nearest_idx: Option<usize> = None;
        let mut min_d = f32::INFINITY;
        let first_pred = pred_idx;
        loop {
            let (height, n_idx, n_n_idx) = self.road_distance(data_coord, current_idx, pred_idx)?;
            if height < min_d {
                min_d = height;
                nearest_idx = n_idx;
                next_nearest_idx = n_n_idx;
            }
            match pred_map.get(pred_idx).and_then(|&opt| opt) {
                Some(next_pred_idx) => {
                    if next_pred_idx == first_pred {
                        break;
                    }
                    current_idx = pred_idx;
                    pred_idx = next_pred_idx;
                }
                None => break,
            }
        }
        Ok((min_d, nearest_idx, next_nearest_idx))
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
