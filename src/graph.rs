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
        new_node_idx.index().try_into().unwrap()
    }
    pub fn get_node_payload(&self, node_idx: usize) -> PyResult<NodePayload> {
        let payload = self.graph.node_weight(NodeIndex::new(node_idx));
        if !payload.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "No payload for requested node idex.",
            ));
        }
        Ok(payload.unwrap().clone())
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
        self.graph.node_count().try_into().unwrap()
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
        self.graph.edge_count().try_into().unwrap()
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
        let selected_edge = self
            .graph
            .edges_connecting(
                NodeIndex::new(start_nd_idx.try_into().unwrap()),
                NodeIndex::new(end_nd_idx.try_into().unwrap()),
            )
            .find(|edge_ref| edge_ref.weight().edge_idx == edge_idx);
        if !selected_edge.is_some() {
            return Err(exceptions::PyValueError::new_err(format!(
                "Edge not found for nodes {0}, {1}, and idx {2}.",
                start_nd_idx, end_nd_idx, edge_idx
            )));
        };
        Ok(selected_edge.unwrap().weight().clone())
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
            let node_payload = self.graph.node_weight(node_idx).unwrap();
            if !node_payload.validate() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid node for node idx {:?}.",
                    node_idx
                )));
            }
        }
        for edge_idx in self.graph.edge_indices() {
            let edge_payload = self.graph.edge_weight(edge_idx).unwrap();
            if !edge_payload.validate() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid edge for edge idx {:?}.",
                    edge_idx
                )));
            }
        }
        Ok(true)
    }

    fn find_nearest(
        &self,
        data_coord: Coord,
        max_dist: f32,
    ) -> (Option<usize>, f32, Option<usize>) {
        /*
        finds the nearest road node, corresponding distance, and next nearest road node
        relative to a provided data point
        */
        let mut min_idx = None;
        let mut min_dist = std::f32::INFINITY;
        let mut next_min_idx = None;
        let mut next_min_dist = std::f32::INFINITY;
        // Iterate all nodes, find nearest
        for node_index in self.graph.node_indices() {
            let node_coord = self.graph.node_weight(node_index).unwrap().coord;
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
        (min_idx, min_dist, next_min_idx)
    }

    fn road_distance(
        &self,
        data_coord: Coord,
        nd_a_idx: usize,
        nd_b_idx: usize,
    ) -> (f32, Option<usize>, Option<usize>) {
        /*
        calculates the nearest perpendicular distance to an adjacent road
        road segment is defined by nodes a and b
        returns a and b sorted in nearest and next nearest order
        */
        let coord_a = self.get_node_payload(nd_a_idx).unwrap().coord;
        let coord_b = self.get_node_payload(nd_b_idx).unwrap().coord;
        // Get the angles from either intersection node to the data point
        // requires the vector of the difference
        let ang_a = calculate_rotation_smallest(
            data_coord.difference(coord_a),
            coord_b.difference(coord_a),
        );
        let ang_b = calculate_rotation_smallest(
            data_coord.difference(coord_b),
            coord_a.difference(coord_b),
        );
        // Assume offset street segment if either is significantly greater than 90
        // (in which case sideways offset from the road)
        if ang_a > 110.0 || ang_b > 110.0 {
            return (f32::INFINITY, None, None);
        }
        // Calculate height from two sides and included angle
        let side_a = data_coord.hypot(coord_a);
        let side_b = data_coord.hypot(coord_b);
        let base = coord_a.hypot(coord_b);
        // Forestall potential division by zero
        if base == 0.0 {
            return (f32::INFINITY, None, None);
        }
        // Heron's formula
        let half_perim = (side_a + side_b + base) / 2.0;
        let area =
            (half_perim * (half_perim - side_a) * (half_perim - side_b) * (half_perim - base))
                .sqrt();
        let height = area / (0.5 * base);
        // NOTE - the height of the triangle may be less than the distance to the nodes
        // happens due to offset segments: can cause wrong assignment where adjacent segments have the same triangle height
        // in this case, set to the length of the closest node so that height (minimum distance) is still meaningful
        // Return indices in order of nearest then the next nearest
        if side_a < side_b {
            if ang_a > 90.0 {
                return (side_a, Some(nd_a_idx), Some(nd_b_idx));
            }
            return (height, Some(nd_a_idx), Some(nd_b_idx));
        }
        if ang_b > 90.0 {
            return (side_b, Some(nd_b_idx), Some(nd_a_idx));
        }
        (height, Some(nd_b_idx), Some(nd_a_idx))
    }

    fn closest_intersections(
        &self,
        data_coord: Coord,
        pred_map: Vec<Option<usize>>,
        last_nd_idx: usize,
    ) -> (f32, Option<usize>, Option<usize>) {
        // finds the closest adjacent roadway segment and corresponding adjacent intersections
        // relative to an input data point
        let mut n_preds = 0;
        for i in 0..pred_map.len() {
            if !pred_map[i].is_none() {
                n_preds += 1;
            }
        }
        // if only one, there is no next nearest and no need to retrace
        if n_preds == 0 {
            return (f32::INFINITY, Some(last_nd_idx), None);
        }
        let mut current_idx = last_nd_idx;
        let mut pred_idx = pred_map[last_nd_idx].unwrap();
        // if only two, no need to retrace
        if n_preds == 1 {
            return self.road_distance(data_coord, current_idx, pred_idx);
        }
        let mut nearest_idx: Option<usize> = None;
        let mut next_nearest_idx: Option<usize> = None;
        let mut min_d = f32::INFINITY;
        let first_pred = pred_idx; // for finding end of loop
        loop {
            let (height, n_idx, n_n_idx) = self.road_distance(data_coord, current_idx, pred_idx);
            if height < min_d {
                min_d = height;
                nearest_idx = n_idx;
                next_nearest_idx = n_n_idx;
            }
            // break if the next item in the chain has no predecessor
            if pred_map[pred_idx].is_none() {
                break;
            }
            current_idx = pred_idx;
            pred_idx = pred_map[pred_idx].unwrap();
            if pred_idx == first_pred {
                break;
            }
        }
        (min_d, nearest_idx, next_nearest_idx)
    }

    fn assign_to_network(
        &self,
        data_coord: Coord,
        max_dist: f32,
    ) -> (Option<usize>, Option<usize>) {
        /*
        1 - find the closest network node from each data point
        2A - wind clockwise along the network to preferably find a block cycle surrounding the node
        2B - in event of topological traps, try anti-clockwise as well
        3A - select the closest block cycle node
        3B - if no enclosing cycle - simply use the closest node
        4 - find the neighbouring node that minimises the distance between the data point on "street-front"
         */
        // Find the nearest and next nearest network nodes
        let (start_min_idx, start_min_dist, start_next_min_idx) =
            self.find_nearest(data_coord, max_dist);
        // In some cases no network node will be within max_dist...
        if start_min_idx.is_none() {
            return (None, None);
        }
        let min_idx = start_min_idx.unwrap();
        // Check if min and next min are connected
        if !start_next_min_idx.is_none() {
            let next_min_idx = start_next_min_idx.unwrap();
            for nb_nd_idx in self
                .graph
                .neighbors_directed(NodeIndex::new(min_idx), Direction::Outgoing)
            {
                // If connected, then no need to circle the block
                if nb_nd_idx.index() == next_min_idx {
                    return (Some(min_idx), Some(next_min_idx));
                }
            }
        }
        // If not connected, find the nearest adjacent by edges
        // Set start node to nearest network node
        let mut current_idx = min_idx;
        // Nearest is initially set for this nearest node, but if a nearer street-edge is found, it will be overridden
        let mut nearest_idx = min_idx;
        // next nearest is None because already connected next-nearest would have returned per above
        let mut next_nearest_idx: Option<usize> = None;
        // Keep track of previous indices
        let mut prev_idx: Option<usize> = None;
        // Keep track of visited nodes
        let mut pred_map: Vec<Option<usize>> = vec![None; self.graph.node_count()];
        // min distance
        let mut min_dist = start_min_dist;
        // State for reversing direction
        let mut reversing = false;
        // Iterate neighbors
        loop {
            // Reset neighbor rotation and index counters
            let mut rotation = std::f32::NAN;
            let mut nb_idx: Option<usize> = None;
            // Iterate the edges
            for candidate_nb_idx in self
                .graph
                .neighbors_directed(NodeIndex::new(current_idx), Direction::Outgoing)
            {
                // Don't follow self-loops
                if candidate_nb_idx.index() == current_idx {
                    continue;
                }
                // Check that this isn't the previous node (already visited as neighbor from other direction)
                if !prev_idx.is_none() && candidate_nb_idx.index() == prev_idx.unwrap() {
                    continue;
                }
                // Look for the new neighbor with the smallest rightwards (anti-clockwise arctan2) angle
                // Measure the angle relative to the data point for the first node
                let candidate_nb_coord = self
                    .get_node_payload(candidate_nb_idx.index())
                    .unwrap()
                    .coord;
                let current_nd_coord = self.get_node_payload(current_idx).unwrap().coord;
                // if there is no previous index, use the data coord
                let rot = if prev_idx.is_none() {
                    calculate_rotation_smallest(
                        candidate_nb_coord.difference(current_nd_coord),
                        data_coord.difference(current_nd_coord),
                    )
                } else {
                    let prev_nd_coord = self.get_node_payload(prev_idx.unwrap()).unwrap().coord;
                    calculate_rotation_smallest(
                        candidate_nb_coord.difference(current_nd_coord),
                        prev_nd_coord.difference(current_nd_coord),
                    )
                };
                // flip rotation if reversing
                if reversing {
                    rotation = 360.0 - rot;
                }
                // If least angle, update
                if rotation.is_nan() || rot < rotation {
                    rotation = rot;
                    nb_idx = Some(candidate_nb_idx.index());
                }
            }
            // Allow backtracking if no neighbour is found - i.e., dead-ends
            if nb_idx.is_none() {
                // if no predecessor
                if pred_map[current_idx].is_none() {
                    // break loop isolated nodes with no neighbours, no predecessors, no previous
                    if prev_idx.is_none() {
                        break;
                    }
                    // For isolated edges, the algorithm gets turned-around back to the starting node with nowhere to go
                    // these have no neibours, no predecessors, but will have previous
                    // In these cases, pass closest_intersections the prev_idx so that it has a predecessor to follow
                    let (dist, n, n_n) =
                        self.closest_intersections(data_coord, pred_map, prev_idx.unwrap());
                    if dist < min_dist {
                        nearest_idx = n.unwrap();
                        next_nearest_idx = n_n;
                    }
                    break;
                }
                // Otherwise, go ahead and backtrack by finding the previous node
                nb_idx = pred_map[current_idx];
            }
            // if the distance is exceeded, reset and attempt in the other direction
            let nb_nd_coord = self.get_node_payload(nb_idx.unwrap()).unwrap().coord;
            let dist = nb_nd_coord.hypot(data_coord);
            if dist > max_dist {
                pred_map[nb_idx.unwrap()] = Some(current_idx);
                let (dist, n, n_n) =
                    self.closest_intersections(data_coord, pred_map, nb_idx.unwrap());
                // if the distance to the street edge is less than the nearest node, or than the prior closest edge
                if dist < min_dist {
                    min_dist = dist;
                    nearest_idx = n.unwrap();
                    next_nearest_idx = n_n;
                }
                // reverse and try in opposite direction
                if !reversing {
                    reversing = true;
                    pred_map = vec![None; self.graph.node_count()];
                    current_idx = min_idx;
                    prev_idx = None;
                    continue;
                }
                // otherwise break
                break;
            }
            // ignore the following conditions while backtracking
            // (if backtracking, the current node's predecessor will be equal to the new neighbour)
            if nb_idx != pred_map[current_idx] {
                // if the new nb node has already been visited then terminate, this prevents infinite loops
                // or, if the algorithm has circled the block back to the original starting node
                if !pred_map[nb_idx.unwrap()].is_none() || nb_idx.unwrap() == min_idx {
                    // set the final predecessor, BUT ONLY if re-encountered the original node
                    // this would otherwise occlude routes (e.g. backtracks) that have passed the same node twice
                    // (such routes are still able to recover the closest edge)
                    if nb_idx.unwrap() == min_idx {
                        pred_map[nb_idx.unwrap()] = Some(current_idx);
                    }
                    let (dist, n, n_n) =
                        self.closest_intersections(data_coord, pred_map, nb_idx.unwrap());
                    if dist < min_dist {
                        nearest_idx = n.unwrap();
                        next_nearest_idx = n_n;
                    }
                    break;
                }
                // set predecessor (only if not backtracking)
                pred_map[nb_idx.unwrap()] = Some(current_idx);
            }
            // otherwise, keep going
            prev_idx = Some(current_idx);
            current_idx = nb_idx.unwrap();
        }
        (Some(nearest_idx), next_nearest_idx)
    }
}

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
