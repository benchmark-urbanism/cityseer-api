use crate::common::MetricResult;
use crate::common::{clipped_beta_wt, pair_distances_and_betas, Coord};
use crate::graph::NetworkStructure;
use indicatif::{ProgressBar, ProgressDrawTarget};
use numpy::PyArray1;
use pyo3::exceptions;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

#[pyclass]
#[derive(Clone)]
pub struct DataEntry {
    #[pyo3(get)]
    data_key: String,
    #[pyo3(get)]
    coord: Coord,
    #[pyo3(get)]
    nearest_assign: Option<usize>,
    #[pyo3(get)]
    next_nearest_assign: Option<usize>,
    #[pyo3(get)]
    data_id: Option<String>,
}
#[pymethods]
impl DataEntry {
    #[new]
    fn new(data_key: String, x: f32, y: f32, data_id: Option<String>) -> DataEntry {
        DataEntry {
            data_key,
            coord: Coord::new(x, y),
            nearest_assign: None,
            next_nearest_assign: None,
            data_id,
        }
    }
    fn is_assigned(&self) -> bool {
        !self.nearest_assign.is_none()
    }
}
#[pyclass]
#[derive(Clone)]
pub struct DataMap {
    #[pyo3(get)]
    entries: HashMap<String, DataEntry>,
}
#[pyclass]
struct AccessibilityResult {
    #[pyo3(get)]
    weighted: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    unweighted: HashMap<u32, Py<PyArray1<f32>>>,
}
#[pymethods]
impl DataMap {
    #[new]
    fn new() -> DataMap {
        DataMap {
            entries: HashMap::new(),
        }
    }
    fn insert(&mut self, data_key: String, x: f32, y: f32, data_id: Option<String>) {
        self.entries
            .insert(data_key.clone(), DataEntry::new(data_key, x, y, data_id));
    }
    fn entry_keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }
    fn get_entry(&self, data_key: &str) -> Option<DataEntry> {
        let entry = self.entries.get(data_key);
        if entry.is_some() {
            return Some(entry.unwrap().clone());
        }
        None
    }
    fn get_data_coord(&self, data_key: &str) -> Option<Coord> {
        let entry = self.entries.get(data_key);
        if entry.is_some() {
            return Some(entry.unwrap().coord);
        }
        None
    }
    #[getter]
    fn count(&self) -> usize {
        self.entries.len()
    }
    #[getter]
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    fn all_assigned(&self) -> bool {
        for (_item_key, item_val) in &self.entries {
            if item_val.is_assigned() == false {
                return false;
            }
        }
        true
    }
    fn none_assigned(&self) -> bool {
        for (_item_key, item_val) in &self.entries {
            if item_val.is_assigned() {
                return false;
            }
        }
        true
    }
    fn set_nearest_assign(&mut self, data_key: String, assign_idx: usize) {
        if let Some(entry) = self.entries.get_mut(&data_key) {
            entry.nearest_assign = Some(assign_idx);
        }
    }
    fn set_next_nearest_assign(&mut self, data_key: String, assign_idx: usize) {
        if let Some(entry) = self.entries.get_mut(&data_key) {
            entry.next_nearest_assign = Some(assign_idx);
        }
    }
    fn aggregate_to_src_idx(
        &self,
        netw_src_idx: usize,
        network_structure: &NetworkStructure,
        max_dist: u32,
        jitter_scale: Option<f32>,
        angular: Option<bool>,
    ) -> HashMap<String, f32> {
        /*
        Aggregate data points relative to a src index.
        Shortest tree dijkstra returns predecessor map is based on impedance heuristic - i.e. angular vs not angular.
        Shortest path distances are in metres and are used for defining max distances regardless.
        # this function is typically called iteratively, so do type checks from parent methods
        # run the shortest tree dijkstra
        # keep in mind that predecessor map is based on impedance heuristic - which can be different from metres
        # NOTE -> use np.inf for max distance so as to explore all paths
        # In some cases the predecessor nodes will be within reach even if the closest node is not
        # Total distance is checked later
        */
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        let angular = angular.unwrap_or(false);
        // track reachable entries
        let mut entries: HashMap<String, f32> = HashMap::new();
        // track nearest instance for each data id
        // e.g. a park denoted by multiple entry points
        // this is for sifting out the closest entry point
        let mut nearest_ids: HashMap<String, (String, f32)> = HashMap::new();
        // shortest paths
        let (_visited_nodes, _visited_edges, tree_map, _edge_map) = network_structure
            .shortest_path_tree(netw_src_idx, max_dist, Some(angular), Some(jitter_scale));
        // iterate data entries
        for (data_key, data_val) in &self.entries {
            let mut nearest_total_dist = f32::INFINITY;
            let mut next_nearest_total_dist = f32::INFINITY;
            // see if it has been assigned or not
            if !data_val.nearest_assign.is_none() {
                // find the corresponding network node
                let node_visit = tree_map[data_val.nearest_assign.unwrap()];
                // proceed if this node is within max dist
                if node_visit.short_dist < max_dist as f32 {
                    // get the node's payload
                    let node_payload = network_structure
                        .get_node_payload(data_val.nearest_assign.unwrap())
                        .unwrap();
                    // calculate the distance more precisely
                    let d_d = data_val.coord.hypot(node_payload.coord);
                    nearest_total_dist = node_visit.short_dist + d_d;
                }
            };
            // the next-nearest may offer a closer route depending on the direction of shortest path approach
            if !data_val.next_nearest_assign.is_none() {
                let node_visit = tree_map[data_val.next_nearest_assign.unwrap()];
                if node_visit.short_dist < max_dist as f32 {
                    let node_payload = network_structure
                        .get_node_payload(data_val.next_nearest_assign.unwrap())
                        .unwrap();
                    let d_d = data_val.coord.hypot(node_payload.coord);
                    next_nearest_total_dist = node_visit.short_dist + d_d;
                }
            };
            // if still within max
            if nearest_total_dist <= max_dist as f32 || next_nearest_total_dist <= max_dist as f32 {
                // select from less of two
                let total_dist = if nearest_total_dist <= next_nearest_total_dist {
                    nearest_total_dist
                } else {
                    next_nearest_total_dist
                };
                // check if the data has an identifier
                // if so, deduplicate, keeping only the shortest path to the data identifier
                let mut update = true;
                if !data_val.data_id.is_none() {
                    let data_id: String = data_val.data_id.clone().unwrap();
                    if let Some((current_data_key, current_data_dist)) = nearest_ids.get(&data_id) {
                        // if existing entry, and that entry is nearer, then don't update
                        // otherwise, remove the existing entry from entries
                        if *current_data_dist < total_dist {
                            update = false;
                        } else {
                            // the nearest_ids update (below) will overwrite the entry
                            entries.remove(current_data_key);
                        }
                    }
                };
                if update == true {
                    entries.insert(data_key.clone(), total_dist);
                    if !data_val.data_id.is_none() {
                        nearest_ids.insert(
                            data_val.data_id.clone().unwrap(),
                            (data_key.clone(), total_dist),
                        );
                    }
                };
            }
        }
        entries
    }
    fn accessibility(
        &self,
        network_structure: &NetworkStructure,
        landuse_encodings: HashMap<String, String>,
        accessibility_keys: Vec<String>,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        angular: Option<bool>,
        max_curve_wts: Option<Vec<f32>>,
        min_threshold_wt: Option<f32>,
        jitter: Option<f32>,
        pbar_disabled: Option<bool>,
    ) -> PyResult<HashMap<String, AccessibilityResult>> {
        let (distances, betas) = pair_distances_and_betas(distances, betas, min_threshold_wt)?;
        let max_dist: u32 = distances.iter().max().unwrap().clone();
        if landuse_encodings.len() != self.count() {
            return Err(exceptions::PyValueError::new_err(
                "The number of landuse encodings must match the number of data points",
            ));
        }
        if accessibility_keys.len() == 0 {
            return Err(exceptions::PyValueError::new_err(
                "At least one accessibility key must be specified.",
            ));
        }
        let max_curve_wts = max_curve_wts.unwrap_or(vec![1.0; distances.len()]);
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        // prepare the containers for tracking results
        let metrics: HashMap<String, MetricResult> = accessibility_keys
            .clone()
            .into_iter()
            .map(|acc_key| {
                (
                    acc_key,
                    MetricResult::new(distances.clone(), network_structure.node_count()),
                )
            })
            .collect();
        let metrics_wt: HashMap<String, MetricResult> = accessibility_keys
            .clone()
            .into_iter()
            .map(|acc_key| {
                (
                    acc_key,
                    MetricResult::new(distances.clone(), network_structure.node_count()),
                )
            })
            .collect();
        // indices
        let node_indices: Vec<usize> = network_structure.node_indices();
        // pbar
        let pbar = ProgressBar::new(node_indices.len() as u64)
            .with_message("Computing shortest path node centrality");
        if pbar_disabled {
            pbar.set_draw_target(ProgressDrawTarget::hidden())
        }
        // iter
        node_indices.par_iter().for_each(|netw_src_idx| {
            pbar.inc(1);
            // skip if not live
            if !network_structure.is_node_live(*netw_src_idx) {
                return;
            }
            /*
            # generate the reachable classes and their respective distances
            # these are non-unique - i.e. simply the class of each data point within the maximum distance
            # the aggregate_to_src_idx method will choose the closer direction of approach to a data point
            # from the nearest or next-nearest network node
            */
            let reachable_entries = self.aggregate_to_src_idx(
                *netw_src_idx,
                network_structure,
                max_dist,
                jitter,
                angular,
            );
            for (data_key, data_dist) in reachable_entries {
                let cl_code = landuse_encodings[&data_key].clone();
                if !accessibility_keys.contains(&cl_code) {
                    continue;
                }
                for i in 0..distances.len() {
                    let d = distances[i];
                    let b = betas[i];
                    let mcw = max_curve_wts[i];
                    if data_dist <= d as f32 {
                        metrics[&cl_code].metric[i][*netw_src_idx]
                            .fetch_add(1.0, Ordering::Relaxed);
                        let val_wt = clipped_beta_wt(b, mcw, data_dist);
                        metrics_wt[&cl_code].metric[i][*netw_src_idx]
                            .fetch_add(val_wt.unwrap(), Ordering::Relaxed);
                    }
                }
            }
        });
        // unpack
        let mut accessibilities: HashMap<String, AccessibilityResult> = HashMap::new();
        for acc_key in accessibility_keys.iter() {
            accessibilities.insert(
                acc_key.clone(),
                AccessibilityResult {
                    weighted: metrics_wt[acc_key].load(),
                    unweighted: metrics[acc_key].load(),
                },
            );
        }
        Ok(accessibilities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_entry() {
        let entry = DataEntry::new(1, 2);
        assert_eq!(entry.xy(), (1, 2));
        assert_eq!(entry.is_assigned(), false);

        let mut entry2 = DataEntry::new(3, 4);
        assert_eq!(entry2.is_assigned(), false);
        entry2.nearest_assign = 1;
        assert_eq!(entry2.is_assigned(), true);
    }
}
