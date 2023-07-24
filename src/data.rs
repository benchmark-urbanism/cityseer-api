use crate::common::Coord;
use crate::graph::NetworkStructure;
use pyo3::prelude::*;
use std::collections::HashMap;

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
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        let angular = angular.unwrap_or(false);
        // track reachable entries
        let mut entries: HashMap<String, f32> = HashMap::new();
        // shortest paths
        let (_visited_nodes, _visited_edges, tree_map, _edge_map) = network_structure
            .shortest_path_tree(netw_src_idx, max_dist, Some(angular), Some(jitter_scale));
        // iterate data entries
        for (data_key, data_val) in &self.entries {
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
                    let total_dist = node_visit.short_dist + d_d;
                    // if still within max
                    if total_dist <= max_dist as f32 {
                        // check if the data has an identifier
                        // e.g. a park denoted by multiple entry points
                        // if not then go ahead and insert
                        let mut update = true;
                        if !data_val.data_id.is_none() {
                            let current_dist = entries.get(data_key);
                            if current_dist.is_some() && total_dist > *current_dist.unwrap() {
                                update = false
                            }
                        };
                        if update == true {
                            entries.insert(data_key.clone(), total_dist);
                        };
                    }
                }
            };
            // the next-nearest may offer a closer route depending on the direction the shortest path approaches from
            // the code is similar to above
            if !data_val.next_nearest_assign.is_none() {
                let node_visit = tree_map[data_val.next_nearest_assign.unwrap()];
                if node_visit.short_dist < max_dist as f32 {
                    let node_payload = network_structure
                        .get_node_payload(data_val.next_nearest_assign.unwrap())
                        .unwrap();
                    let d_d = data_val.coord.hypot(node_payload.coord);
                    let total_dist = node_visit.short_dist + d_d;
                    if total_dist <= max_dist as f32 {
                        let mut update = true;
                        if !data_val.data_id.is_none() {
                            let current_dist = entries.get(data_key);
                            if current_dist.is_some() && total_dist > *current_dist.unwrap() {
                                update = false
                            }
                        };
                        if update == true {
                            entries.insert(data_key.clone(), total_dist);
                        };
                    }
                }
            };
        }
        entries
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
