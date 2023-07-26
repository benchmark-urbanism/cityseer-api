use crate::common::MetricResult;
use crate::common::{clip_wts_curve, clipped_beta_wt, pair_distances_and_betas, Coord};
use crate::diversity;
use crate::graph::NetworkStructure;
use indicatif::{ProgressBar, ProgressDrawTarget};
use numpy::PyArray1;
use pyo3::exceptions;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

#[pyclass]
struct AccessibilityResult {
    #[pyo3(get)]
    weighted: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    unweighted: HashMap<u32, Py<PyArray1<f32>>>,
}
#[pyclass]
struct MixedUsesHillResult {
    #[pyo3(get)]
    hill: HashMap<u32, HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    hill_weighted: HashMap<u32, HashMap<u32, Py<PyArray1<f32>>>>,
}
#[pyclass]
struct MixedUsesOtherResult {
    #[pyo3(get)]
    shannon: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    gini: HashMap<u32, Py<PyArray1<f32>>>,
}
struct ClassesState {
    count: u32,
    nearest: f32,
}
#[pyclass]
#[derive(Clone)]
pub struct DataEntry {
    #[pyo3(get)]
    data_key: String,
    #[pyo3(get)]
    coord: Coord,
    #[pyo3(get)]
    data_id: Option<String>,
    #[pyo3(get)]
    nearest_assign: Option<usize>,
    #[pyo3(get)]
    next_nearest_assign: Option<usize>,
}
#[pymethods]
impl DataEntry {
    #[new]
    fn new(
        data_key: String,
        x: f32,
        y: f32,
        data_id: Option<String>,
        nearest_assign: Option<usize>,
        next_nearest_assign: Option<usize>,
    ) -> DataEntry {
        DataEntry {
            data_key,
            coord: Coord::new(x, y),
            data_id,
            nearest_assign,
            next_nearest_assign,
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
    fn insert(
        &mut self,
        data_key: String,
        x: f32,
        y: f32,
        data_id: Option<String>,
        nearest_assign: Option<usize>,
        next_nearest_assign: Option<usize>,
    ) {
        self.entries.insert(
            data_key.clone(),
            DataEntry::new(data_key, x, y, data_id, nearest_assign, next_nearest_assign),
        );
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
        landuses_map: HashMap<String, String>,
        accessibility_keys: Vec<String>,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        angular: Option<bool>,
        spatial_tolerance: Option<u32>,
        min_threshold_wt: Option<f32>,
        jitter_scale: Option<f32>,
        pbar_disabled: Option<bool>,
    ) -> PyResult<HashMap<String, AccessibilityResult>> {
        let (distances, betas) = pair_distances_and_betas(distances, betas, min_threshold_wt)?;
        let max_dist: u32 = distances.iter().max().unwrap().clone();
        if landuses_map.len() != self.count() {
            return Err(exceptions::PyValueError::new_err(
                "The number of landuse encodings must match the number of data points",
            ));
        }
        if accessibility_keys.len() == 0 {
            return Err(exceptions::PyValueError::new_err(
                "At least one accessibility key must be specified.",
            ));
        }
        let spatial_tolerance = spatial_tolerance.unwrap_or(0);
        let max_curve_wts = clip_wts_curve(distances.clone(), betas.clone(), spatial_tolerance)?;
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
            generate the reachable classes and their respective distances
            these are non-unique - i.e. simply the class of each data point within the maximum distance
            the aggregate_to_src_idx method will choose the closer direction of approach to a data point
            from the nearest or next-nearest network node
            */
            let reachable_entries = self.aggregate_to_src_idx(
                *netw_src_idx,
                network_structure,
                max_dist,
                jitter_scale,
                angular,
            );
            for (data_key, data_dist) in reachable_entries {
                let cl_code = landuses_map[&data_key].clone();
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
        pbar.finish();
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
    fn mixed_uses(
        &self,
        network_structure: &NetworkStructure,
        landuses_map: HashMap<String, String>,
        mixed_uses_hill: Option<bool>,
        mixed_uses_other: Option<bool>,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        angular: Option<bool>,
        spatial_tolerance: Option<u32>,
        min_threshold_wt: Option<f32>,
        jitter_scale: Option<f32>,
        pbar_disabled: Option<bool>,
    ) -> PyResult<(Option<MixedUsesHillResult>, Option<MixedUsesOtherResult>)> {
        let (distances, betas) = pair_distances_and_betas(distances, betas, min_threshold_wt)?;
        let max_dist: u32 = distances.iter().max().unwrap().clone();
        if landuses_map.len() != self.count() {
            return Err(exceptions::PyValueError::new_err(
                "The number of landuse encodings must match the number of data points",
            ));
        }
        let mixed_uses_hill = mixed_uses_hill.unwrap_or(false);
        let mixed_uses_other = mixed_uses_other.unwrap_or(false);
        if !mixed_uses_hill && !mixed_uses_other {
            return Err(exceptions::PyValueError::new_err(
                "Either or both closeness and mixed-use flags is required, but both parameters are False.",
            ));
        }
        let spatial_tolerance = spatial_tolerance.unwrap_or(0);
        let max_curve_wts = clip_wts_curve(distances.clone(), betas.clone(), spatial_tolerance)?;
        // metrics
        let hill_mu: HashMap<u32, MetricResult> = vec![0, 1, 2]
            .into_iter()
            .map(|q| {
                (
                    q,
                    MetricResult::new(distances.clone(), network_structure.node_count()),
                )
            })
            .collect();
        let hill_wt_mu: HashMap<u32, MetricResult> = vec![0, 1, 2]
            .into_iter()
            .map(|q| {
                (
                    q,
                    MetricResult::new(distances.clone(), network_structure.node_count()),
                )
            })
            .collect();
        let shannon_mu = MetricResult::new(distances.clone(), network_structure.node_count());
        let gini_mu = MetricResult::new(distances.clone(), network_structure.node_count());
        // prepare unique landuse classes
        let mut classes_uniq: HashSet<String> = HashSet::new();
        for cl_code in landuses_map.values() {
            classes_uniq.insert(cl_code.clone());
        }
        // indices
        let node_indices: Vec<usize> = network_structure.node_indices();
        // pbar
        let pbar_disabled = pbar_disabled.unwrap_or(false);
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
            // reachable
            let reachable_entries = self.aggregate_to_src_idx(
                *netw_src_idx,
                network_structure,
                max_dist,
                jitter_scale,
                angular,
            );
            // counts and nearest instance of each class type by distance threshold
            let mut classes: HashMap<u32, HashMap<String, ClassesState>> = HashMap::new();
            for &dist_key in &distances {
                let mut temp: HashMap<String, ClassesState> = HashMap::new();
                for cl_code in &classes_uniq {
                    temp.insert(
                        cl_code.clone(),
                        ClassesState {
                            count: 0,
                            nearest: 0.0,
                        },
                    );
                }
                classes.insert(dist_key, temp);
            }
            // iterate reachables to calculate reachable class counts and the nearest of each specimen
            for (data_key, data_dist) in reachable_entries {
                // get the class category
                let cl_code = landuses_map[&data_key].clone();
                // iterate the distance dimensions
                for d in distances.iter() {
                    // increment class counts at respective distances if the distance is less than current dist
                    if data_dist <= *d as f32 {
                        let class_state = classes
                            .get_mut(&d)
                            .unwrap()
                            .get_mut(&cl_code.to_string())
                            .unwrap();
                        class_state.count += 1;
                        // if distance is nearer, update the nearest distance vector too
                        if data_dist < class_state.nearest {
                            class_state.nearest = data_dist;
                        }
                    }
                }
            }
            // iterate the distance dimensions
            for i in 0..distances.len() {
                let d = distances[i];
                let b = betas[i];
                let mcw = max_curve_wts[i];
                // extract counts and nearest
                let mut counts: Vec<u32> = Vec::new();
                let mut nearest: Vec<f32> = Vec::new();
                // Iterating over the classes HashMap
                for classes_state in classes[&d].values() {
                    counts.push(classes_state.count);
                    nearest.push(classes_state.nearest);
                }
                // hill
                if mixed_uses_hill {
                    hill_mu[&0].metric[i][*netw_src_idx].fetch_add(
                        diversity::hill_diversity(counts.clone(), 0.0).unwrap(),
                        Ordering::Relaxed,
                    );
                    hill_mu[&1].metric[i][*netw_src_idx].fetch_add(
                        diversity::hill_diversity(counts.clone(), 1.0).unwrap(),
                        Ordering::Relaxed,
                    );
                    hill_mu[&2].metric[i][*netw_src_idx].fetch_add(
                        diversity::hill_diversity(counts.clone(), 2.0).unwrap(),
                        Ordering::Relaxed,
                    );
                    hill_wt_mu[&0].metric[i][*netw_src_idx].fetch_add(
                        diversity::hill_diversity_branch_distance_wt(
                            counts.clone(),
                            nearest.clone(),
                            0.0,
                            b,
                            mcw,
                        )
                        .unwrap(),
                        Ordering::Relaxed,
                    );
                    hill_wt_mu[&1].metric[i][*netw_src_idx].fetch_add(
                        diversity::hill_diversity_branch_distance_wt(
                            counts.clone(),
                            nearest.clone(),
                            1.0,
                            b,
                            mcw,
                        )
                        .unwrap(),
                        Ordering::Relaxed,
                    );
                    hill_wt_mu[&2].metric[i][*netw_src_idx].fetch_add(
                        diversity::hill_diversity_branch_distance_wt(
                            counts.clone(),
                            nearest.clone(),
                            2.0,
                            b,
                            mcw,
                        )
                        .unwrap(),
                        Ordering::Relaxed,
                    );
                }
                if mixed_uses_other {
                    shannon_mu.metric[i][*netw_src_idx].fetch_add(
                        diversity::shannon_diversity(counts.clone()).unwrap(),
                        Ordering::Relaxed,
                    );
                    gini_mu.metric[i][*netw_src_idx].fetch_add(
                        diversity::gini_simpson_diversity(counts.clone()).unwrap(),
                        Ordering::Relaxed,
                    );
                }
            }
        });
        pbar.finish();
        let mu_hill_result: Option<MixedUsesHillResult> = if mixed_uses_hill {
            let mut hill: HashMap<u32, HashMap<u32, Py<PyArray1<f32>>>> = HashMap::new();
            let mut hill_weighted: HashMap<u32, HashMap<u32, Py<PyArray1<f32>>>> = HashMap::new();
            for q_key in vec![0, 1, 2].iter() {
                hill.insert(q_key.clone(), hill_mu[q_key].load());
                hill_weighted.insert(q_key.clone(), hill_mu[q_key].load());
            }
            Some(MixedUsesHillResult {
                hill,
                hill_weighted,
            })
        } else {
            None
        };
        let mu_other_result: Option<MixedUsesOtherResult> = if mixed_uses_other {
            Some(MixedUsesOtherResult {
                shannon: shannon_mu.load(),
                gini: gini_mu.load(),
            })
        } else {
            None
        };
        Ok((mu_hill_result, mu_other_result))
    }
}
