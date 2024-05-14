use crate::common::MetricResult;
use crate::common::{clip_wts_curve, clipped_beta_wt, pair_distances_and_betas, Coord};
use crate::diversity;
use crate::graph::NetworkStructure;
use numpy::PyArray1;
use pyo3::exceptions;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[pyclass]
pub struct AccessibilityResult {
    #[pyo3(get)]
    weighted: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    unweighted: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    distance: HashMap<u32, Py<PyArray1<f32>>>,
}
#[pyclass]
pub struct MixedUsesResult {
    #[pyo3(get)]
    hill: Option<HashMap<u32, HashMap<u32, Py<PyArray1<f32>>>>>,
    #[pyo3(get)]
    hill_weighted: Option<HashMap<u32, HashMap<u32, Py<PyArray1<f32>>>>>,
    #[pyo3(get)]
    shannon: Option<HashMap<u32, Py<PyArray1<f32>>>>,
    #[pyo3(get)]
    gini: Option<HashMap<u32, Py<PyArray1<f32>>>>,
}
#[pyclass]
pub struct StatsResult {
    #[pyo3(get)]
    sum: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    sum_wt: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    mean: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    mean_wt: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    count: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    count_wt: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    variance: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    variance_wt: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    max: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    min: HashMap<u32, Py<PyArray1<f32>>>,
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
    pub progress: Arc<AtomicUsize>,
}
#[pymethods]
impl DataMap {
    #[new]
    fn new() -> DataMap {
        DataMap {
            entries: HashMap::new(),
            progress: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn progress_init(&self) {
        self.progress.store(0, Ordering::Relaxed);
    }
    fn progress(&self) -> usize {
        self.progress.as_ref().load(Ordering::Relaxed)
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
    fn count(&self) -> usize {
        self.entries.len()
    }
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
        let (_visited_nodes, tree_map) = if !angular {
            network_structure.dijkstra_tree_shortest(netw_src_idx, max_dist, Some(jitter_scale))
        } else {
            network_structure.dijkstra_tree_simplest(netw_src_idx, max_dist, Some(jitter_scale))
        };
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
        landuses_map: HashMap<String, Option<String>>,
        accessibility_keys: Vec<String>,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        angular: Option<bool>,
        spatial_tolerance: Option<u32>,
        min_threshold_wt: Option<f32>,
        jitter_scale: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<HashMap<String, AccessibilityResult>> {
        let (distances, betas) = pair_distances_and_betas(distances, betas, min_threshold_wt)?;
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
        // track progress
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        // iter
        let result = py.allow_threads(move || {
            // prepare the containers for tracking results
            let metrics: HashMap<String, MetricResult> = accessibility_keys
                .clone()
                .into_iter()
                .map(|acc_key| {
                    (
                        acc_key,
                        MetricResult::new(distances.clone(), network_structure.node_count(), 0.0),
                    )
                })
                .collect();
            let metrics_wt: HashMap<String, MetricResult> = accessibility_keys
                .clone()
                .into_iter()
                .map(|acc_key| {
                    (
                        acc_key,
                        MetricResult::new(distances.clone(), network_structure.node_count(), 0.0),
                    )
                })
                .collect();
            let max_dist = distances.iter().max().copied().unwrap();
            let dists: HashMap<String, MetricResult> = accessibility_keys
                .clone()
                .into_iter()
                .map(|acc_key| {
                    (
                        acc_key,
                        MetricResult::new(
                            // single dist
                            vec![max_dist],
                            network_structure.node_count(),
                            f32::NAN,
                        ),
                    )
                })
                .collect();
            // indices
            let node_indices: Vec<usize> = network_structure.node_indices();
            // iter
            node_indices.par_iter().for_each(|netw_src_idx| {
                // progress
                if !pbar_disabled {
                    self.progress.fetch_add(1, Ordering::Relaxed);
                }
                // skip if not live
                if !network_structure.is_node_live(*netw_src_idx).unwrap() {
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
                    if cl_code.is_none() {
                        continue;
                    }
                    let lu_class = cl_code.unwrap();
                    if !accessibility_keys.contains(&lu_class) {
                        continue;
                    }
                    for i in 0..distances.len() {
                        let d = distances[i];
                        let b = betas[i];
                        let mcw = max_curve_wts[i];
                        if data_dist <= d as f32 {
                            metrics[&lu_class].metric[i][*netw_src_idx]
                                .fetch_add(1.0, Ordering::Relaxed);
                            let val_wt = clipped_beta_wt(b, mcw, data_dist);
                            metrics_wt[&lu_class].metric[i][*netw_src_idx]
                                .fetch_add(val_wt.unwrap(), Ordering::Relaxed);
                            if d == max_dist {
                                // there is a single max dist so use 0 for index
                                let current_dist = dists[&lu_class].metric[0][*netw_src_idx]
                                    .load(Ordering::Relaxed);
                                if current_dist.is_nan() || data_dist < current_dist {
                                    dists[&lu_class].metric[0][*netw_src_idx]
                                        .store(data_dist, Ordering::Relaxed);
                                }
                            }
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
                        distance: dists[acc_key].load(),
                    },
                );
            }
            accessibilities
        });
        Ok(result)
    }

    fn mixed_uses(
        &self,
        network_structure: &NetworkStructure,
        landuses_map: HashMap<String, Option<String>>,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        compute_hill: Option<bool>,
        compute_hill_weighted: Option<bool>,
        compute_shannon: Option<bool>,
        compute_gini: Option<bool>,
        angular: Option<bool>,
        spatial_tolerance: Option<u32>,
        min_threshold_wt: Option<f32>,
        jitter_scale: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<MixedUsesResult> {
        let (distances, betas) = pair_distances_and_betas(distances, betas, min_threshold_wt)?;
        let max_dist: u32 = distances.iter().max().unwrap().clone();
        if landuses_map.len() != self.count() {
            return Err(exceptions::PyValueError::new_err(
                "The number of landuse encodings must match the number of data points",
            ));
        }
        let compute_hill = compute_hill.unwrap_or(true);
        let compute_hill_weighted = compute_hill_weighted.unwrap_or(true);
        let compute_shannon = compute_shannon.unwrap_or(false);
        let compute_gini = compute_gini.unwrap_or(false);
        if !compute_hill && !compute_hill_weighted && !compute_shannon && !compute_gini {
            return Err(exceptions::PyValueError::new_err(
                "One of the compute_<measure> flags must be True, but all are currently False.",
            ));
        }
        let spatial_tolerance = spatial_tolerance.unwrap_or(0);
        let max_curve_wts = clip_wts_curve(distances.clone(), betas.clone(), spatial_tolerance)?;
        // track progress
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        // iter
        let result = py.allow_threads(move || {
            // metrics
            let hill_mu: HashMap<u32, MetricResult> = vec![0, 1, 2]
                .into_iter()
                .map(|q| {
                    (
                        q,
                        MetricResult::new(distances.clone(), network_structure.node_count(), 0.0),
                    )
                })
                .collect();
            let hill_wt_mu: HashMap<u32, MetricResult> = vec![0, 1, 2]
                .into_iter()
                .map(|q| {
                    (
                        q,
                        MetricResult::new(distances.clone(), network_structure.node_count(), 0.0),
                    )
                })
                .collect();
            let shannon_mu =
                MetricResult::new(distances.clone(), network_structure.node_count(), 0.0);
            let gini_mu = MetricResult::new(distances.clone(), network_structure.node_count(), 0.0);
            // prepare unique landuse classes
            let mut classes_uniq: HashSet<String> = HashSet::new();
            for cl_code in landuses_map.values() {
                if cl_code.is_none() {
                    continue;
                }
                classes_uniq.insert(cl_code.clone().unwrap());
            }
            // indices
            let node_indices: Vec<usize> = network_structure.node_indices();
            // iter
            node_indices.par_iter().for_each(|netw_src_idx| {
                // progress
                if !pbar_disabled {
                    self.progress.fetch_add(1, Ordering::Relaxed);
                }
                // skip if not live
                if !network_structure.is_node_live(*netw_src_idx).unwrap() {
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
                                nearest: f32::INFINITY,
                            },
                        );
                    }
                    classes.insert(dist_key, temp);
                }
                // iterate reachables to calculate reachable class counts and the nearest of each specimen
                for (data_key, data_dist) in reachable_entries {
                    // get the class category
                    let cl_code = landuses_map[&data_key].clone();
                    if cl_code.is_none() {
                        continue;
                    }
                    let lu_class = cl_code.unwrap();
                    // iterate the distance dimensions
                    for &dist_key in &distances {
                        // increment class counts at respective distances if the distance is less than current dist
                        if data_dist <= dist_key as f32 {
                            let class_state = classes
                                .get_mut(&dist_key)
                                .unwrap()
                                .get_mut(&lu_class.to_string())
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
                    if compute_hill {
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
                    }
                    if compute_hill_weighted {
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
                    if compute_shannon {
                        shannon_mu.metric[i][*netw_src_idx].fetch_add(
                            diversity::shannon_diversity(counts.clone()).unwrap(),
                            Ordering::Relaxed,
                        );
                    }
                    if compute_gini {
                        gini_mu.metric[i][*netw_src_idx].fetch_add(
                            diversity::gini_simpson_diversity(counts.clone()).unwrap(),
                            Ordering::Relaxed,
                        );
                    }
                }
            });
            let mut hill_result: Option<HashMap<u32, HashMap<u32, Py<PyArray1<f32>>>>> = None;
            if compute_hill == true {
                let mut hr: HashMap<u32, HashMap<u32, Py<PyArray1<f32>>>> = HashMap::new();
                for q_key in vec![0, 1, 2].iter() {
                    hr.insert(q_key.clone(), hill_mu[q_key].load());
                }
                hill_result = Some(hr)
            };
            let mut hill_weighted_result: Option<HashMap<u32, HashMap<u32, Py<PyArray1<f32>>>>> =
                None;
            if compute_hill_weighted {
                let mut hr = HashMap::new();
                for q_key in vec![0, 1, 2].iter() {
                    hr.insert(q_key.clone(), hill_wt_mu[q_key].load());
                }
                hill_weighted_result = Some(hr)
            };
            let shannon_result: Option<HashMap<u32, Py<PyArray1<f32>>>> = if compute_shannon {
                Some(shannon_mu.load())
            } else {
                None
            };
            let gini_result: Option<HashMap<u32, Py<PyArray1<f32>>>> = if compute_gini {
                Some(gini_mu.load())
            } else {
                None
            };
            MixedUsesResult {
                hill: hill_result,
                hill_weighted: hill_weighted_result,
                shannon: shannon_result,
                gini: gini_result,
            }
        });
        Ok(result)
    }
    fn stats(
        &self,
        network_structure: &NetworkStructure,
        numerical_map: HashMap<String, f32>,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        angular: Option<bool>,
        spatial_tolerance: Option<u32>,
        min_threshold_wt: Option<f32>,
        jitter_scale: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<StatsResult> {
        let (distances, betas) = pair_distances_and_betas(distances, betas, min_threshold_wt)?;
        let max_dist: u32 = distances.iter().max().unwrap().clone();
        if numerical_map.len() != self.count() {
            return Err(exceptions::PyValueError::new_err(
                "The number of numerical entries must match the number of data points",
            ));
        }
        let spatial_tolerance = spatial_tolerance.unwrap_or(0);
        let max_curve_wts = clip_wts_curve(distances.clone(), betas.clone(), spatial_tolerance)?;
        // track progress
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        let result = py.allow_threads(move || {
            // prepare the containers for tracking results
            let sum = MetricResult::new(distances.clone(), network_structure.node_count(), 0.0);
            let sum_wt = MetricResult::new(distances.clone(), network_structure.node_count(), 0.0);
            let count = MetricResult::new(distances.clone(), network_structure.node_count(), 0.0);
            let count_wt =
                MetricResult::new(distances.clone(), network_structure.node_count(), 0.0);
            let max = MetricResult::new(
                distances.clone(),
                network_structure.node_count(),
                f32::NEG_INFINITY,
            );
            let min = MetricResult::new(
                distances.clone(),
                network_structure.node_count(),
                f32::INFINITY,
            );
            let mean =
                MetricResult::new(distances.clone(), network_structure.node_count(), f32::NAN);
            let mean_wt =
                MetricResult::new(distances.clone(), network_structure.node_count(), f32::NAN);
            let variance =
                MetricResult::new(distances.clone(), network_structure.node_count(), f32::NAN);
            let variance_wt =
                MetricResult::new(distances.clone(), network_structure.node_count(), f32::NAN);
            // indices
            let node_indices: Vec<usize> = network_structure.node_indices();
            // iter
            node_indices.par_iter().for_each(|netw_src_idx| {
                // progress
                if !pbar_disabled {
                    self.progress.fetch_add(1, Ordering::Relaxed);
                }
                // skip if not live
                if !network_structure.is_node_live(*netw_src_idx).unwrap() {
                    return;
                }
                // generate the reachable classes and their respective distances
                let reachable_entries = self.aggregate_to_src_idx(
                    *netw_src_idx,
                    network_structure,
                    max_dist,
                    jitter_scale,
                    angular,
                );
                /*
                IDW
                the order of the loops matters because the nested aggregations happen per distance per numerical array
                iterate the reachable indices and related distances
                sort by increasing distance re: deduplication via data keys
                because these are sorted, no need to deduplicate by respective distance thresholds
                */
                for (data_key, data_dist) in reachable_entries.iter() {
                    let num = numerical_map[data_key].clone();
                    if num.is_nan() {
                        continue;
                    };
                    for i in 0..distances.len() {
                        let d = distances[i];
                        let b = betas[i];
                        let mcw = max_curve_wts[i];
                        if *data_dist <= d as f32 {
                            let wt = clipped_beta_wt(b, mcw, *data_dist).unwrap();
                            let num_wt = num * wt;
                            // agg
                            sum.metric[i][*netw_src_idx].fetch_add(num, Ordering::Relaxed);
                            sum_wt.metric[i][*netw_src_idx].fetch_add(num_wt, Ordering::Relaxed);
                            count.metric[i][*netw_src_idx].fetch_add(1.0, Ordering::Relaxed);
                            count_wt.metric[i][*netw_src_idx].fetch_add(wt, Ordering::Relaxed);
                            // not using compare_exchang in loop because only one netw_src_idx is processed at a time
                            let current_max = max.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                            if num > current_max {
                                max.metric[i][*netw_src_idx].store(num, Ordering::Relaxed);
                            };
                            let current_min = min.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                            if num < current_min {
                                min.metric[i][*netw_src_idx].store(num, Ordering::Relaxed);
                            };
                        }
                    }
                }
                // finalise mean calculations - this is happening for a single netw_src_idx, so fairly fast
                for i in 0..distances.len() {
                    let sum_val = sum.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                    let count_val = count.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                    mean.metric[i][*netw_src_idx].store(sum_val / count_val, Ordering::Relaxed);
                    let sum_wt_val = sum_wt.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                    let count_wt_val = count_wt.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                    mean_wt.metric[i][*netw_src_idx]
                        .store(sum_wt_val / count_wt_val, Ordering::Relaxed);
                    // also clean up min and max - e.g. isolated locations
                    let current_max_val = max.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                    if current_max_val.is_infinite() {
                        max.metric[i][*netw_src_idx].store(f32::NAN, Ordering::Relaxed);
                    }
                    let current_min_val = min.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                    if current_min_val.is_infinite() {
                        min.metric[i][*netw_src_idx].store(f32::NAN, Ordering::Relaxed);
                    }
                }
                // calculate variances - counts are already computed per above
                // weighted version is IDW by division through equivalently weighted counts above
                for (data_key, data_dist) in reachable_entries {
                    let num = numerical_map[&data_key].clone();
                    if num.is_nan() {
                        continue;
                    };
                    for i in 0..distances.len() {
                        let d = distances[i];
                        if data_dist <= d as f32 {
                            let current_var_val =
                                variance.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                            let current_mean_val =
                                mean.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                            let diff = num - current_mean_val;
                            if current_var_val.is_nan() {
                                variance.metric[i][*netw_src_idx]
                                    .store(diff * diff, Ordering::Relaxed);
                            } else {
                                variance.metric[i][*netw_src_idx]
                                    .fetch_add(diff * diff, Ordering::Relaxed);
                            }
                            let current_var_wt_val =
                                variance_wt.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                            let current_mean_wt_val =
                                mean_wt.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                            let diff_wt = num - current_mean_wt_val;
                            if current_var_wt_val.is_nan() {
                                variance_wt.metric[i][*netw_src_idx]
                                    .store(diff_wt * diff_wt, Ordering::Relaxed);
                            } else {
                                variance_wt.metric[i][*netw_src_idx]
                                    .fetch_add(diff_wt * diff_wt, Ordering::Relaxed);
                            }
                        }
                    }
                }
                // finalise variance calculations
                for i in 0..distances.len() {
                    let current_var_val = variance.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                    let current_count_val = count.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                    if current_count_val == 0.0 {
                        variance.metric[i][*netw_src_idx].store(f32::NAN, Ordering::Relaxed);
                    } else {
                        variance.metric[i][*netw_src_idx]
                            .store(current_var_val / current_count_val, Ordering::Relaxed);
                    }
                    let current_var_wt_val =
                        variance_wt.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                    let current_count_wt_val =
                        count_wt.metric[i][*netw_src_idx].load(Ordering::Relaxed);
                    if current_count_wt_val == 0.0 {
                        variance_wt.metric[i][*netw_src_idx].store(f32::NAN, Ordering::Relaxed);
                    } else {
                        variance_wt.metric[i][*netw_src_idx]
                            .store(current_var_wt_val / current_count_wt_val, Ordering::Relaxed);
                    }
                }
            });
            // unpack
            StatsResult {
                sum: sum.load(),
                sum_wt: sum_wt.load(),
                mean: mean.load(),
                mean_wt: mean_wt.load(),
                count: count.load(),
                count_wt: count_wt.load(),
                variance: variance.load(),
                variance_wt: variance_wt.load(),
                max: max.load(),
                min: min.load(),
            }
        });
        Ok(result)
    }
}
