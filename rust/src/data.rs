use crate::common::MetricResult;
use crate::common::WALKING_SPEED;
use crate::common::{
    clip_wts_curve, clipped_beta_wt, pair_distances_betas_time, py_key_to_composite,
};
use crate::diversity;
use crate::graph::NetworkStructure;
use core::f32;
use geo::geometry::Geometry;
use log;
use numpy::PyArray1;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use wkt::TryFromWkt;

#[pyclass]
#[derive(Clone)]
pub struct LanduseAccess {
    weighted_vec: MetricResult,
    unweighted_vec: MetricResult,
    distance_vec: MetricResult,
}

#[pymethods]
impl LanduseAccess {
    #[getter]
    pub fn weighted(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.weighted_vec.load()
    }
    #[getter]
    pub fn unweighted(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.unweighted_vec.load()
    }
    #[getter]
    pub fn distance(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.distance_vec.load()
    }
}

/// Accessibility computation result.
#[pyclass]
pub struct AccessibilityResult {
    #[pyo3(get)]
    distances: Vec<u32>,
    #[pyo3(get)]
    node_keys_py: Vec<Py<PyAny>>,
    #[pyo3(get)]
    node_indices: Vec<usize>,

    lu_map: HashMap<String, LanduseAccess>,
}

impl AccessibilityResult {
    pub fn new(
        distances: Vec<u32>,
        node_keys_py: Vec<Py<PyAny>>,
        node_indices: Vec<usize>,
        lu_keys: Vec<String>,
        max_dist: u32,
    ) -> Self {
        let len = node_indices.len();
        let mut lu_map = HashMap::with_capacity(lu_keys.len());
        for lu_key in lu_keys {
            lu_map.insert(
                lu_key,
                LanduseAccess {
                    weighted_vec: MetricResult::new(&distances, len, 0.0),
                    unweighted_vec: MetricResult::new(&distances, len, 0.0),
                    distance_vec: MetricResult::new(&vec![max_dist], len, f32::NAN),
                },
            );
        }
        AccessibilityResult {
            distances: distances.clone(),
            node_keys_py: node_keys_py,
            node_indices: node_indices.clone(),
            lu_map,
        }
    }
}

#[pymethods]
impl AccessibilityResult {
    #[getter]
    pub fn result(&self) -> HashMap<String, LanduseAccess> {
        let mut result = HashMap::new();
        for (lu_key, lu_access) in self.lu_map.iter() {
            result.insert(lu_key.clone(), lu_access.clone());
        }
        result
    }
}

/// Mixed uses computation result.
#[pyclass]
pub struct MixedUsesResult {
    #[pyo3(get)]
    distances: Vec<u32>,
    #[pyo3(get)]
    node_keys_py: Vec<Py<PyAny>>,
    #[pyo3(get)]
    node_indices: Vec<usize>,

    hill_vec: HashMap<u32, MetricResult>,
    hill_weighted_vec: HashMap<u32, MetricResult>,
    shannon_vec: MetricResult,
    gini_vec: MetricResult,
}

impl MixedUsesResult {
    pub fn new(
        distances: Vec<u32>,
        node_keys_py: Vec<Py<PyAny>>,
        node_indices: Vec<usize>,
    ) -> Self {
        let len = node_indices.len();
        let mut hill_vec = HashMap::new();
        let mut hill_weighted_vec = HashMap::new();
        for q in [0, 1, 2] {
            hill_vec.insert(q, MetricResult::new(&distances, len, 0.0));
            hill_weighted_vec.insert(q, MetricResult::new(&distances, len, 0.0));
        }
        MixedUsesResult {
            distances: distances.clone(),
            node_keys_py: node_keys_py,
            node_indices: node_indices.clone(),
            hill_vec,
            hill_weighted_vec,
            shannon_vec: MetricResult::new(&distances, len, 0.0),
            gini_vec: MetricResult::new(&distances, len, 0.0),
        }
    }
}

#[pymethods]
impl MixedUsesResult {
    #[getter]
    pub fn hill(&self) -> HashMap<u32, HashMap<u32, Py<PyArray1<f32>>>> {
        self.hill_vec.iter().map(|(q, m)| (*q, m.load())).collect()
    }
    #[getter]
    pub fn hill_weighted(&self) -> HashMap<u32, HashMap<u32, Py<PyArray1<f32>>>> {
        self.hill_weighted_vec
            .iter()
            .map(|(q, m)| (*q, m.load()))
            .collect()
    }
    #[getter]
    pub fn shannon(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.shannon_vec.load()
    }
    #[getter]
    pub fn gini(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.gini_vec.load()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Stats {
    sum_vec: MetricResult,
    sum_wt_vec: MetricResult,
    mean_vec: MetricResult,
    mean_wt_vec: MetricResult,
    median_vec: MetricResult,
    median_wt_vec: MetricResult,
    count_vec: MetricResult,
    count_wt_vec: MetricResult,
    variance_vec: MetricResult,
    variance_wt_vec: MetricResult,
    mad_vec: MetricResult,    // Median Absolute Deviation (unweighted)
    mad_wt_vec: MetricResult, // Median Absolute Deviation (weighted)
    max_vec: MetricResult,
    min_vec: MetricResult,
}

#[pymethods]
impl Stats {
    #[getter]
    pub fn sum(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.sum_vec.load()
    }
    #[getter]
    pub fn sum_wt(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.sum_wt_vec.load()
    }
    #[getter]
    pub fn mean(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.mean_vec.load()
    }
    #[getter]
    pub fn mean_wt(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.mean_wt_vec.load()
    }
    #[getter]
    pub fn median(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.median_vec.load()
    }
    #[getter]
    pub fn median_wt(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.median_wt_vec.load()
    }
    #[getter]
    pub fn count(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.count_vec.load()
    }
    #[getter]
    pub fn count_wt(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.count_wt_vec.load()
    }
    #[getter]
    pub fn variance(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.variance_vec.load()
    }
    #[getter]
    pub fn variance_wt(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.variance_wt_vec.load()
    }
    #[getter]
    pub fn mad(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.mad_vec.load()
    }
    #[getter]
    pub fn mad_wt(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.mad_wt_vec.load()
    }
    #[getter]
    pub fn max(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.max_vec.load()
    }
    #[getter]
    pub fn min(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.min_vec.load()
    }
}

/// Statistics computation result.
#[pyclass]
pub struct StatsResult {
    #[pyo3(get)]
    distances: Vec<u32>,
    #[pyo3(get)]
    node_keys_py: Vec<Py<PyAny>>,
    #[pyo3(get)]
    node_indices: Vec<usize>,

    stats_vec: Vec<Stats>,
}

impl StatsResult {
    pub fn new(
        distances: Vec<u32>,
        node_keys_py: Vec<Py<PyAny>>,
        node_indices: Vec<usize>,
        stats_n: usize,
    ) -> Self {
        let len = node_indices.len();
        let mut stats_vec = Vec::with_capacity(stats_n);
        for _ in 0..stats_n {
            stats_vec.push(Stats {
                sum_vec: MetricResult::new(&distances, len, 0.0),
                sum_wt_vec: MetricResult::new(&distances, len, 0.0),
                mean_vec: MetricResult::new(&distances, len, f32::NAN),
                mean_wt_vec: MetricResult::new(&distances, len, f32::NAN),
                median_vec: MetricResult::new(&distances, len, f32::NAN),
                median_wt_vec: MetricResult::new(&distances, len, f32::NAN),
                count_vec: MetricResult::new(&distances, len, 0.0),
                count_wt_vec: MetricResult::new(&distances, len, 0.0),
                variance_vec: MetricResult::new(&distances, len, f32::NAN),
                variance_wt_vec: MetricResult::new(&distances, len, f32::NAN),
                mad_vec: MetricResult::new(&distances, len, f32::NAN),
                mad_wt_vec: MetricResult::new(&distances, len, f32::NAN),
                max_vec: MetricResult::new(&distances, len, f32::NAN),
                min_vec: MetricResult::new(&distances, len, f32::NAN),
            });
        }
        StatsResult {
            distances: distances.clone(),
            node_keys_py: node_keys_py,
            node_indices: node_indices.clone(),
            stats_vec,
        }
    }
}

#[pymethods]
impl StatsResult {
    #[getter]
    pub fn result(&self) -> Vec<Stats> {
        self.stats_vec.clone()
    }
}

struct ClassesState {
    count: u32,
    nearest: f32,
}

/// Data entry for spatial analysis.
#[pyclass]
pub struct DataEntry {
    #[pyo3(get)]
    pub data_key_py: Py<PyAny>,
    #[pyo3(get)]
    pub data_key: String,
    #[pyo3(get)]
    pub dedupe_key_py: Py<PyAny>,
    #[pyo3(get)]
    pub dedupe_key: String,
    #[pyo3(get)]
    pub geom_wkt: String,
    pub geom: Geometry<f64>,
}

impl Clone for DataEntry {
    fn clone(&self) -> Self {
        Python::with_gil(|py| DataEntry {
            data_key_py: self.data_key_py.clone_ref(py),
            data_key: self.data_key.clone(),
            dedupe_key_py: self.dedupe_key_py.clone_ref(py),
            dedupe_key: self.dedupe_key.clone(),
            geom_wkt: self.geom_wkt.clone(),
            geom: self.geom.clone(),
        })
    }
}

#[pymethods]
impl DataEntry {
    #[new]
    #[pyo3(signature = (data_key_py, geom_wkt, dedupe_key_py=None))]
    #[inline]
    fn new(
        py: Python,
        data_key_py: Py<PyAny>,
        geom_wkt: String,
        dedupe_key_py: Option<Py<PyAny>>,
    ) -> PyResult<DataEntry> {
        let data_key = py_key_to_composite(data_key_py.bind(py).clone())?;

        // Determine the dedupe key (string and Python object)
        // If dedupe_key_py is provided, use it. Otherwise, use data_key_py.
        let (dedupe_key_py_final, dedupe_key_final) = match dedupe_key_py {
            Some(key_py) => {
                let key_str = py_key_to_composite(key_py.bind(py).clone())?;
                (key_py, key_str)
            }
            None => (data_key_py.clone_ref(py), data_key.clone()),
        };

        let geom = match Geometry::try_from_wkt_str(&geom_wkt) {
            Ok(geom) => geom,
            Err(e) => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Failed to parse WKT for key '{}': {}",
                    data_key, e
                )));
            }
        };

        Ok(DataEntry {
            data_key_py,
            data_key,
            dedupe_key_py: dedupe_key_py_final,
            dedupe_key: dedupe_key_final,
            geom_wkt,
            geom,
        })
    }
}

/// Map of data entries for spatial analysis.
#[pyclass]
pub struct DataMap {
    #[pyo3(get)]
    entries: HashMap<String, DataEntry>,
    pub progress: Arc<AtomicUsize>,
    #[pyo3(get)]
    node_data_map: HashMap<usize, Vec<(String, f64)>>, // Stores (data_key, distance_to_node)
}

#[pymethods]
impl DataMap {
    #[new]
    fn new() -> DataMap {
        let map = DataMap {
            entries: HashMap::new(),
            progress: Arc::new(AtomicUsize::new(0)),
            node_data_map: HashMap::new(),
        };
        map
    }

    pub fn progress_init(&self) {
        self.progress.store(0, AtomicOrdering::Relaxed);
    }

    fn progress(&self) -> usize {
        self.progress.load(AtomicOrdering::Relaxed)
    }

    #[pyo3(signature = (data_key_py, geom_wkt, dedupe_key_py=None))]
    fn insert(
        &mut self,
        py: Python,
        data_key_py: Py<PyAny>,
        geom_wkt: String,
        dedupe_key_py: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        // Create DataEntry first (parses WKT and stores geom internally)
        let entry = DataEntry::new(py, data_key_py, geom_wkt, dedupe_key_py)?;
        let data_key = entry.data_key.clone(); // Clone data_key for use below

        // Insert the DataEntry into the main map
        if self.entries.insert(data_key.clone(), entry).is_some() {
            log::warn!("Overwriting existing data entry for key: {}", data_key);
        }

        Ok(())
    }

    fn entry_keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    fn get_entry(&self, data_key: &str) -> Option<DataEntry> {
        // Use clone() which is implemented for DataEntry
        self.entries.get(data_key).cloned()
    }

    fn count(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Assigns data entries to network nodes based on proximity and accessibility checks.
    /// This method iterates through all data entries and uses `NetworkStructure::find_assignments_for_entry`
    /// to determine valid node assignments for each entry. The results are collected and stored
    /// in the `node_data_map`.
    #[pyo3(signature = (
        network_structure,
        max_assignment_dist,
        n_nearest_candidates,
    ))]
    pub fn assign_data_to_network(
        &mut self,
        network_structure: &NetworkStructure,
        max_assignment_dist: f64,
        n_nearest_candidates: usize,
    ) -> PyResult<()> {
        log::info!(
            "Assigning {} data entries to network nodes (max_dist: {}).",
            self.entries.len(),
            max_assignment_dist
        );

        // Collect assignments in parallel using rayon's flat_map
        // Each call to find_assignments_for_entry returns Vec<(usize, String, f64)>
        // flat_map combines these Vecs into a single Vec.
        let assignments: Vec<(usize, String, f64)> = self
            .entries
            .par_iter() // Parallel iterator over entries
            .flat_map(|(data_key, data_entry)| {
                // This closure is executed in parallel for each entry
                network_structure.find_assignments_for_entry(
                    data_key,
                    &data_entry.geom,
                    max_assignment_dist,
                    n_nearest_candidates,
                )
                // find_assignments_for_entry returns Vec<(node_idx, data_key, node_dist)>
                // We need to ensure data_key is owned if it needs to be moved across threads,
                // but find_assignments_for_entry already returns an owned String.
            })
            .collect(); // Collect all assignments into a single Vec

        log::debug!(
            "Collected {} potential node assignments from data entries.",
            assignments.len()
        );

        // Clear the existing map and rebuild it from the collected assignments.
        // This part is done sequentially after parallel collection.
        self.node_data_map.clear();
        let mut assigned_data_count = 0;
        for (node_idx, data_key, node_dist) in assignments {
            // Add the assignment (data_key, distance) to the list for the node_idx
            self.node_data_map
                .entry(node_idx)
                .or_default()
                .push((data_key, node_dist));
            assigned_data_count += 1; // Count total assignments added
        }

        log::info!(
            "Finished assigning data. {} assignments added to {} nodes.",
            assigned_data_count,
            self.node_data_map.len()
        );

        Ok(())
    }

    #[pyo3(signature = (
        netw_src_idx,
        network_structure,
        max_walk_seconds,
        speed_m_s,
        jitter_scale=None,
        angular=None
    ))]
    fn aggregate_to_src_idx(
        &self,
        netw_src_idx: usize,
        network_structure: &NetworkStructure,
        max_walk_seconds: u32,
        speed_m_s: f32,
        jitter_scale: Option<f32>,
        angular: Option<bool>,
    ) -> HashMap<String, f32> {
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        let angular = angular.unwrap_or(false);
        let mut entries_result: HashMap<String, f32> = HashMap::new();
        let mut nearest_ids: HashMap<String, (String, f32)> = HashMap::new();

        // Calculate max distance based on time and speed
        let max_walk_dist = max_walk_seconds as f32 * speed_m_s;

        // Perform Dijkstra search
        let (_, tree_map) = if !angular {
            network_structure.dijkstra_tree_shortest(
                netw_src_idx,
                max_walk_seconds,
                speed_m_s,
                Some(jitter_scale),
                None,
            )
        } else {
            network_structure.dijkstra_tree_simplest(
                netw_src_idx,
                max_walk_seconds,
                speed_m_s,
                Some(jitter_scale),
                None,
            )
        };

        // Iterate through reachable nodes
        for (node_idx, node_visit) in tree_map.iter().enumerate() {
            if node_visit.agg_seconds >= max_walk_seconds as f32 {
                continue;
            }

            // Use node_data_map for candidate_keys and dists
            let candidate_pairs = self
                .node_data_map
                .get(&node_idx)
                .cloned()
                .unwrap_or_default();

            // Iterate through locally relevant data keys
            for (data_key, data_dist) in candidate_pairs {
                let data_entry = match self.entries.get(&data_key) {
                    Some(entry) => entry,
                    None => continue,
                };

                // Calculate network distance to the current node
                let network_dist = node_visit.agg_seconds * speed_m_s;
                // Calculate total distance
                let current_total_dist = network_dist + data_dist as f32;

                // Check total distance limit
                if current_total_dist <= max_walk_dist {
                    // Apply Deduplication Logic Directly
                    let dedupe_key = &data_entry.dedupe_key;

                    match nearest_ids.entry(dedupe_key.clone()) {
                        std::collections::hash_map::Entry::Occupied(mut entry) => {
                            let (current_data_key, current_dist) = entry.get_mut();
                            // Check if the new distance is better
                            if current_total_dist < *current_dist {
                                entries_result.remove(current_data_key);
                                *current_data_key = data_key.clone();
                                *current_dist = current_total_dist; // Store distance
                                entries_result.insert(data_key.clone(), current_total_dist);
                                // Store distance
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert((data_key.clone(), current_total_dist)); // Store distance
                            entries_result.insert(data_key.clone(), current_total_dist);
                            // Store distance
                        }
                    }
                }
            }
        }
        // 12. Return the final result map (data_key -> min_distance)
        entries_result
    }

    #[pyo3(signature = (
        network_structure,
        landuses_map,
        accessibility_keys,
        distances=None,
        betas=None,
        minutes=None,
        angular=None,
        spatial_tolerance=None,
        min_threshold_wt=None,
        speed_m_s=None,
        jitter_scale=None,
        pbar_disabled=None,
    ))]
    fn accessibility(
        &self,
        network_structure: &NetworkStructure,
        landuses_map: Py<PyAny>,
        accessibility_keys: Vec<String>,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        minutes: Option<Vec<f32>>,
        angular: Option<bool>,
        spatial_tolerance: Option<u32>,
        min_threshold_wt: Option<f32>,
        speed_m_s: Option<f32>,
        jitter_scale: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<AccessibilityResult> {
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, betas, seconds) =
            pair_distances_betas_time(speed_m_s, distances, betas, minutes, min_threshold_wt)?;
        let max_walk_seconds = *seconds.iter().max().unwrap();
        let max_dist = *distances
            .iter()
            .max()
            .expect("Distances should not be empty");
        let landuses_map = landuses_map.bind(py).downcast::<PyDict>()?;
        if landuses_map.len() != self.count() {
            return Err(exceptions::PyValueError::new_err(
                "The number of landuse encodings must match the number of data points",
            ));
        }
        let mut lu_map: HashMap<String, String> = HashMap::with_capacity(self.count());
        for (py_key, py_val) in landuses_map.iter() {
            let py_key = py_key.downcast::<PyAny>()?;
            let comp_key = py_key_to_composite(py_key.clone())?;
            let lu_val: String = py_val.extract()?;
            if !self.get_entry(&comp_key).is_some() {
                return Err(exceptions::PyKeyError::new_err(format!(
                    "Data entries key missing: {}",
                    comp_key
                )));
            }
            lu_map.insert(comp_key, lu_val);
        }

        let spatial_tolerance = spatial_tolerance.unwrap_or(0);
        let max_curve_wts = clip_wts_curve(distances.clone(), betas.clone(), spatial_tolerance)?;

        let node_keys_py = network_structure.node_keys_py(py);
        let node_indices = network_structure.node_indices();
        let res = AccessibilityResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
            accessibility_keys.clone(),
            max_dist,
        );

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        let result = py.allow_threads(move || {
            node_indices.par_iter().for_each(|netw_src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !network_structure.is_node_live(*netw_src_idx) {
                    return;
                }
                let reachable_entries = self.aggregate_to_src_idx(
                    *netw_src_idx,
                    network_structure,
                    max_walk_seconds,
                    speed_m_s,
                    jitter_scale,
                    angular,
                );
                for (data_key, data_dist) in reachable_entries {
                    if let Some(lu_class) = lu_map.get(&data_key) {
                        if !accessibility_keys.contains(lu_class) {
                            continue;
                        }
                        for (i, (&d, (&b, &mcw))) in distances
                            .iter()
                            .zip(betas.iter().zip(max_curve_wts.iter()))
                            .enumerate()
                        {
                            if data_dist <= d as f32 {
                                res.lu_map[lu_class].unweighted_vec.metric[i][*netw_src_idx]
                                    .fetch_add(1.0, AtomicOrdering::Relaxed);
                                let val_wt = clipped_beta_wt(b, mcw, data_dist).unwrap_or(0.0);
                                res.lu_map[lu_class].weighted_vec.metric[i][*netw_src_idx]
                                    .fetch_add(val_wt, AtomicOrdering::Relaxed);

                                if d == max_dist {
                                    let current_dist = res.lu_map[lu_class].distance_vec.metric[0]
                                        [*netw_src_idx]
                                        .load(AtomicOrdering::Relaxed);
                                    if current_dist.is_nan() || data_dist < current_dist {
                                        res.lu_map[lu_class].distance_vec.metric[0][*netw_src_idx]
                                            .store(data_dist, AtomicOrdering::Relaxed);
                                    }
                                }
                            }
                        }
                    }
                }
            });
            res
        });
        Ok(result)
    }

    #[pyo3(signature = (
        network_structure,
        landuses_map,
        distances=None,
        betas=None,
        minutes=None,
        compute_hill=None,
        compute_hill_weighted=None,
        compute_shannon=None,
        compute_gini=None,
        angular=None,
        spatial_tolerance=None,
        min_threshold_wt=None,
        speed_m_s=None,
        jitter_scale=None,
        pbar_disabled=None
    ))]
    fn mixed_uses(
        &self,
        network_structure: &NetworkStructure,
        landuses_map: Py<PyAny>,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        minutes: Option<Vec<f32>>,
        compute_hill: Option<bool>,
        compute_hill_weighted: Option<bool>,
        compute_shannon: Option<bool>,
        compute_gini: Option<bool>,
        angular: Option<bool>,
        spatial_tolerance: Option<u32>,
        min_threshold_wt: Option<f32>,
        speed_m_s: Option<f32>,
        jitter_scale: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<MixedUsesResult> {
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, betas, seconds) =
            pair_distances_betas_time(speed_m_s, distances, betas, minutes, min_threshold_wt)?;

        let max_walk_seconds = *seconds.iter().max().unwrap();
        let landuses_map = landuses_map.bind(py).downcast::<PyDict>()?;
        if landuses_map.len() != self.count() {
            return Err(exceptions::PyValueError::new_err(
                "The number of landuse encodings must match the number of data points",
            ));
        }
        let mut lu_map: HashMap<String, String> = HashMap::with_capacity(self.count());
        for (py_key, py_val) in landuses_map.iter() {
            let py_key = py_key.downcast::<PyAny>()?;
            let comp_key = py_key_to_composite(py_key.clone())?;
            let lu_val: String = py_val.extract()?;
            if !self.get_entry(&comp_key).is_some() {
                return Err(exceptions::PyKeyError::new_err(format!(
                    "Data entries key missing: {}",
                    comp_key
                )));
            }
            lu_map.insert(comp_key, lu_val);
        }
        let compute_hill = compute_hill.unwrap_or(true);
        let compute_hill_weighted = compute_hill_weighted.unwrap_or(true);
        let compute_shannon = compute_shannon.unwrap_or(false);
        let compute_gini = compute_gini.unwrap_or(false);
        if !(compute_hill || compute_hill_weighted || compute_shannon || compute_gini) {
            return Err(exceptions::PyValueError::new_err(
                "One of the compute_<measure> flags must be True, but all are currently False.",
            ));
        }
        let spatial_tolerance = spatial_tolerance.unwrap_or(0);
        let max_curve_wts = clip_wts_curve(distances.clone(), betas.clone(), spatial_tolerance)?;

        let node_keys_py = network_structure.node_keys_py(py);
        let node_indices = network_structure.node_indices();
        let res = MixedUsesResult::new(distances.clone(), node_keys_py, node_indices.clone());

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        let result = py.allow_threads(move || {
            let mut classes_uniq: HashSet<String> = HashSet::new();
            for cl_code in lu_map.values() {
                classes_uniq.insert(cl_code.clone());
            }

            node_indices.par_iter().for_each(|netw_src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !network_structure.is_node_live(*netw_src_idx) {
                    return;
                }
                let reachable_entries = self.aggregate_to_src_idx(
                    *netw_src_idx,
                    network_structure,
                    max_walk_seconds,
                    speed_m_s,
                    jitter_scale,
                    angular,
                );
                let mut classes: HashMap<u32, HashMap<String, ClassesState>> =
                    HashMap::with_capacity(distances.len());
                for &dist_key in &distances {
                    let temp: HashMap<String, ClassesState> = classes_uniq
                        .iter()
                        .map(|cl_code| {
                            (
                                cl_code.clone(),
                                ClassesState {
                                    count: 0,
                                    nearest: f32::INFINITY,
                                },
                            )
                        })
                        .collect();
                    classes.insert(dist_key, temp);
                }
                for (data_key, data_dist) in &reachable_entries {
                    if let Some(lu_class) = lu_map.get(data_key) {
                        for &dist_key in &distances {
                            if *data_dist <= dist_key as f32 {
                                let class_state = classes
                                    .get_mut(&dist_key)
                                    .expect("Distance key should exist in classes map")
                                    .get_mut(lu_class)
                                    .expect("Land use class should exist in inner map");
                                class_state.count += 1;
                                class_state.nearest = class_state.nearest.min(*data_dist);
                            }
                        }
                    }
                }
                for (i, (&d, (&b, &mcw))) in distances
                    .iter()
                    .zip(betas.iter().zip(max_curve_wts.iter()))
                    .enumerate()
                {
                    let mut counts = Vec::with_capacity(classes[&d].len());
                    let mut nearest = Vec::with_capacity(classes[&d].len());
                    for classes_state in classes[&d].values() {
                        counts.push(classes_state.count);
                        nearest.push(classes_state.nearest);
                    }
                    if compute_hill {
                        res.hill_vec[&0].metric[i][*netw_src_idx].fetch_add(
                            diversity::hill_diversity(counts.clone(), 0.0).unwrap_or(0.0),
                            AtomicOrdering::Relaxed,
                        );
                        res.hill_vec[&1].metric[i][*netw_src_idx].fetch_add(
                            diversity::hill_diversity(counts.clone(), 1.0).unwrap_or(0.0),
                            AtomicOrdering::Relaxed,
                        );
                        res.hill_vec[&2].metric[i][*netw_src_idx].fetch_add(
                            diversity::hill_diversity(counts.clone(), 2.0).unwrap_or(0.0),
                            AtomicOrdering::Relaxed,
                        );
                    }
                    if compute_hill_weighted {
                        res.hill_weighted_vec[&0].metric[i][*netw_src_idx].fetch_add(
                            diversity::hill_diversity_branch_distance_wt(
                                counts.clone(),
                                nearest.clone(),
                                0.0,
                                b,
                                mcw,
                            )
                            .unwrap_or(0.0),
                            AtomicOrdering::Relaxed,
                        );
                        res.hill_weighted_vec[&1].metric[i][*netw_src_idx].fetch_add(
                            diversity::hill_diversity_branch_distance_wt(
                                counts.clone(),
                                nearest.clone(),
                                1.0,
                                b,
                                mcw,
                            )
                            .unwrap_or(0.0),
                            AtomicOrdering::Relaxed,
                        );
                        res.hill_weighted_vec[&2].metric[i][*netw_src_idx].fetch_add(
                            diversity::hill_diversity_branch_distance_wt(
                                counts.clone(),
                                nearest.clone(),
                                2.0,
                                b,
                                mcw,
                            )
                            .unwrap_or(0.0),
                            AtomicOrdering::Relaxed,
                        );
                    }
                    if compute_shannon {
                        res.shannon_vec.metric[i][*netw_src_idx].fetch_add(
                            diversity::shannon_diversity(counts.clone()).unwrap_or(0.0),
                            AtomicOrdering::Relaxed,
                        );
                    }
                    if compute_gini {
                        res.gini_vec.metric[i][*netw_src_idx].fetch_add(
                            diversity::gini_simpson_diversity(counts.clone()).unwrap_or(0.0),
                            AtomicOrdering::Relaxed,
                        );
                    }
                }
            });
            res
        });
        Ok(result)
    }
}

/// Returns the median of a sorted vector of f32 values.
fn median(vals: &Vec<f32>) -> f32 {
    let n = vals.len();
    if n == 0 {
        return f32::NAN;
    }
    // sort
    let mut sorted = vals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Returns the weighted median from a vector of (value, weight) pairs.
fn weighted_median(pairs: &Vec<(f32, f32)>, total_wt: f32) -> f32 {
    if pairs.is_empty() {
        return f32::NAN;
    }
    let mut sorted = pairs.clone();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if total_wt == 0.0 {
        return f32::NAN;
    }
    let midpoint = total_wt / 2.0;
    // If any single weight is more than half the total weight, it's the median
    for (val, wt) in &sorted {
        if *wt > midpoint {
            return *val;
        }
    }
    let mut agg_wt = 0.0;
    for (i, (val, wt)) in sorted.iter().enumerate() {
        agg_wt += *wt;
        if agg_wt == midpoint {
            // If the cumulative weight is exactly the midpoint, average with the next value,
            // unless it's the last element.
            return if i + 1 < sorted.len() {
                (*val + sorted[i + 1].0) / 2.0
            } else {
                *val
            };
        }
        if agg_wt > midpoint {
            return *val;
        }
    }
    // Fallback for floating point inaccuracies, should ideally not be reached with robust logic.
    sorted.last().unwrap().0
}

#[pymethods]
impl DataMap {
    #[pyo3(signature = (
        network_structure,
        numerical_maps,
        distances=None,
        betas=None,
        minutes=None,
        angular=None,
        spatial_tolerance=None,
        min_threshold_wt=None,
        speed_m_s=None,
        jitter_scale=None,
        pbar_disabled=None
    ))]
    fn stats(
        &self,
        network_structure: &NetworkStructure,
        numerical_maps: Vec<Py<PyAny>>,
        distances: Option<Vec<u32>>,
        betas: Option<Vec<f32>>,
        minutes: Option<Vec<f32>>,
        angular: Option<bool>,
        spatial_tolerance: Option<u32>,
        min_threshold_wt: Option<f32>,
        speed_m_s: Option<f32>,
        jitter_scale: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<StatsResult> {
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, betas, seconds) =
            pair_distances_betas_time(speed_m_s, distances, betas, minutes, min_threshold_wt)?;
        let max_walk_seconds = *seconds.iter().max().unwrap();
        let mut num_maps: Vec<HashMap<String, f32>> = Vec::with_capacity(numerical_maps.len());
        for numerical_map in numerical_maps.iter() {
            let numerical_map = numerical_map.bind(py).downcast::<PyDict>()?;
            if numerical_map.len() != self.count() {
                return Err(exceptions::PyValueError::new_err(
                    "The number of numeric data points must match the number of data points",
                ));
            }
            let mut num_map: HashMap<String, f32> = HashMap::with_capacity(self.count());
            // ToDo check order?
            for (py_key, py_val) in numerical_map.iter() {
                let py_key = py_key.downcast::<PyAny>()?;
                let comp_key = py_key_to_composite(py_key.clone())?;
                let num_val: f32 = py_val.extract()?;
                if !self.get_entry(&comp_key).is_some() {
                    return Err(exceptions::PyKeyError::new_err(format!(
                        "Data entries key missing: {}",
                        comp_key
                    )));
                }
                num_map.insert(comp_key, num_val);
            }
            num_maps.push(num_map);
        }

        let spatial_tolerance = spatial_tolerance.unwrap_or(0);
        let max_curve_wts = clip_wts_curve(distances.clone(), betas.clone(), spatial_tolerance)?;

        let node_keys_py = network_structure.node_keys_py(py);
        let node_indices = network_structure.node_indices();
        let res = StatsResult::new(
            distances.clone(),
            node_keys_py,
            node_indices.clone(),
            num_maps.len(),
        );

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        let result = py.allow_threads(move || {
            node_indices.par_iter().for_each(|netw_src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !network_structure.is_node_live(*netw_src_idx) {
                    return;
                }
                let reachable_entries = self.aggregate_to_src_idx(
                    *netw_src_idx,
                    network_structure,
                    max_walk_seconds,
                    speed_m_s,
                    jitter_scale,
                    angular,
                );
                for (map_idx, num_map) in num_maps.iter().enumerate() {
                    for (i, (&d, (&b, &mcw))) in distances
                        .iter()
                        .zip(betas.iter().zip(max_curve_wts.iter()))
                        .enumerate()
                    {
                        let mut vals = Vec::new();
                        let mut vals_wts = Vec::new();
                        let mut sum_val = 0.0;
                        let mut sum_wt_val = 0.0;
                        let mut count_val = 0.0;
                        let mut count_wt_val = 0.0;
                        let mut sum_sq_val = 0.0;
                        let mut sum_sq_wt_val = 0.0;
                        let mut min_val = f32::NAN;
                        let mut max_val = f32::NAN;
                        for (data_key, data_dist) in &reachable_entries {
                            if *data_dist <= d as f32 {
                                if let Some(&num) = num_map.get(data_key) {
                                    if num.is_nan() {
                                        continue; // Skip NaN values
                                    }
                                    // gather data
                                    let wt = clipped_beta_wt(b, mcw, *data_dist).unwrap_or(0.0);
                                    let num_wt = num * wt;
                                    // Accumulate sums and counts
                                    sum_val += num;
                                    sum_wt_val += num_wt;
                                    count_val += 1.0;
                                    count_wt_val += wt;
                                    sum_sq_val += num * num;
                                    sum_sq_wt_val += wt * num * num;
                                    // Max
                                    max_val = if max_val.is_nan() {
                                        num
                                    } else {
                                        max_val.max(num)
                                    };
                                    // Min
                                    min_val = if min_val.is_nan() {
                                        num
                                    } else {
                                        min_val.min(num)
                                    };
                                    // Median calcs (unweighted & weighted)
                                    vals.push(num);
                                    vals_wts.push((num, wt));
                                }
                            }
                        }
                        // Sums
                        res.stats_vec[map_idx].sum_vec.metric[i][*netw_src_idx]
                            .store(sum_val, AtomicOrdering::Relaxed);
                        res.stats_vec[map_idx].sum_wt_vec.metric[i][*netw_src_idx]
                            .store(sum_wt_val, AtomicOrdering::Relaxed);
                        // Counts
                        res.stats_vec[map_idx].count_vec.metric[i][*netw_src_idx]
                            .store(count_val, AtomicOrdering::Relaxed);
                        res.stats_vec[map_idx].count_wt_vec.metric[i][*netw_src_idx]
                            .store(count_wt_val, AtomicOrdering::Relaxed);
                        // Max
                        res.stats_vec[map_idx].max_vec.metric[i][*netw_src_idx]
                            .store(max_val, AtomicOrdering::Relaxed);
                        // Min
                        res.stats_vec[map_idx].min_vec.metric[i][*netw_src_idx]
                            .fetch_min(min_val, AtomicOrdering::Relaxed);
                        // Mean
                        let mean_val = if count_val > 0.0 {
                            sum_val / count_val
                        } else {
                            f32::NAN
                        };
                        res.stats_vec[map_idx].mean_vec.metric[i][*netw_src_idx]
                            .store(mean_val, AtomicOrdering::Relaxed);
                        // Weighted Mean
                        let mean_wt_val = if count_wt_val > 0.0 {
                            sum_wt_val / count_wt_val
                        } else {
                            f32::NAN
                        };
                        res.stats_vec[map_idx].mean_wt_vec.metric[i][*netw_src_idx]
                            .store(mean_wt_val, AtomicOrdering::Relaxed);
                        // Calculate Variance (using Welford's online algorithm principle implicitly)
                        // Variance = E[X^2] - (E[X])^2
                        // Ensure non-negative due to potential float inaccuracies
                        let variance_val = if count_val > 0.0 {
                            (sum_sq_val / count_val - mean_val.powi(2)).max(0.0)
                        } else {
                            f32::NAN
                        };
                        res.stats_vec[map_idx].variance_vec.metric[i][*netw_src_idx]
                            .store(variance_val, AtomicOrdering::Relaxed);
                        // Weighted Variance
                        // Ensure non-negative due to potential float inaccuracies
                        let variance_wt_val = if count_wt_val > 0.0 {
                            (sum_sq_wt_val / count_wt_val - mean_wt_val.powi(2)).max(0.0)
                        } else {
                            f32::NAN
                        };
                        res.stats_vec[map_idx].variance_wt_vec.metric[i][*netw_src_idx]
                            .store(variance_wt_val, AtomicOrdering::Relaxed);
                        // Calculate Median
                        let median_val = median(&vals);
                        res.stats_vec[map_idx].median_vec.metric[i][*netw_src_idx]
                            .store(median_val, AtomicOrdering::Relaxed);
                        // Weighted Median
                        let median_wt_val = weighted_median(&vals_wts, count_wt_val);
                        res.stats_vec[map_idx].median_wt_vec.metric[i][*netw_src_idx]
                            .store(median_wt_val, AtomicOrdering::Relaxed);
                        // Median Absolute Deviation (MAD)
                        let mad_val = if !vals.is_empty() && !median_val.is_nan() {
                            let abs_devs: Vec<f32> =
                                vals.iter().map(|v| (v - median_val).abs()).collect();
                            median(&abs_devs)
                        } else {
                            f32::NAN
                        };
                        res.stats_vec[map_idx].mad_vec.metric[i][*netw_src_idx]
                            .store(mad_val, AtomicOrdering::Relaxed);
                        // Weighted MAD: build abs deviations with same weights; use weighted median
                        let mad_wt_val = if !vals_wts.is_empty()
                            && !median_wt_val.is_nan()
                            && count_wt_val > 0.0
                        {
                            let abs_wt: Vec<(f32, f32)> = vals_wts
                                .iter()
                                .map(|(v, wt)| ((v - median_wt_val).abs(), *wt))
                                .collect();
                            weighted_median(&abs_wt, count_wt_val)
                        } else {
                            f32::NAN
                        };
                        res.stats_vec[map_idx].mad_wt_vec.metric[i][*netw_src_idx]
                            .store(mad_wt_val, AtomicOrdering::Relaxed);
                    }
                }
            });
            res
        });
        Ok(result)
    }
}
