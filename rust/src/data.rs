use crate::common::MetricResult;
use crate::common::WALKING_SPEED;
use crate::common::{
    pair_distances_and_time, parse_decay_fn, py_key_to_composite, validate_decay_fn,
    DEFAULT_DECAY_EXPR,
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

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct LanduseAccess {
    node_indices: Vec<usize>,
    count_vec: MetricResult,
    distance_vec: MetricResult,
}

#[pymethods]
impl LanduseAccess {
    #[getter]
    pub fn count(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.count_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn distance(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.distance_vec.load_compact(&self.node_indices)
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
        capacity: usize,
        lu_keys: Vec<String>,
        max_dist: u32,
    ) -> Self {
        let mut lu_map = HashMap::with_capacity(lu_keys.len());
        for lu_key in lu_keys {
            lu_map.insert(
                lu_key,
                LanduseAccess {
                    node_indices: node_indices.clone(),
                    count_vec: MetricResult::new(&distances, capacity, 0.0),
                    distance_vec: MetricResult::new(&vec![max_dist], capacity, f32::NAN),
                },
            );
        }
        AccessibilityResult {
            distances: distances.clone(),
            node_keys_py,
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
    shannon_vec: MetricResult,
    gini_vec: MetricResult,
}

impl MixedUsesResult {
    pub fn new(
        distances: Vec<u32>,
        node_keys_py: Vec<Py<PyAny>>,
        node_indices: Vec<usize>,
        capacity: usize,
    ) -> Self {
        let mut hill_vec = HashMap::new();
        for q in [0, 1, 2] {
            hill_vec.insert(q, MetricResult::new(&distances, capacity, 0.0));
        }
        MixedUsesResult {
            distances: distances.clone(),
            node_keys_py,
            node_indices: node_indices.clone(),
            hill_vec,
            shannon_vec: MetricResult::new(&distances, capacity, 0.0),
            gini_vec: MetricResult::new(&distances, capacity, 0.0),
        }
    }
}

#[pymethods]
impl MixedUsesResult {
    #[getter]
    pub fn hill(&self) -> HashMap<u32, HashMap<u32, Py<PyArray1<f64>>>> {
        self.hill_vec
            .iter()
            .map(|(q, m)| (*q, m.load_compact(&self.node_indices)))
            .collect()
    }
    #[getter]
    pub fn shannon(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.shannon_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn gini(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.gini_vec.load_compact(&self.node_indices)
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Stats {
    node_indices: Vec<usize>,
    sum_vec: MetricResult,
    mean_vec: MetricResult,
    median_vec: MetricResult,
    count_vec: MetricResult,
    variance_vec: MetricResult,
    mad_vec: MetricResult,
    max_vec: MetricResult,
    min_vec: MetricResult,
}

#[pymethods]
impl Stats {
    #[getter]
    pub fn sum(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.sum_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn mean(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.mean_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn median(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.median_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn count(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.count_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn variance(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.variance_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn mad(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.mad_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn max(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.max_vec.load_compact(&self.node_indices)
    }
    #[getter]
    pub fn min(&self) -> HashMap<u32, Py<PyArray1<f64>>> {
        self.min_vec.load_compact(&self.node_indices)
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
        capacity: usize,
        stats_n: usize,
    ) -> Self {
        let mut stats_vec = Vec::with_capacity(stats_n);
        for _ in 0..stats_n {
            stats_vec.push(Stats {
                node_indices: node_indices.clone(),
                sum_vec: MetricResult::new(&distances, capacity, 0.0),
                mean_vec: MetricResult::new(&distances, capacity, f32::NAN),
                median_vec: MetricResult::new(&distances, capacity, f32::NAN),
                count_vec: MetricResult::new(&distances, capacity, 0.0),
                variance_vec: MetricResult::new(&distances, capacity, f32::NAN),
                mad_vec: MetricResult::new(&distances, capacity, f32::NAN),
                max_vec: MetricResult::new(&distances, capacity, f32::NAN),
                min_vec: MetricResult::new(&distances, capacity, f32::NAN),
            });
        }
        StatsResult {
            distances: distances.clone(),
            node_keys_py,
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

/// Data entry for spatial analysis.
#[pyclass(skip_from_py_object)]
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
        Python::attach(|py| DataEntry {
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
/// A stored point-to-network assignment: `(data_key, offset, along, toward)`. `offset` is the
/// unsigned distance component (primal: along-street to the node + setback; dual: setback only);
/// `along` is the dual's signed-component magnitude (0 on primal); `toward` is the coordinate of
/// the street end the point leans toward, used to resolve the sign by direction of approach
/// (`None` on primal). See `graph::resolve_assignment_dist`.
pub type StoredAssignment = (String, f64, f64, Option<(f64, f64)>);

#[pyclass]
pub struct DataMap {
    #[pyo3(get)]
    entries: HashMap<String, DataEntry>,
    pub progress: Arc<AtomicUsize>,
    #[pyo3(get)]
    node_data_map: HashMap<usize, Vec<StoredAssignment>>,
}

impl DataMap {
    /// Crate-internal: number of data entries (mirrors the Python-exposed `count`).
    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Crate-internal: whether a composite data key exists.
    pub(crate) fn has_entry(&self, data_key: &str) -> bool {
        self.entries.contains_key(data_key)
    }

    /// Crate-internal: invert `node_data_map` to per-entry assignment lists,
    /// `data_key -> [(node_idx, offset, along, toward)]`. Entries with no valid assignment are
    /// absent. Used by consumers that traverse per data point (e.g. demand betweenness) rather
    /// than per network node (the aggregation direction used by the data layers).
    #[allow(clippy::type_complexity)]
    pub(crate) fn entry_assignments(
        &self,
    ) -> HashMap<String, Vec<(usize, f64, f64, Option<(f64, f64)>)>> {
        let mut out: HashMap<String, Vec<(usize, f64, f64, Option<(f64, f64)>)>> =
            HashMap::with_capacity(self.entries.len());
        for (node_idx, pairs) in &self.node_data_map {
            for (data_key, offset, along, toward) in pairs {
                out.entry(data_key.clone())
                    .or_default()
                    .push((*node_idx, *offset, *along, *toward));
            }
        }
        out
    }
}

#[pymethods]
impl DataMap {
    #[new]
    fn new() -> DataMap {
        DataMap {
            entries: HashMap::new(),
            progress: Arc::new(AtomicUsize::new(0)),
            node_data_map: HashMap::new(),
        }
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
        // Each call to find_assignments_for_entry returns Vec<PointAssignment>
        // flat_map combines these Vecs into a single Vec.
        let assignments: Vec<crate::graph::PointAssignment> = self
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
                // Returns (node_idx, data_key, offset, along, toward) per assignment.
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
        for (node_idx, data_key, offset, along, toward) in assignments {
            self.node_data_map
                .entry(node_idx)
                .or_default()
                .push((data_key, offset, along, toward));
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
        angular=None
    ))]
    fn aggregate_to_src_idx(
        &self,
        netw_src_idx: usize,
        network_structure: &NetworkStructure,
        max_walk_seconds: u32,
        speed_m_s: f32,
        angular: Option<bool>,
    ) -> PyResult<HashMap<String, f32>> {
        let angular = angular.unwrap_or(false);
        if angular {
            network_structure.validate_dual_for_angular("aggregate_to_src_idx")?;
        }
        let mut entries_result: HashMap<String, f32> = HashMap::new();
        let mut nearest_ids: HashMap<String, (String, f32)> = HashMap::new();

        // Calculate max distance based on time and speed
        let max_walk_dist = max_walk_seconds as f32 * speed_m_s;

        // Perform Dijkstra search
        let (visited_nodes, tree_map) = if !angular {
            network_structure
                .dijkstra_tree_shortest(netw_src_idx, max_walk_seconds, speed_m_s)
                .expect("pre-validated Dijkstra inputs")
        } else {
            network_structure
                .dijkstra_tree_simplest(netw_src_idx, max_walk_seconds, speed_m_s)
                .expect("pre-validated Dijkstra inputs")
        };

        // Iterate through reachable nodes only
        for &node_idx in &visited_nodes {
            let node_visit = &tree_map[node_idx];
            if node_visit.agg_seconds >= max_walk_seconds as f32 {
                continue;
            }

            // Use node_data_map for candidate_keys and dists
            let candidate_pairs = match self.node_data_map.get(&node_idx) {
                Some(pairs) => pairs,
                None => continue,
            };

            // Resolve the direction of approach once per node: which primal end the tree entered
            // this (dual) node through. `None` on primal graphs or at the source node itself.
            let entry_end = node_visit
                .pred
                .and_then(|pred| network_structure.entry_end_coord(pred, node_idx));

            // Iterate through locally relevant data keys
            for (data_key, offset, along, toward) in candidate_pairs {
                let data_entry = match self.entries.get(data_key) {
                    Some(entry) => entry,
                    None => continue,
                };

                // Calculate network distance to the current node
                let network_dist = node_visit.agg_seconds * speed_m_s;
                // Total distance: unsigned offset always added; the dual's along-street component
                // is credited or debited by direction of approach.
                let current_total_dist = crate::graph::resolve_assignment_dist(
                    network_dist,
                    *offset,
                    *along,
                    toward,
                    entry_end,
                );

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
        Ok(entries_result)
    }

    #[pyo3(signature = (
        network_structure,
        landuses_map,
        accessibility_keys,
        distances=None,
        minutes=None,
        angular=None,
        speed_m_s=None,
        decay_fn=None,
        pbar_disabled=None,
    ))]
    fn accessibility(
        &self,
        network_structure: &NetworkStructure,
        landuses_map: Py<PyAny>,
        accessibility_keys: Vec<String>,
        distances: Option<Vec<u32>>,
        minutes: Option<Vec<f32>>,
        angular: Option<bool>,
        speed_m_s: Option<f32>,
        decay_fn: Option<String>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<AccessibilityResult> {
        // Single-decay wrapper retained for backwards compatibility; delegates to
        // `accessibility_decays`, which computes one or more decays in a single traversal.
        let decay_fns = vec![decay_fn.unwrap_or_else(|| DEFAULT_DECAY_EXPR.to_string())];
        let mut results = self.accessibility_decays(
            network_structure,
            landuses_map,
            accessibility_keys,
            decay_fns,
            distances,
            minutes,
            angular,
            speed_m_s,
            pbar_disabled,
            py,
        )?;
        Ok(results.remove(0))
    }

    #[pyo3(signature = (
        network_structure,
        landuses_map,
        accessibility_keys,
        decay_fns,
        distances=None,
        minutes=None,
        angular=None,
        speed_m_s=None,
        pbar_disabled=None,
    ))]
    fn accessibility_decays(
        &self,
        network_structure: &NetworkStructure,
        landuses_map: Py<PyAny>,
        accessibility_keys: Vec<String>,
        decay_fns: Vec<String>,
        distances: Option<Vec<u32>>,
        minutes: Option<Vec<f32>>,
        angular: Option<bool>,
        speed_m_s: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<Vec<AccessibilityResult>> {
        if angular.unwrap_or(false) {
            network_structure.validate_dual_for_angular("accessibility")?;
        }
        if decay_fns.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "At least one decay function must be provided.",
            ));
        }
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, seconds) =
            pair_distances_and_time(speed_m_s, distances, minutes)?;
        let max_walk_seconds = *seconds.iter().max().unwrap();
        let max_dist = *distances
            .iter()
            .max()
            .expect("Distances should not be empty");
        let landuses_map: &Bound<'_, PyDict> = landuses_map.bind(py).cast()?;
        if landuses_map.len() != self.count() {
            return Err(exceptions::PyValueError::new_err(
                "The number of landuse encodings must match the number of data points",
            ));
        }
        let mut lu_map: HashMap<String, String> = HashMap::with_capacity(self.count());
        for (py_key, py_val) in landuses_map.iter() {
            let comp_key = py_key_to_composite(py_key.clone())?;
            let lu_val: String = py_val.extract()?;
            if self.get_entry(&comp_key).is_none() {
                return Err(exceptions::PyKeyError::new_err(format!(
                    "Data entries key missing: {}",
                    comp_key
                )));
            }
            lu_map.insert(comp_key, lu_val);
        }

        let decay_strs = decay_fns
            .iter()
            .map(|expr| validate_decay_fn(expr))
            .collect::<PyResult<Vec<_>>>()?;

        let node_keys_py = network_structure.node_keys_py(py);
        let node_indices = network_structure.node_indices();
        let accessibility_keys_set: HashSet<String> = accessibility_keys.iter().cloned().collect();
        let mut results: Vec<AccessibilityResult> = Vec::with_capacity(decay_strs.len());
        for _ in 0..decay_strs.len() {
            let nk: Vec<Py<PyAny>> = node_keys_py.iter().map(|k| k.clone_ref(py)).collect();
            results.push(AccessibilityResult::new(
                distances.clone(),
                nk,
                node_indices.clone(),
                network_structure.node_bound(),
                accessibility_keys.clone(),
                max_dist,
            ));
        }

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        let results = py.detach(move || {
            node_indices.par_iter().for_each(|netw_src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !network_structure.is_node_live_unchecked(*netw_src_idx) {
                    return;
                }
                let reachable_entries = self
                    .aggregate_to_src_idx(
                        *netw_src_idx,
                        network_structure,
                        max_walk_seconds,
                        speed_m_s,
                        angular,
                    )
                    .expect("angular topology should be pre-validated");
                // One shared traversal feeds every decay function.
                for (decay_idx, decay_str) in decay_strs.iter().enumerate() {
                    let decay = parse_decay_fn(decay_str);
                    for (data_key, data_dist) in &reachable_entries {
                        if let Some(lu_class) = lu_map.get(data_key) {
                            if !accessibility_keys_set.contains(lu_class) {
                                continue;
                            }
                            for (i, &d) in distances.iter().enumerate() {
                                if *data_dist <= d as f32 {
                                    let p = *data_dist / d as f32;
                                    let val_wt = decay(p);
                                    results[decay_idx].lu_map[lu_class].count_vec.metric[i]
                                        [*netw_src_idx]
                                        .fetch_add(val_wt as f64, AtomicOrdering::Relaxed);

                                    if d == max_dist {
                                        let current_dist = results[decay_idx].lu_map[lu_class]
                                            .distance_vec
                                            .metric[0][*netw_src_idx]
                                            .load(AtomicOrdering::Relaxed);
                                        if current_dist.is_nan()
                                            || (*data_dist as f64) < current_dist
                                        {
                                            results[decay_idx].lu_map[lu_class]
                                                .distance_vec
                                                .metric[0][*netw_src_idx]
                                                .store(*data_dist as f64, AtomicOrdering::Relaxed);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
            results
        });
        Ok(results)
    }

    #[pyo3(signature = (
        network_structure,
        landuses_map,
        distances=None,
        minutes=None,
        compute_hill=None,
        compute_shannon=None,
        compute_gini=None,
        angular=None,
        speed_m_s=None,
        decay_fn=None,
        pbar_disabled=None
    ))]
    fn mixed_uses(
        &self,
        network_structure: &NetworkStructure,
        landuses_map: Py<PyAny>,
        distances: Option<Vec<u32>>,
        minutes: Option<Vec<f32>>,
        compute_hill: Option<bool>,
        compute_shannon: Option<bool>,
        compute_gini: Option<bool>,
        angular: Option<bool>,
        speed_m_s: Option<f32>,
        decay_fn: Option<String>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<MixedUsesResult> {
        // Single-decay wrapper retained for backwards compatibility; delegates to
        // `mixed_uses_decays`, which computes one or more decays in a single traversal.
        let decay_fns = vec![decay_fn.unwrap_or_else(|| DEFAULT_DECAY_EXPR.to_string())];
        let mut results = self.mixed_uses_decays(
            network_structure,
            landuses_map,
            decay_fns,
            distances,
            minutes,
            compute_hill,
            compute_shannon,
            compute_gini,
            angular,
            speed_m_s,
            pbar_disabled,
            py,
        )?;
        Ok(results.remove(0))
    }

    #[pyo3(signature = (
        network_structure,
        landuses_map,
        decay_fns,
        distances=None,
        minutes=None,
        compute_hill=None,
        compute_shannon=None,
        compute_gini=None,
        angular=None,
        speed_m_s=None,
        pbar_disabled=None
    ))]
    fn mixed_uses_decays(
        &self,
        network_structure: &NetworkStructure,
        landuses_map: Py<PyAny>,
        decay_fns: Vec<String>,
        distances: Option<Vec<u32>>,
        minutes: Option<Vec<f32>>,
        compute_hill: Option<bool>,
        compute_shannon: Option<bool>,
        compute_gini: Option<bool>,
        angular: Option<bool>,
        speed_m_s: Option<f32>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<Vec<MixedUsesResult>> {
        if angular.unwrap_or(false) {
            network_structure.validate_dual_for_angular("mixed_uses")?;
        }
        if decay_fns.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "At least one decay function must be provided.",
            ));
        }
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, seconds) =
            pair_distances_and_time(speed_m_s, distances, minutes)?;

        let max_walk_seconds = *seconds.iter().max().unwrap();
        let landuses_map = landuses_map.bind(py).cast::<PyDict>()?;
        if landuses_map.len() != self.count() {
            return Err(exceptions::PyValueError::new_err(
                "The number of landuse encodings must match the number of data points",
            ));
        }
        let mut lu_map: HashMap<String, String> = HashMap::with_capacity(self.count());
        for (py_key, py_val) in landuses_map.iter() {
            let py_key = py_key.cast::<PyAny>()?;
            let comp_key = py_key_to_composite(py_key.clone())?;
            let lu_val: String = py_val.extract()?;
            if self.get_entry(&comp_key).is_none() {
                return Err(exceptions::PyKeyError::new_err(format!(
                    "Data entries key missing: {}",
                    comp_key
                )));
            }
            lu_map.insert(comp_key, lu_val);
        }
        let compute_hill = compute_hill.unwrap_or(true);
        let compute_shannon = compute_shannon.unwrap_or(false);
        let compute_gini = compute_gini.unwrap_or(false);
        if !(compute_hill || compute_shannon || compute_gini) {
            return Err(exceptions::PyValueError::new_err(
                "One of the compute_<measure> flags must be True, but all are currently False.",
            ));
        }
        let decay_strs = decay_fns
            .iter()
            .map(|expr| validate_decay_fn(expr))
            .collect::<PyResult<Vec<_>>>()?;

        let node_keys_py = network_structure.node_keys_py(py);
        let node_indices = network_structure.node_indices();
        let mut results: Vec<MixedUsesResult> = Vec::with_capacity(decay_strs.len());
        for _ in 0..decay_strs.len() {
            let nk: Vec<Py<PyAny>> = node_keys_py.iter().map(|k| k.clone_ref(py)).collect();
            results.push(MixedUsesResult::new(
                distances.clone(),
                nk,
                node_indices.clone(),
                network_structure.node_bound(),
            ));
        }

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        let results = py.detach(move || {
            // Build a stable ordering of unique classes for flat-Vec indexing
            let classes_vec: Vec<String> = {
                let mut uniq: HashSet<String> = HashSet::new();
                for cl_code in lu_map.values() {
                    uniq.insert(cl_code.clone());
                }
                let mut v: Vec<String> = uniq.into_iter().collect();
                v.sort();
                v
            };
            let n_classes = classes_vec.len();
            let class_to_idx: HashMap<&str, usize> = classes_vec
                .iter()
                .enumerate()
                .map(|(i, s)| (s.as_str(), i))
                .collect();

            node_indices.par_iter().for_each(|netw_src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !network_structure.is_node_live_unchecked(*netw_src_idx) {
                    return;
                }
                let reachable_entries = self
                    .aggregate_to_src_idx(
                        *netw_src_idx,
                        network_structure,
                        max_walk_seconds,
                        speed_m_s,
                        angular,
                    )
                    .expect("angular topology should be pre-validated");
                // Class counts and nearest distances are decay-independent, so they are
                // built once from the shared traversal and reused across every decay.
                // Flat arrays: [dist_idx * n_classes + class_idx]
                let n_dists = distances.len();
                let mut counts = vec![0u32; n_dists * n_classes];
                let mut nearest = vec![f32::INFINITY; n_dists * n_classes];
                for (data_key, data_dist) in &reachable_entries {
                    if let Some(lu_class) = lu_map.get(data_key) {
                        if let Some(&cls_idx) = class_to_idx.get(lu_class.as_str()) {
                            for (dist_idx, &dist_key) in distances.iter().enumerate() {
                                if *data_dist <= dist_key as f32 {
                                    let flat_idx = dist_idx * n_classes + cls_idx;
                                    counts[flat_idx] += 1;
                                    nearest[flat_idx] = nearest[flat_idx].min(*data_dist);
                                }
                            }
                        }
                    }
                }
                for (decay_idx, decay_str) in decay_strs.iter().enumerate() {
                    let decay = parse_decay_fn(decay_str);
                    for (i, &d) in distances.iter().enumerate() {
                        let offset = i * n_classes;
                        let dist_counts = &counts[offset..offset + n_classes];
                        let dist_nearest = &nearest[offset..offset + n_classes];
                        let wt_fn = |dist: f32| {
                            let p = dist / d as f32;
                            decay(p)
                        };
                        if compute_hill {
                            results[decay_idx].hill_vec[&0].metric[i][*netw_src_idx].fetch_add(
                                diversity::hill_diversity_branch_distance_wt_core(
                                    dist_counts,
                                    dist_nearest,
                                    0.0,
                                    &wt_fn,
                                )
                                .unwrap_or(0.0) as f64,
                                AtomicOrdering::Relaxed,
                            );
                            results[decay_idx].hill_vec[&1].metric[i][*netw_src_idx].fetch_add(
                                diversity::hill_diversity_branch_distance_wt_core(
                                    dist_counts,
                                    dist_nearest,
                                    1.0,
                                    &wt_fn,
                                )
                                .unwrap_or(0.0) as f64,
                                AtomicOrdering::Relaxed,
                            );
                            results[decay_idx].hill_vec[&2].metric[i][*netw_src_idx].fetch_add(
                                diversity::hill_diversity_branch_distance_wt_core(
                                    dist_counts,
                                    dist_nearest,
                                    2.0,
                                    &wt_fn,
                                )
                                .unwrap_or(0.0) as f64,
                                AtomicOrdering::Relaxed,
                            );
                        }
                        if compute_shannon {
                            results[decay_idx].shannon_vec.metric[i][*netw_src_idx].fetch_add(
                                diversity::shannon_diversity_core(dist_counts).unwrap_or(0.0) as f64,
                                AtomicOrdering::Relaxed,
                            );
                        }
                        if compute_gini {
                            results[decay_idx].gini_vec.metric[i][*netw_src_idx].fetch_add(
                                diversity::gini_simpson_diversity_core(dist_counts).unwrap_or(0.0)
                                    as f64,
                                AtomicOrdering::Relaxed,
                            );
                        }
                    }
                }
            });
            results
        });
        Ok(results)
    }
}

/// Returns the weighted median from a vector of (value, weight) pairs.
fn weighted_median(pairs: &[(f32, f32)], total_wt: f32) -> f32 {
    if pairs.is_empty() {
        return f32::NAN;
    }
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
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
        minutes=None,
        angular=None,
        speed_m_s=None,
        decay_fn=None,
        measures=None,
        pbar_disabled=None
    ))]
    fn stats(
        &self,
        network_structure: &NetworkStructure,
        numerical_maps: Vec<Py<PyAny>>,
        distances: Option<Vec<u32>>,
        minutes: Option<Vec<f32>>,
        angular: Option<bool>,
        speed_m_s: Option<f32>,
        decay_fn: Option<String>,
        measures: Option<Vec<String>>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<StatsResult> {
        // Single-decay wrapper retained for backwards compatibility; delegates to
        // `stats_decays`, which computes one or more decays in a single traversal.
        let decay_fns = vec![decay_fn.unwrap_or_else(|| DEFAULT_DECAY_EXPR.to_string())];
        let mut results = self.stats_decays(
            network_structure,
            numerical_maps,
            decay_fns,
            distances,
            minutes,
            angular,
            speed_m_s,
            measures,
            pbar_disabled,
            py,
        )?;
        Ok(results.remove(0))
    }

    #[pyo3(signature = (
        network_structure,
        numerical_maps,
        decay_fns,
        distances=None,
        minutes=None,
        angular=None,
        speed_m_s=None,
        measures=None,
        pbar_disabled=None
    ))]
    fn stats_decays(
        &self,
        network_structure: &NetworkStructure,
        numerical_maps: Vec<Py<PyAny>>,
        decay_fns: Vec<String>,
        distances: Option<Vec<u32>>,
        minutes: Option<Vec<f32>>,
        angular: Option<bool>,
        speed_m_s: Option<f32>,
        measures: Option<Vec<String>>,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<Vec<StatsResult>> {
        if angular.unwrap_or(false) {
            network_structure.validate_dual_for_angular("stats")?;
        }
        if decay_fns.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "At least one decay function must be provided.",
            ));
        }
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, seconds) =
            pair_distances_and_time(speed_m_s, distances, minutes)?;
        let max_walk_seconds = *seconds.iter().max().unwrap();
        let mut num_maps: Vec<HashMap<String, f32>> = Vec::with_capacity(numerical_maps.len());
        for numerical_map in numerical_maps.iter() {
            let numerical_map = numerical_map.bind(py).cast::<PyDict>()?;
            if numerical_map.len() != self.count() {
                return Err(exceptions::PyValueError::new_err(
                    "The number of numeric data points must match the number of data points",
                ));
            }
            let mut num_map: HashMap<String, f32> = HashMap::with_capacity(self.count());
            // ToDo check order?
            for (py_key, py_val) in numerical_map.iter() {
                let py_key = py_key.cast::<PyAny>()?;
                let comp_key = py_key_to_composite(py_key.clone())?;
                let num_val: f32 = py_val.extract()?;
                if self.get_entry(&comp_key).is_none() {
                    return Err(exceptions::PyKeyError::new_err(format!(
                        "Data entries key missing: {}",
                        comp_key
                    )));
                }
                num_map.insert(comp_key, num_val);
            }
            num_maps.push(num_map);
        }

        let decay_strs = decay_fns
            .iter()
            .map(|expr| validate_decay_fn(expr))
            .collect::<PyResult<Vec<_>>>()?;

        // Determine which statistical measures to compute. None/empty => all of them.
        const ALLOWED_MEASURES: [&str; 8] =
            ["sum", "mean", "count", "var", "median", "mad", "max", "min"];
        let measures = measures.unwrap_or_default();
        for m in &measures {
            if !ALLOWED_MEASURES.contains(&m.as_str()) {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Unknown stats measure '{}'. Allowed: {}",
                    m,
                    ALLOWED_MEASURES.join(", ")
                )));
            }
        }
        let want = |m: &str| measures.is_empty() || measures.iter().any(|x| x == m);
        let want_sum = want("sum");
        let want_mean = want("mean");
        let want_count = want("count");
        let want_var = want("var");
        let want_median = want("median");
        let want_mad = want("mad");
        let want_max = want("max");
        let want_min = want("min");
        // The weighted median (and the MAD that derives from it) is the only costly
        // measure, so only collect the per-value pairs when one of them is requested.
        let need_vals = want_median || want_mad;

        let node_keys_py = network_structure.node_keys_py(py);
        let node_indices = network_structure.node_indices();
        let mut results: Vec<StatsResult> = Vec::with_capacity(decay_strs.len());
        for _ in 0..decay_strs.len() {
            let nk: Vec<Py<PyAny>> = node_keys_py.iter().map(|k| k.clone_ref(py)).collect();
            results.push(StatsResult::new(
                distances.clone(),
                nk,
                node_indices.clone(),
                network_structure.node_bound(),
                num_maps.len(),
            ));
        }

        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        let results = py.detach(move || {
            node_indices.par_iter().for_each(|netw_src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, AtomicOrdering::Relaxed);
                }
                if !network_structure.is_node_live_unchecked(*netw_src_idx) {
                    return;
                }
                let reachable_entries = self
                    .aggregate_to_src_idx(
                        *netw_src_idx,
                        network_structure,
                        max_walk_seconds,
                        speed_m_s,
                        angular,
                    )
                    .expect("angular topology should be pre-validated");
                // One shared traversal feeds every decay function.
                for (decay_idx, decay_str) in decay_strs.iter().enumerate() {
                    let decay = parse_decay_fn(decay_str);
                    for (map_idx, num_map) in num_maps.iter().enumerate() {
                        for (i, &d) in distances.iter().enumerate() {
                            let mut vals_wts = Vec::new();
                            let mut sum_val = 0.0_f32;
                            let mut count_val = 0.0_f32;
                            let mut sum_sq_val = 0.0_f32;
                            let mut min_val = f32::NAN;
                            let mut max_val = f32::NAN;
                            for (data_key, data_dist) in &reachable_entries {
                                if *data_dist <= d as f32 {
                                    if let Some(&num) = num_map.get(data_key) {
                                        if num.is_nan() {
                                            continue; // Skip NaN values
                                        }
                                        // gather data
                                        let p = *data_dist / d as f32;
                                        let wt = decay(p);
                                        let num_wt = num * wt;
                                        // Accumulate sums and counts
                                        sum_val += num_wt;
                                        count_val += wt;
                                        sum_sq_val += wt * num * num;
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
                                        // Per-value pairs only needed for the weighted median / MAD.
                                        if need_vals {
                                            vals_wts.push((num, wt));
                                        }
                                    }
                                }
                            }
                            let stats = &results[decay_idx].stats_vec[map_idx];
                            // Only the requested measures are written; unrequested ones keep
                            // their initialised value and are not emitted on the Python side.
                            if want_sum {
                                stats.sum_vec.metric[i][*netw_src_idx]
                                    .store(sum_val as f64, AtomicOrdering::Relaxed);
                            }
                            if want_count {
                                stats.count_vec.metric[i][*netw_src_idx]
                                    .store(count_val as f64, AtomicOrdering::Relaxed);
                            }
                            if want_max {
                                stats.max_vec.metric[i][*netw_src_idx]
                                    .store(max_val as f64, AtomicOrdering::Relaxed);
                            }
                            if want_min {
                                stats.min_vec.metric[i][*netw_src_idx]
                                    .store(min_val as f64, AtomicOrdering::Relaxed);
                            }
                            // Mean is needed for variance, so compute it whenever either is wanted.
                            let mean_val = if count_val > 0.0 {
                                sum_val / count_val
                            } else {
                                f32::NAN
                            };
                            if want_mean {
                                stats.mean_vec.metric[i][*netw_src_idx]
                                    .store(mean_val as f64, AtomicOrdering::Relaxed);
                            }
                            if want_var {
                                // Ensure non-negative due to potential float inaccuracies
                                let variance_val = if count_val > 0.0 {
                                    (sum_sq_val / count_val - mean_val.powi(2)).max(0.0)
                                } else {
                                    f32::NAN
                                };
                                stats.variance_vec.metric[i][*netw_src_idx]
                                    .store(variance_val as f64, AtomicOrdering::Relaxed);
                            }
                            if need_vals {
                                // Median (weighted); MAD derives from it.
                                let median_val = weighted_median(&vals_wts, count_val);
                                if want_median {
                                    stats.median_vec.metric[i][*netw_src_idx]
                                        .store(median_val as f64, AtomicOrdering::Relaxed);
                                }
                                if want_mad {
                                    // MAD: build abs deviations with same weights; weighted median.
                                    let mad_val = if !vals_wts.is_empty()
                                        && !median_val.is_nan()
                                        && count_val > 0.0
                                    {
                                        let abs_wt: Vec<(f32, f32)> = vals_wts
                                            .iter()
                                            .map(|(v, wt)| ((v - median_val).abs(), *wt))
                                            .collect();
                                        weighted_median(&abs_wt, count_val)
                                    } else {
                                        f32::NAN
                                    };
                                    stats.mad_vec.metric[i][*netw_src_idx]
                                        .store(mad_val as f64, AtomicOrdering::Relaxed);
                                }
                            }
                        }
                    }
                }
            });
            results
        });
        Ok(results)
    }
}
