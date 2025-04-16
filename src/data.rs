use crate::common::MetricResult;
use crate::common::WALKING_SPEED;
use crate::common::{clip_wts_curve, clipped_beta_wt, pair_distances_betas_time, Coord};
use crate::diversity;
use crate::graph::NetworkStructure;
use core::f32;
use numpy::PyArray1;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Node match result for a data entry.
#[pyclass]
#[derive(Clone)]
pub struct NodeMatch {
    #[pyo3(get)]
    pub idx: usize,
    #[pyo3(get)]
    pub dist: f32,
}

/// Holds nearest and next-nearest node matches for a data entry.
#[pyclass]
#[derive(Clone)]
pub struct NodeMatches {
    #[pyo3(get)]
    pub nearest: Option<NodeMatch>,
    #[pyo3(get)]
    pub next_nearest: Option<NodeMatch>,
}

/// Accessibility computation result.
#[pyclass]
pub struct AccessibilityResult {
    #[pyo3(get)]
    weighted: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    unweighted: HashMap<u32, Py<PyArray1<f32>>>,
    #[pyo3(get)]
    distance: HashMap<u32, Py<PyArray1<f32>>>,
}

/// Mixed uses computation result.
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

/// Statistics computation result.
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

/// Data entry for spatial analysis.
#[pyclass]
pub struct DataEntry {
    #[pyo3(get)]
    pub data_key_py: Py<PyAny>,
    #[pyo3(get)]
    pub data_key: String,
    #[pyo3(get)]
    pub coord: Coord,
    #[pyo3(get)]
    pub dedupe_key_py: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub dedupe_key: Option<String>,
    #[pyo3(get)]
    pub node_matches: Option<NodeMatches>,
}

impl Clone for DataEntry {
    fn clone(&self) -> Self {
        Python::with_gil(|py| DataEntry {
            data_key_py: self.data_key_py.clone_ref(py),
            data_key: self.data_key.clone(),
            coord: self.coord,
            dedupe_key_py: self.dedupe_key_py.as_ref().map(|k| k.clone_ref(py)),
            dedupe_key: self.dedupe_key.clone(),
            node_matches: self.node_matches.clone(),
        })
    }
}

/// Helper to generate a composite key from a Python object.
fn py_key_to_composite(py_obj: Bound<'_, PyAny>) -> PyResult<String> {
    let type_name = py_obj.get_type().name()?;
    let value_pystr = py_obj.str()?;
    let value_str = value_pystr.to_str()?;
    Ok(format!("{}:{}", type_name, value_str))
}

#[pymethods]
impl DataEntry {
    #[new]
    #[pyo3(signature = (data_key_py, x, y, dedupe_key_py=None))]
    #[inline]
    fn new(
        py: Python,
        data_key_py: Py<PyAny>,
        x: f32,
        y: f32,
        dedupe_key_py: Option<Py<PyAny>>,
    ) -> PyResult<DataEntry> {
        let data_key = py_key_to_composite(data_key_py.bind(py).clone())?;
        let dedupe_key = if let Some(ref key_py) = dedupe_key_py {
            Some(py_key_to_composite(key_py.bind(py).clone())?)
        } else {
            None
        };
        Ok(DataEntry {
            data_key_py,
            data_key,
            coord: Coord::new(x, y),
            dedupe_key_py,
            dedupe_key,
            node_matches: None,
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
    assigned_to_network: bool,
}

#[pymethods]
impl DataMap {
    #[new]
    fn new() -> DataMap {
        DataMap {
            entries: HashMap::new(),
            progress: Arc::new(AtomicUsize::new(0)),
            assigned_to_network: false,
        }
    }

    pub fn progress_init(&self) {
        self.progress.store(0, Ordering::Relaxed);
    }

    fn progress(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }

    #[pyo3(signature = (data_key_py, x, y, dedupe_key_py=None))]
    fn insert(
        &mut self,
        py: Python,
        data_key_py: Py<PyAny>,
        x: f32,
        y: f32,
        dedupe_key_py: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let entry = DataEntry::new(py, data_key_py, x, y, dedupe_key_py)?;
        self.entries.insert(entry.data_key.clone(), entry);
        Ok(())
    }

    fn entry_keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    fn get_entry(&self, data_key: &str) -> Option<DataEntry> {
        self.entries.get(data_key).map(|entry| entry.clone())
    }

    fn get_data_coord(&self, data_key: &str) -> Option<Coord> {
        self.entries.get(data_key).map(|entry| entry.coord)
    }

    fn count(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Assign nearest and next nearest network node (and distances) to each entry in the DataMap.
    #[pyo3(signature = (
        network_structure,
        max_dist,
        pbar_disabled=None
    ))]
    pub fn assign_to_network(
        &mut self,
        network_structure: &mut NetworkStructure,
        max_dist: f32,
        pbar_disabled: Option<bool>,
    ) -> PyResult<()> {
        network_structure.build_edge_rtree()?;
        let rtree = network_structure
            .edge_rtree
            .as_ref()
            .expect("Edge RTree has not been built.");
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();

        let keys: Vec<String> = self.entries.keys().cloned().collect();
        let results: Vec<(String, Option<NodeMatches>)> = keys
            .par_iter()
            .map(|key| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, Ordering::Relaxed);
                }
                let entry = self.entries.get(key).expect("Key must exist");
                let query = [entry.coord.x, entry.coord.y];
                let mut nearest: Option<NodeMatch> = None;
                let mut next_nearest: Option<NodeMatch> = None;
                if let Some(seg) = rtree.nearest_neighbor(&query) {
                    let a_dist = {
                        let dx = query[0] - seg.a[0];
                        let dy = query[1] - seg.a[1];
                        (dx * dx + dy * dy).sqrt()
                    };
                    let b_dist = {
                        let dx = query[0] - seg.b[0];
                        let dy = query[1] - seg.b[1];
                        (dx * dx + dy * dy).sqrt()
                    };
                    match (a_dist <= max_dist, b_dist <= max_dist) {
                        (true, true) => {
                            if a_dist <= b_dist {
                                nearest = Some(NodeMatch {
                                    idx: seg.a_idx,
                                    dist: a_dist,
                                });
                                next_nearest = Some(NodeMatch {
                                    idx: seg.b_idx,
                                    dist: b_dist,
                                });
                            } else {
                                nearest = Some(NodeMatch {
                                    idx: seg.b_idx,
                                    dist: b_dist,
                                });
                                next_nearest = Some(NodeMatch {
                                    idx: seg.a_idx,
                                    dist: a_dist,
                                });
                            }
                        }
                        (true, false) => {
                            nearest = Some(NodeMatch {
                                idx: seg.a_idx,
                                dist: a_dist,
                            });
                            next_nearest = None;
                        }
                        (false, true) => {
                            nearest = Some(NodeMatch {
                                idx: seg.b_idx,
                                dist: b_dist,
                            });
                            next_nearest = None;
                        }
                        (false, false) => {
                            nearest = None;
                            next_nearest = None;
                        }
                    }
                }
                (
                    key.clone(),
                    Some(NodeMatches {
                        nearest,
                        next_nearest,
                    }),
                )
            })
            .collect();

        for (key, node_matches) in results {
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.node_matches = node_matches;
            }
        }
        self.assigned_to_network = true;
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
    ) -> PyResult<HashMap<String, f32>> {
        if !self.assigned_to_network {
            return Err(exceptions::PyRuntimeError::new_err(
                "DataMap must be assigned to network before calling aggregate_to_src_idx. Call assign_to_network first."
            ));
        }
        let jitter_scale = jitter_scale.unwrap_or(0.0);
        let angular = angular.unwrap_or(false);
        let mut entries = HashMap::with_capacity(self.entries.len());
        let mut nearest_ids: HashMap<String, (String, f32)> = HashMap::new();

        let (_, tree_map) = if !angular {
            network_structure.dijkstra_tree_shortest(
                netw_src_idx,
                max_walk_seconds,
                speed_m_s,
                Some(jitter_scale),
            )
        } else {
            network_structure.dijkstra_tree_simplest(
                netw_src_idx,
                max_walk_seconds,
                speed_m_s,
                Some(jitter_scale),
            )
        };

        let calculate_time = |assign_idx: Option<usize>, data_val: &DataEntry| -> Option<f32> {
            assign_idx.and_then(|idx| {
                let node_visit = &tree_map[idx];
                if node_visit.agg_seconds < max_walk_seconds as f32 {
                    network_structure
                        .get_node_payload(idx)
                        .ok()
                        .map(|node_payload| {
                            let d_d = data_val.coord.hypot(node_payload.coord);
                            node_visit.agg_seconds + d_d / speed_m_s
                        })
                } else {
                    None
                }
            })
        };

        for (data_key, data_val) in &self.entries {
            let nearest_total_time = data_val
                .node_matches
                .as_ref()
                .and_then(|nm| calculate_time(nm.nearest.as_ref().map(|m| m.idx), data_val))
                .unwrap_or(f32::INFINITY);
            let next_nearest_total_time = data_val
                .node_matches
                .as_ref()
                .and_then(|nm| calculate_time(nm.next_nearest.as_ref().map(|m| m.idx), data_val))
                .unwrap_or(f32::INFINITY);

            let min_total_time = nearest_total_time.min(next_nearest_total_time);

            if min_total_time <= max_walk_seconds as f32 {
                let total_dist = min_total_time * speed_m_s;

                if let Some(dedupe_key) = &data_val.dedupe_key {
                    match nearest_ids.entry(dedupe_key.clone()) {
                        std::collections::hash_map::Entry::Occupied(mut entry) => {
                            let (current_key, current_dist) = entry.get_mut();
                            if total_dist < *current_dist {
                                entries.remove(current_key);
                                *current_key = data_key.clone();
                                *current_dist = total_dist;
                                entries.insert(data_key.clone(), total_dist);
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert((data_key.clone(), total_dist));
                            entries.insert(data_key.clone(), total_dist);
                        }
                    }
                } else {
                    entries.insert(data_key.clone(), total_dist);
                }
            }
        }
        Ok(entries)
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
        pbar_disabled=None
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
    ) -> PyResult<HashMap<String, AccessibilityResult>> {
        let (distances, betas, seconds) =
            pair_distances_betas_time(distances, betas, minutes, min_threshold_wt, speed_m_s)?;
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
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
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        let result = py.allow_threads(move || -> PyResult<_> {
            let mut metrics: HashMap<String, MetricResult> =
                HashMap::with_capacity(accessibility_keys.len());
            let mut metrics_wt: HashMap<String, MetricResult> =
                HashMap::with_capacity(accessibility_keys.len());
            let mut dists: HashMap<String, MetricResult> =
                HashMap::with_capacity(accessibility_keys.len());

            let node_count = network_structure.node_count();
            for key in &accessibility_keys {
                metrics.insert(
                    key.clone(),
                    MetricResult::new(distances.clone(), node_count, 0.0),
                );
                metrics_wt.insert(
                    key.clone(),
                    MetricResult::new(distances.clone(), node_count, 0.0),
                );
                dists.insert(
                    key.clone(),
                    MetricResult::new(vec![max_dist], node_count, f32::NAN),
                );
            }

            let node_indices = network_structure.node_indices();
            node_indices.par_iter().try_for_each(|&netw_src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, Ordering::Relaxed);
                }
                if !network_structure.is_node_live(netw_src_idx).unwrap_or(true) {
                    return Ok::<(), PyErr>(());
                }
                let reachable_entries = self.aggregate_to_src_idx(
                    netw_src_idx,
                    network_structure,
                    max_walk_seconds,
                    speed_m_s,
                    jitter_scale,
                    angular,
                )?;
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
                                metrics[lu_class].metric[i][netw_src_idx]
                                    .fetch_add(1.0, Ordering::Relaxed);
                                let val_wt = clipped_beta_wt(b, mcw, data_dist).unwrap_or(0.0);
                                metrics_wt[lu_class].metric[i][netw_src_idx]
                                    .fetch_add(val_wt, Ordering::Relaxed);

                                if d == max_dist {
                                    let current_dist = dists[lu_class].metric[0][netw_src_idx]
                                        .load(Ordering::Relaxed);
                                    if current_dist.is_nan() || data_dist < current_dist {
                                        dists[lu_class].metric[0][netw_src_idx]
                                            .store(data_dist, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(())
            })?;
            let accessibilities = accessibility_keys
                .into_iter()
                .map(|key| {
                    let result = AccessibilityResult {
                        weighted: metrics_wt[&key].load(),
                        unweighted: metrics[&key].load(),
                        distance: dists[&key].load(),
                    };
                    (key, result)
                })
                .collect();
            Ok(accessibilities)
        })?;
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
        let (distances, betas, seconds) =
            pair_distances_betas_time(distances, betas, minutes, min_threshold_wt, speed_m_s)?;
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
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
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        let result = py.allow_threads(move || -> PyResult<_> {
            let hill_mu: HashMap<u32, MetricResult> = [0, 1, 2]
                .iter()
                .map(|&q| {
                    (
                        q,
                        MetricResult::new(distances.clone(), network_structure.node_count(), 0.0),
                    )
                })
                .collect();
            let hill_wt_mu: HashMap<u32, MetricResult> = [0, 1, 2]
                .iter()
                .map(|&q| {
                    (
                        q,
                        MetricResult::new(distances.clone(), network_structure.node_count(), 0.0),
                    )
                })
                .collect();
            let shannon_mu =
                MetricResult::new(distances.clone(), network_structure.node_count(), 0.0);
            let gini_mu = MetricResult::new(distances.clone(), network_structure.node_count(), 0.0);
            let mut classes_uniq: HashSet<String> = HashSet::new();
            for cl_code in lu_map.values() {
                classes_uniq.insert(cl_code.clone());
            }
            let node_indices = network_structure.node_indices();
            node_indices.par_iter().try_for_each(|&netw_src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, Ordering::Relaxed);
                }
                if !network_structure.is_node_live(netw_src_idx).unwrap_or(true) {
                    return Ok::<(), PyErr>(());
                }
                let reachable_entries = self.aggregate_to_src_idx(
                    netw_src_idx,
                    network_structure,
                    max_walk_seconds,
                    speed_m_s,
                    jitter_scale,
                    angular,
                )?;
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
                        hill_mu[&0].metric[i][netw_src_idx].fetch_add(
                            diversity::hill_diversity(counts.clone(), 0.0).unwrap_or(0.0),
                            Ordering::Relaxed,
                        );
                        hill_mu[&1].metric[i][netw_src_idx].fetch_add(
                            diversity::hill_diversity(counts.clone(), 1.0).unwrap_or(0.0),
                            Ordering::Relaxed,
                        );
                        hill_mu[&2].metric[i][netw_src_idx].fetch_add(
                            diversity::hill_diversity(counts.clone(), 2.0).unwrap_or(0.0),
                            Ordering::Relaxed,
                        );
                    }
                    if compute_hill_weighted {
                        hill_wt_mu[&0].metric[i][netw_src_idx].fetch_add(
                            diversity::hill_diversity_branch_distance_wt(
                                counts.clone(),
                                nearest.clone(),
                                0.0,
                                b,
                                mcw,
                            )
                            .unwrap_or(0.0),
                            Ordering::Relaxed,
                        );
                        hill_wt_mu[&1].metric[i][netw_src_idx].fetch_add(
                            diversity::hill_diversity_branch_distance_wt(
                                counts.clone(),
                                nearest.clone(),
                                1.0,
                                b,
                                mcw,
                            )
                            .unwrap_or(0.0),
                            Ordering::Relaxed,
                        );
                        hill_wt_mu[&2].metric[i][netw_src_idx].fetch_add(
                            diversity::hill_diversity_branch_distance_wt(
                                counts.clone(),
                                nearest.clone(),
                                2.0,
                                b,
                                mcw,
                            )
                            .unwrap_or(0.0),
                            Ordering::Relaxed,
                        );
                    }
                    if compute_shannon {
                        shannon_mu.metric[i][netw_src_idx].fetch_add(
                            diversity::shannon_diversity(counts.clone()).unwrap_or(0.0),
                            Ordering::Relaxed,
                        );
                    }
                    if compute_gini {
                        gini_mu.metric[i][netw_src_idx].fetch_add(
                            diversity::gini_simpson_diversity(counts.clone()).unwrap_or(0.0),
                            Ordering::Relaxed,
                        );
                    }
                }
                Ok(())
            })?;
            let mut hill_result = None;
            if compute_hill {
                let hr = [0, 1, 2]
                    .iter()
                    .map(|&q_key| (q_key, hill_mu[&q_key].load()))
                    .collect();
                hill_result = Some(hr);
            }
            let mut hill_weighted_result = None;
            if compute_hill_weighted {
                let hr = [0, 1, 2]
                    .iter()
                    .map(|&q_key| (q_key, hill_wt_mu[&q_key].load()))
                    .collect();
                hill_weighted_result = Some(hr);
            }
            let shannon_result = if compute_shannon {
                Some(shannon_mu.load())
            } else {
                None
            };
            let gini_result = if compute_gini {
                Some(gini_mu.load())
            } else {
                None
            };
            Ok(MixedUsesResult {
                hill: hill_result,
                hill_weighted: hill_weighted_result,
                shannon: shannon_result,
                gini: gini_result,
            })
        })?;
        Ok(result)
    }

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
    ) -> PyResult<Vec<StatsResult>> {
        let (distances, betas, seconds) =
            pair_distances_betas_time(distances, betas, minutes, min_threshold_wt, speed_m_s)?;
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let max_walk_seconds = *seconds.iter().max().unwrap();
        let mut num_maps: Vec<HashMap<String, f32>> = Vec::with_capacity(numerical_maps.len());
        for numerical_map in numerical_maps.iter() {
            let numerical_map = numerical_map.bind(py).downcast::<PyDict>()?;
            if numerical_map.len() != self.count() {
                return Err(exceptions::PyValueError::new_err(
                    "The number of landuse encodings must match the number of data points",
                ));
            }
            let mut num_map: HashMap<String, f32> = HashMap::with_capacity(self.count());
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
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        let result = py.allow_threads(move || -> PyResult<_> {
            let mut sum = Vec::new();
            let mut sum_wt = Vec::new();
            let mut count = Vec::new();
            let mut count_wt = Vec::new();
            let mut max = Vec::new();
            let mut min = Vec::new();
            let mut sum_sq = Vec::new();
            let mut sum_sq_wt = Vec::new();
            let node_count = network_structure.node_count();
            for _ in 0..num_maps.len() {
                sum.push(MetricResult::new(distances.clone(), node_count, 0.0));
                sum_wt.push(MetricResult::new(distances.clone(), node_count, 0.0));
                count.push(MetricResult::new(distances.clone(), node_count, 0.0));
                count_wt.push(MetricResult::new(distances.clone(), node_count, 0.0));
                max.push(MetricResult::new(distances.clone(), node_count, f32::NAN));
                min.push(MetricResult::new(distances.clone(), node_count, f32::NAN));
                sum_sq.push(MetricResult::new(distances.clone(), node_count, 0.0));
                sum_sq_wt.push(MetricResult::new(distances.clone(), node_count, 0.0));
            }

            let node_indices = network_structure.node_indices();
            node_indices.par_iter().try_for_each(|&netw_src_idx| {
                if !pbar_disabled {
                    self.progress.fetch_add(1, Ordering::Relaxed);
                }
                if !network_structure.is_node_live(netw_src_idx).unwrap_or(true) {
                    return Ok::<(), PyErr>(());
                }
                let reachable_entries = self.aggregate_to_src_idx(
                    netw_src_idx,
                    network_structure,
                    max_walk_seconds,
                    speed_m_s,
                    jitter_scale,
                    angular,
                )?;
                for (data_key, data_dist) in &reachable_entries {
                    for (map_idx, num_map) in num_maps.iter().enumerate() {
                        if let Some(&num) = num_map.get(data_key) {
                            if num.is_nan() {
                                continue;
                            }
                            for (i, (&d, (&b, &mcw))) in distances
                                .iter()
                                .zip(betas.iter().zip(max_curve_wts.iter()))
                                .enumerate()
                            {
                                if *data_dist <= d as f32 {
                                    let wt = clipped_beta_wt(b, mcw, *data_dist).unwrap_or(0.0);
                                    let num_wt = num * wt;
                                    sum[map_idx].metric[i][netw_src_idx]
                                        .fetch_add(num, Ordering::Relaxed);
                                    sum_wt[map_idx].metric[i][netw_src_idx]
                                        .fetch_add(num_wt, Ordering::Relaxed);
                                    count[map_idx].metric[i][netw_src_idx]
                                        .fetch_add(1.0, Ordering::Relaxed);
                                    count_wt[map_idx].metric[i][netw_src_idx]
                                        .fetch_add(wt, Ordering::Relaxed);
                                    sum_sq[map_idx].metric[i][netw_src_idx]
                                        .fetch_add(num * num, Ordering::Relaxed);
                                    sum_sq_wt[map_idx].metric[i][netw_src_idx]
                                        .fetch_add(wt * num * num, Ordering::Relaxed);
                                    let current_max = max[map_idx].metric[i][netw_src_idx]
                                        .load(Ordering::Relaxed);
                                    if current_max.is_nan() || num > current_max {
                                        max[map_idx].metric[i][netw_src_idx]
                                            .store(num, Ordering::Relaxed);
                                    };
                                    let current_min = min[map_idx].metric[i][netw_src_idx]
                                        .load(Ordering::Relaxed);
                                    if current_min.is_nan() || num < current_min {
                                        min[map_idx].metric[i][netw_src_idx]
                                            .store(num, Ordering::Relaxed);
                                    };
                                }
                            }
                        }
                    }
                }
                Ok(())
            })?;
            let mut results = Vec::with_capacity(num_maps.len());
            for map_idx in 0..num_maps.len() {
                let mean_res = MetricResult::new(distances.clone(), node_count, f32::NAN);
                let mean_wt_res = MetricResult::new(distances.clone(), node_count, f32::NAN);
                let variance_res = MetricResult::new(distances.clone(), node_count, f32::NAN);
                let variance_wt_res = MetricResult::new(distances.clone(), node_count, f32::NAN);
                for node_idx in 0..node_count {
                    for (i, _) in distances.iter().enumerate() {
                        let sum_val = sum[map_idx].metric[i][node_idx].load(Ordering::Relaxed);
                        let count_val = count[map_idx].metric[i][node_idx].load(Ordering::Relaxed);
                        let sum_wt_val =
                            sum_wt[map_idx].metric[i][node_idx].load(Ordering::Relaxed);
                        let count_wt_val =
                            count_wt[map_idx].metric[i][node_idx].load(Ordering::Relaxed);
                        let sum_sq_val =
                            sum_sq[map_idx].metric[i][node_idx].load(Ordering::Relaxed);
                        let sum_sq_wt_val =
                            sum_sq_wt[map_idx].metric[i][node_idx].load(Ordering::Relaxed);
                        let mean_val = if count_val > 0.0 {
                            sum_val / count_val
                        } else {
                            f32::NAN
                        };
                        let mean_wt_val = if count_wt_val > 0.0 {
                            sum_wt_val / count_wt_val
                        } else {
                            f32::NAN
                        };
                        let variance_val = if count_val > 0.0 {
                            ((sum_sq_val / count_val) - mean_val.powi(2)).max(0.0)
                        } else {
                            f32::NAN
                        };
                        let variance_wt_val = if count_wt_val > 0.0 {
                            ((sum_sq_wt_val / count_wt_val) - mean_wt_val.powi(2)).max(0.0)
                        } else {
                            f32::NAN
                        };
                        mean_res.metric[i][node_idx].store(mean_val, Ordering::Relaxed);
                        mean_wt_res.metric[i][node_idx].store(mean_wt_val, Ordering::Relaxed);
                        variance_res.metric[i][node_idx].store(variance_val, Ordering::Relaxed);
                        variance_wt_res.metric[i][node_idx]
                            .store(variance_wt_val, Ordering::Relaxed);
                    }
                }
                results.push(StatsResult {
                    sum: sum[map_idx].load(),
                    sum_wt: sum_wt[map_idx].load(),
                    mean: mean_res.load(),
                    mean_wt: mean_wt_res.load(),
                    count: count[map_idx].load(),
                    count_wt: count_wt[map_idx].load(),
                    variance: variance_res.load(),
                    variance_wt: variance_wt_res.load(),
                    max: max[map_idx].load(),
                    min: min[map_idx].load(),
                });
            }
            Ok(results)
        })?;
        Ok(result)
    }
}

// Note: PyResult (and PyErr) are Send and Sync, so they can be used across threads.
// However, Python objects (e.g., Py<PyAny>, PyDict, etc.) must not be accessed from threads
// other than the one holding the GIL. In this code, all Python object access is done
// before entering py.allow_threads, and only Rust types are used inside the parallel region.
// Returning PyResult from the thread pool is safe as long as no Python objects are accessed
// or created inside the thread pool. This pattern is correct for pyo3 and rayon.
