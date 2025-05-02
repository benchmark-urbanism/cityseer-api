use crate::common::MetricResult;
use crate::common::{clip_wts_curve, clipped_beta_wt, pair_distances_betas_time};
use crate::common::{PROGRESS_UPDATE_INTERVAL, WALKING_SPEED};
use crate::diversity;
use crate::graph::NetworkStructure;
use core::f32;
use geo::algorithm::bounding_rect::BoundingRect;
use geo::algorithm::closest_point::ClosestPoint;
use geo::algorithm::intersects::Intersects;
use geo::algorithm::line_intersection::{line_intersection, LineIntersection};
use geo::algorithm::Euclidean;
use geo::geometry::Geometry;
use geo::{Centroid, Distance, Line, Point};
use numpy::PyArray1;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict};
use rayon::prelude::*;
use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{RTree, AABB};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use wkt::TryFromWkt;

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
    #[pyo3(signature = (data_key_py, geom_wkt, dedupe_key_py=None))]
    #[inline]
    fn new(
        py: Python,
        data_key_py: Py<PyAny>,
        geom_wkt: String,
        dedupe_key_py: Option<Py<PyAny>>,
    ) -> PyResult<DataEntry> {
        let data_key = py_key_to_composite(data_key_py.bind(py).clone())?;

        let dedupe_key_fallback: String = if let Some(ref key_py) = dedupe_key_py {
            py_key_to_composite(key_py.bind(py).clone())?
        } else {
            data_key.clone()
        };

        let dedupe_key_py_fallback: Py<PyAny> = if let Some(key_py) = dedupe_key_py {
            key_py
        } else {
            data_key_py.clone_ref(py)
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
            dedupe_key_py: dedupe_key_py_fallback,
            dedupe_key: dedupe_key_fallback,
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
    barrier_geoms: Option<Vec<Geometry<f64>>>,
    barrier_rtree: Option<RTree<GeomWithData<Rectangle<[f64; 2]>, usize>>>,
    data_rtree_items: Vec<GeomWithData<Rectangle<[f64; 2]>, String>>,
    data_rtree: Option<RTree<GeomWithData<Rectangle<[f64; 2]>, String>>>,
    #[pyo3(get)]
    node_data_map: HashMap<usize, Vec<(String, f64)>>,
}

#[pymethods]
impl DataMap {
    #[new]
    #[pyo3(signature = (barriers_wkt = None))]
    fn new(barriers_wkt: Option<Vec<String>>) -> PyResult<DataMap> {
        let mut barrier_geoms: Option<Vec<Geometry<f64>>> = None;
        let mut barriers_rtree: Option<RTree<GeomWithData<Rectangle<[f64; 2]>, usize>>> = None;

        if let Some(wkt_data) = barriers_wkt {
            let mut loaded_barriers_vec: Vec<Geometry<f64>> = Vec::new();
            let mut rtree_items: Vec<GeomWithData<Rectangle<[f64; 2]>, usize>> = Vec::new();
            let mut current_index = 0;
            for wkt in wkt_data.into_iter() {
                match Geometry::try_from_wkt_str(&wkt) {
                    Ok(wkt_geom) => {
                        if let Some(rect) = wkt_geom.bounding_rect() {
                            let envelope = Rectangle::from_corners(
                                [rect.min().x, rect.min().y],
                                [rect.max().x, rect.max().y],
                            );
                            loaded_barriers_vec.push(wkt_geom);
                            rtree_items.push(GeomWithData::new(envelope, current_index));
                            current_index += 1;
                        } else {
                            eprintln!(
                                "Warning: Skipping barrier geom with no bounding box: {}",
                                wkt
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to parse WKT barrier: {}. Error: {}",
                            wkt, e
                        );
                    }
                }
            }

            if !rtree_items.is_empty() {
                barriers_rtree = Some(RTree::bulk_load(rtree_items));
                barrier_geoms = Some(loaded_barriers_vec);
            } else {
                eprintln!("Warning: No valid barriers were loaded from the provided WKT data.");
            }
        }

        let map = DataMap {
            entries: HashMap::new(),
            progress: Arc::new(AtomicUsize::new(0)),
            barrier_geoms: barrier_geoms,
            barrier_rtree: barriers_rtree,
            data_rtree_items: Vec::new(),
            data_rtree: None,
            node_data_map: HashMap::new(),
        };
        Ok(map)
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

        // Get bounding box from the stored geom
        if let Some(rect) = entry.geom.bounding_rect() {
            let envelope =
                Rectangle::from_corners([rect.min().x, rect.min().y], [rect.max().x, rect.max().y]);
            // Store the data_key (String) with the envelope for R-tree build
            self.data_rtree_items
                .push(GeomWithData::new(envelope, data_key.clone()));

            // Invalidate the R-tree as new data is pending
            self.data_rtree = None;
        } else {
            // This might happen for Point geometries, which is okay.
            // We might only want to warn if it's unexpected (e.g., for LineString/Polygon).
            // For now, just proceed without adding to R-tree items if no bounding box.
            eprintln!(
                "Info: Geometry for key '{}' has no bounding box. Not added to R-tree.",
                data_key
            );
        }

        // Insert the DataEntry into the main map
        self.entries.insert(data_key, entry); // Use the cloned data_key
        self.data_rtree = None; // Invalidate the R-tree on data change

        Ok(())
    }

    fn entry_keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    fn get_entry(&self, data_key: &str) -> Option<DataEntry> {
        self.entries.get(data_key).map(|entry| entry.clone())
    }

    fn count(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Builds the R-tree for polygon geometries. Should be called after insertions.
    fn build_data_rtree(&mut self) -> PyResult<()> {
        if !self.data_rtree_items.is_empty() {
            let items = std::mem::take(&mut self.data_rtree_items);
            self.data_rtree = Some(RTree::bulk_load(items));
            eprintln!(
                "Data R-tree built with {} items.", // Renamed message slightly
                self.data_rtree.as_ref().map_or(0, |r| r.size())
            );
        } else {
            eprintln!("Warning: No geometries with bounding boxes found to build R-tree."); // Adjusted message
            self.data_rtree = None;
        }
        self.data_rtree_items.clear();
        Ok(())
    }

    /// Assigns data entries to network nodes based on proximity to edges. (Parallelized)
    /// Iterates through data entries in parallel, finds the nearest 6 edges for each,
    /// and assigns the data to the nodes of those edges if within distance
    /// and not blocked by barriers or other edges.
    #[pyo3(signature = (
        network_structure,
        max_assignment_dist
    ))]
    pub fn assign_data_to_network(
        &mut self,
        network_structure: &NetworkStructure,
        max_assignment_dist: f64,
        py: Python,
    ) -> PyResult<()> {
        // Ensure both edge R-tree and data R-tree are built
        if network_structure.edge_rtree.is_none() {
            return Err(exceptions::PyRuntimeError::new_err(
                "NetworkStructure's edge R-tree must be built before calling assign_data_to_network.",
            ));
        }
        if self.data_rtree.is_none() {
            self.build_data_rtree()?;
        }

        let edge_rtree = network_structure
            .edge_rtree
            .as_ref()
            .ok_or_else(|| exceptions::PyRuntimeError::new_err("Edge R-tree not built"))?;
        let data_rtree = self
            .data_rtree
            .as_ref()
            .ok_or_else(|| exceptions::PyRuntimeError::new_err("Data R-tree not built"))?;

        let assignments: Vec<(usize, String, f64)> = py.allow_threads(|| {
            data_rtree
                .iter() // Get the sequential iterator first
                .par_bridge() // Bridge it to a parallel iterator
                .flat_map(|geom_item| {
                    let data_key = &geom_item.data;
                    let data_entry = match self.entries.get(data_key) {
                        Some(entry) => entry,
                        None => {
                            return Vec::new();
                        }
                    };
                    let data_geom = &data_entry.geom;

                    let representative_point_geom = match data_geom.centroid() {
                        Some(centroid) => centroid,
                        None => {
                            return Vec::new();
                        }
                    };
                    let representative_point_arr =
                        [representative_point_geom.x(), representative_point_geom.y()];

                    // --- Workflow Storing Node Pairs per Edge ---
                    // 1. Get 6 nearest neighbors by bounding box
                    let candidate_edges_rtree = edge_rtree
                        .nearest_neighbor_iter(&representative_point_arr)
                        .take(6)
                        .collect::<Vec<_>>();

                    // 2. Validate candidates and store results per edge
                    let mut valid_edge_results: Vec<(
                        f64,                  // true_edge_dist
                        Option<(usize, f64)>, // Option<(start_node_idx, start_node_dist)>
                        Option<(usize, f64)>, // Option<(end_node_idx, end_node_dist)>
                    )> = Vec::new();

                    // Closure to check node validity and calculate distance
                    let check_node_validity = |node_idx: usize| -> Option<(usize, f64)> {
                        if let Ok(node_payload) = network_structure.get_node_payload(node_idx) {
                            let node_point = Point::new(node_payload.coord.x, node_payload.coord.y);
                            let closest_point_on_data = match data_geom {
                                Geometry::Point(p) => *p,
                                _ => match data_geom.closest_point(&node_point) {
                                    geo::Closest::Intersection(p) => p,
                                    geo::Closest::SinglePoint(p) => p,
                                    geo::Closest::Indeterminate => representative_point_geom,
                                },
                            };
                            let assignment_line = Line::new(closest_point_on_data.0, node_point.0);
                            if !self.intersects_barrier(&assignment_line)
                                && !self.intersects_edge(&assignment_line, network_structure)
                            {
                                let node_dist =
                                    Euclidean.distance(closest_point_on_data, node_point);
                                return Some((node_idx, node_dist));
                            }
                        }
                        None // Return None if payload fetch fails or intersection checks fail
                    };

                    for edge_geom_entry in &candidate_edges_rtree {
                        let start_node_idx = edge_geom_entry.data.0;
                        let end_node_idx = edge_geom_entry.data.1;
                        let edge_idx = edge_geom_entry.data.2;

                        let edge_payload = match network_structure.get_edge_payload(
                            start_node_idx,
                            end_node_idx,
                            edge_idx,
                        ) {
                            Ok(payload) => payload,
                            Err(_) => continue,
                        };

                        // Calculate true distance between data geom and edge geom
                        let true_edge_dist = Euclidean.distance(data_geom, &edge_payload.geom);

                        // Skip candidate if edge itself is too far
                        if true_edge_dist > max_assignment_dist {
                            continue;
                        }

                        // Check validity for start and end nodes using the closure
                        let valid_start_node = check_node_validity(start_node_idx);
                        let valid_end_node = check_node_validity(end_node_idx);

                        // Store result for this edge if at least one node was valid
                        if valid_start_node.is_some() || valid_end_node.is_some() {
                            valid_edge_results.push((
                                true_edge_dist,
                                valid_start_node,
                                valid_end_node,
                            ));
                        }
                    }

                    // 3. Select final edge results based on geometry type
                    let final_edge_results: Vec<&(
                        f64,
                        Option<(usize, f64)>,
                        Option<(usize, f64)>,
                    )> = match data_geom {
                        Geometry::Point(_) => {
                            // Find the minimum true_edge_dist among valid edges
                            if let Some(min_edge_dist) = valid_edge_results
                                .iter()
                                .map(|(edge_dist, _, _)| edge_dist)
                                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                            {
                                // Keep only edges matching the minimum edge distance
                                valid_edge_results
                                    .iter()
                                    .filter(|(edge_dist, _, _)| edge_dist == min_edge_dist)
                                    .collect()
                            } else {
                                Vec::new() // No valid edges found
                            }
                        }
                        _ => {
                            // Use all valid edge results for non-point geometries
                            valid_edge_results.iter().collect()
                        }
                    };

                    // 4. Generate final assignments by unpacking the selected edge results
                    let mut thread_local_assignments: Vec<(usize, String, f64)> = Vec::new();
                    for (_edge_dist, start_node_opt, end_node_opt) in final_edge_results {
                        if let Some((node_idx, node_dist)) = start_node_opt {
                            thread_local_assignments.push((
                                *node_idx,
                                data_key.clone(),
                                *node_dist,
                            ));
                        }
                        if let Some((node_idx, node_dist)) = end_node_opt {
                            thread_local_assignments.push((
                                *node_idx,
                                data_key.clone(),
                                *node_dist,
                            ));
                        }
                    }

                    thread_local_assignments // Return assignments for this data entry
                })
                .collect()
        });

        self.node_data_map.clear();
        for (node_idx, data_key, node_dist) in assignments {
            self.node_data_map
                .entry(node_idx)
                .or_default()
                .push((data_key, node_dist));
        }

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
        // Ensure data R-tree is built
        if self.data_rtree.is_none() && !self.data_rtree_items.is_empty() {
            return Err(exceptions::PyRuntimeError::new_err(
                "Data R-tree has not been built or is outdated. Call build_data_rtree() after inserting all data.",
            ));
        }

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
            )
        } else {
            network_structure.dijkstra_tree_simplest(
                netw_src_idx,
                max_walk_seconds,
                speed_m_s,
                Some(jitter_scale),
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
        Ok(entries_result)
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
    ) -> PyResult<HashMap<String, AccessibilityResult>> {
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
            node_indices
                .par_iter()
                .enumerate()
                .try_for_each(|(i, &netw_src_idx)| {
                    if !pbar_disabled && i % PROGRESS_UPDATE_INTERVAL == 0 {
                        self.progress
                            .fetch_add(PROGRESS_UPDATE_INTERVAL, AtomicOrdering::Relaxed);
                    }
                    if !network_structure.is_node_live(netw_src_idx)? {
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
                                        .fetch_add(1.0, AtomicOrdering::Relaxed);
                                    let val_wt = clipped_beta_wt(b, mcw, data_dist).unwrap_or(0.0);
                                    metrics_wt[lu_class].metric[i][netw_src_idx]
                                        .fetch_add(val_wt, AtomicOrdering::Relaxed);

                                    if d == max_dist {
                                        let current_dist = dists[lu_class].metric[0][netw_src_idx]
                                            .load(AtomicOrdering::Relaxed);
                                        if current_dist.is_nan() || data_dist < current_dist {
                                            dists[lu_class].metric[0][netw_src_idx]
                                                .store(data_dist, AtomicOrdering::Relaxed);
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
            node_indices
                .par_iter()
                .enumerate()
                .try_for_each(|(i, &netw_src_idx)| {
                    if !pbar_disabled && i % PROGRESS_UPDATE_INTERVAL == 0 {
                        self.progress
                            .fetch_add(PROGRESS_UPDATE_INTERVAL, AtomicOrdering::Relaxed);
                    }
                    if !network_structure.is_node_live(netw_src_idx)? {
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
                                AtomicOrdering::Relaxed,
                            );
                            hill_mu[&1].metric[i][netw_src_idx].fetch_add(
                                diversity::hill_diversity(counts.clone(), 1.0).unwrap_or(0.0),
                                AtomicOrdering::Relaxed,
                            );
                            hill_mu[&2].metric[i][netw_src_idx].fetch_add(
                                diversity::hill_diversity(counts.clone(), 2.0).unwrap_or(0.0),
                                AtomicOrdering::Relaxed,
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
                                AtomicOrdering::Relaxed,
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
                                AtomicOrdering::Relaxed,
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
                                AtomicOrdering::Relaxed,
                            );
                        }
                        if compute_shannon {
                            shannon_mu.metric[i][netw_src_idx].fetch_add(
                                diversity::shannon_diversity(counts.clone()).unwrap_or(0.0),
                                AtomicOrdering::Relaxed,
                            );
                        }
                        if compute_gini {
                            gini_mu.metric[i][netw_src_idx].fetch_add(
                                diversity::gini_simpson_diversity(counts.clone()).unwrap_or(0.0),
                                AtomicOrdering::Relaxed,
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
        let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
        let (distances, betas, seconds) =
            pair_distances_betas_time(speed_m_s, distances, betas, minutes, min_threshold_wt)?;
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
                // Initialize max/min with NaN to correctly handle the first value
                max.push(MetricResult::new(distances.clone(), node_count, f32::NAN));
                min.push(MetricResult::new(distances.clone(), node_count, f32::NAN));
                sum_sq.push(MetricResult::new(distances.clone(), node_count, 0.0));
                sum_sq_wt.push(MetricResult::new(distances.clone(), node_count, 0.0));
            }

            let node_indices = network_structure.node_indices();
            node_indices
                .par_iter()
                .enumerate()
                .try_for_each(|(i, &netw_src_idx)| {
                    if !pbar_disabled && i % PROGRESS_UPDATE_INTERVAL == 0 {
                        self.progress
                            .fetch_add(PROGRESS_UPDATE_INTERVAL, AtomicOrdering::Relaxed);
                    }
                    // Propagate error if is_node_live fails, otherwise skip if node is not live.
                    if !network_structure.is_node_live(netw_src_idx)? {
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
                                    continue; // Skip NaN values
                                }
                                for (i, (&d, (&b, &mcw))) in distances
                                    .iter()
                                    .zip(betas.iter().zip(max_curve_wts.iter()))
                                    .enumerate()
                                {
                                    if *data_dist <= d as f32 {
                                        let wt = clipped_beta_wt(b, mcw, *data_dist).unwrap_or(0.0);
                                        let num_wt = num * wt;
                                        // --- Accumulate sums and counts ---
                                        sum[map_idx].metric[i][netw_src_idx]
                                            .fetch_add(num, AtomicOrdering::Relaxed);
                                        sum_wt[map_idx].metric[i][netw_src_idx]
                                            .fetch_add(num_wt, AtomicOrdering::Relaxed);
                                        count[map_idx].metric[i][netw_src_idx]
                                            .fetch_add(1.0, AtomicOrdering::Relaxed);
                                        count_wt[map_idx].metric[i][netw_src_idx]
                                            .fetch_add(wt, AtomicOrdering::Relaxed);
                                        sum_sq[map_idx].metric[i][netw_src_idx]
                                            .fetch_add(num * num, AtomicOrdering::Relaxed);
                                        sum_sq_wt[map_idx].metric[i][netw_src_idx]
                                            .fetch_add(wt * num * num, AtomicOrdering::Relaxed);
                                        // --- Atomically update max and min ---
                                        // Assumes MetricResult uses atomic_float::AtomicF32 internally
                                        // which provides fetch_max/fetch_min that handle NaN correctly.
                                        max[map_idx].metric[i][netw_src_idx]
                                            .fetch_max(num, AtomicOrdering::Relaxed);
                                        min[map_idx].metric[i][netw_src_idx]
                                            .fetch_min(num, AtomicOrdering::Relaxed);
                                    }
                                }
                            }
                        }
                    }
                    Ok(())
                })?;
            // --- Post-processing (Mean, Variance) ---
            let mut results = Vec::with_capacity(num_maps.len());
            for map_idx in 0..num_maps.len() {
                let mean_res = MetricResult::new(distances.clone(), node_count, f32::NAN);
                let mean_wt_res = MetricResult::new(distances.clone(), node_count, f32::NAN);
                let variance_res = MetricResult::new(distances.clone(), node_count, f32::NAN);
                let variance_wt_res = MetricResult::new(distances.clone(), node_count, f32::NAN);
                for node_idx in 0..node_count {
                    for (i, _) in distances.iter().enumerate() {
                        let sum_val =
                            sum[map_idx].metric[i][node_idx].load(AtomicOrdering::Relaxed);
                        let count_val =
                            count[map_idx].metric[i][node_idx].load(AtomicOrdering::Relaxed);
                        let sum_wt_val =
                            sum_wt[map_idx].metric[i][node_idx].load(AtomicOrdering::Relaxed);
                        let count_wt_val =
                            count_wt[map_idx].metric[i][node_idx].load(AtomicOrdering::Relaxed);
                        let sum_sq_val =
                            sum_sq[map_idx].metric[i][node_idx].load(AtomicOrdering::Relaxed);
                        let sum_sq_wt_val =
                            sum_sq_wt[map_idx].metric[i][node_idx].load(AtomicOrdering::Relaxed);

                        // Calculate Mean
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

                        // Calculate Variance (using Welford's online algorithm principle implicitly)
                        // Variance = E[X^2] - (E[X])^2
                        let variance_val = if count_val > 0.0 {
                            (sum_sq_val / count_val - mean_val.powi(2)).max(0.0)
                        // Ensure non-negative due to potential float inaccuracies
                        } else {
                            f32::NAN
                        };
                        let variance_wt_val = if count_wt_val > 0.0 {
                            (sum_sq_wt_val / count_wt_val - mean_wt_val.powi(2)).max(0.0)
                        // Ensure non-negative
                        } else {
                            f32::NAN
                        };

                        // Store results (using relaxed ordering as this is post-parallel processing)
                        mean_res.metric[i][node_idx].store(mean_val, AtomicOrdering::Relaxed);
                        mean_wt_res.metric[i][node_idx].store(mean_wt_val, AtomicOrdering::Relaxed);
                        variance_res.metric[i][node_idx]
                            .store(variance_val, AtomicOrdering::Relaxed);
                        variance_wt_res.metric[i][node_idx]
                            .store(variance_wt_val, AtomicOrdering::Relaxed);
                    }
                }
                // --- Assemble final result struct ---
                results.push(StatsResult {
                    sum: sum[map_idx].load(),
                    sum_wt: sum_wt[map_idx].load(),
                    mean: mean_res.load(),
                    mean_wt: mean_wt_res.load(),
                    count: count[map_idx].load(),
                    count_wt: count_wt[map_idx].load(),
                    variance: variance_res.load(),
                    variance_wt: variance_wt_res.load(),
                    max: max[map_idx].load(), // Load final max/min after parallel updates
                    min: min[map_idx].load(),
                });
            }
            Ok(results)
        })?;
        Ok(result)
    }
}

impl DataMap {
    /// --- Helper function for barrier intersection check ---
    #[inline]
    fn intersects_barrier(&self, line: &Line<f64>) -> bool {
        if let (Some(barriers_rtree), Some(orig_barriers)) =
            (self.barrier_rtree.as_ref(), self.barrier_geoms.as_ref())
        {
            let line_aabb = AABB::from_corners(
                [line.start.x.min(line.end.x), line.start.y.min(line.end.y)],
                [line.start.x.max(line.end.x), line.start.y.max(line.end.y)],
            );
            let potential_blockers = barriers_rtree.locate_in_envelope_intersecting(&line_aabb);

            for barrier_item in potential_blockers {
                let original_geom_index = barrier_item.data;
                if let Some(barrier_geom) = orig_barriers.get(original_geom_index) {
                    if line.intersects(barrier_geom) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Check if a given line intersects any network edge geometry.
    #[inline]
    pub fn intersects_edge(&self, line: &Line<f64>, network_structure: &NetworkStructure) -> bool {
        if let Some(edge_rtree) = network_structure.edge_rtree.as_ref() {
            let line_aabb = AABB::from_corners(
                [line.start.x.min(line.end.x), line.start.y.min(line.end.y)],
                [line.start.x.max(line.end.x), line.start.y.max(line.end.y)],
            );
            let potential_edges = edge_rtree.locate_in_envelope_intersecting(&line_aabb);

            for edge_geom_entry in potential_edges {
                let start_node_idx = edge_geom_entry.data.0;
                let end_node_idx = edge_geom_entry.data.1;
                let edge_idx = edge_geom_entry.data.2;

                let edge_payload = match network_structure.get_edge_payload(
                    start_node_idx,
                    end_node_idx,
                    edge_idx,
                ) {
                    Ok(payload) => payload,
                    Err(_) => continue,
                };

                // Iterate through the segments of the LineString
                for edge_segment in edge_payload.geom.lines() {
                    if let Some(intersection) = line_intersection(*line, edge_segment) {
                        if let LineIntersection::SinglePoint {
                            is_proper: true, ..
                        } = intersection
                        {
                            // Found a proper crossing intersection with a segment
                            return true;
                        }
                        // Ignore collinear overlaps and endpoint touches (is_proper: false)
                    }
                }
            }
        }
        false // No proper intersection found
    }
}
