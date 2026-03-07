use crate::common::py_key_to_composite;

use geo::algorithm::centroid::Centroid;
use geo::algorithm::closest_point::ClosestPoint;
use geo::algorithm::intersects::Intersects;
use geo::algorithm::line_intersection::{line_intersection, LineIntersection};
use geo::geometry::{Coord, Line, LineString};
use geo::BoundingRect;
use geo::{Distance, Euclidean, Geometry, Length, Point};
use log;
use petgraph::prelude::*;
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::{EdgeIndexable, IntoEdgeReferences, NodeIndexable};
use pyo3::exceptions;
use pyo3::prelude::*;
use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{RTree, AABB};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use wkt::TryFromWkt;

/// Payload for a network node.
#[pyclass(skip_from_py_object)]
pub struct NodePayload {
    #[pyo3(get)]
    pub node_key: Py<PyAny>,
    pub point: Point<f64>,
    #[pyo3(get)]
    pub z: Option<f64>,
    #[pyo3(get)]
    pub live: bool,
    #[pyo3(get)]
    pub weight: f32,
    #[pyo3(get)]
    pub is_transport: bool,
}

impl Clone for NodePayload {
    fn clone(&self) -> Self {
        Python::attach(|py| NodePayload {
            node_key: self.node_key.clone_ref(py),
            point: self.point.clone(),
            z: self.z,                       // Option<f64> is Copy
            live: self.live,                 // bool is Copy
            weight: self.weight,             // f32 is Copy
            is_transport: self.is_transport, // bool is Copy
        })
    }
}

#[pymethods]
impl NodePayload {
    /// Validates the payload. Returns Ok(()) if valid, Err(PyValueError) otherwise.
    #[inline]
    pub fn validate(&self, py: Python) -> PyResult<()> {
        if self.is_transport {
            if self.live || self.weight != 0.0 {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid transport node payload: live must be false and weight must be 0.0. Node key: {:?}",
                    self.node_key.bind(py).repr().ok()
                )));
            }
        } else {
            if !self.weight.is_finite() || self.weight < 0.0 {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid street node payload: weight must be finite and non-negative (>= 0.0). Found {}. Node key: {:?}",
                    self.weight,
                    self.node_key.bind(py).repr().ok()
                )));
            }
        }
        Ok(())
    }

    #[getter]
    pub fn coord(&self) -> (f64, f64) {
        (self.point.x(), self.point.y())
    }

    #[getter]
    pub fn coord_z(&self) -> (f64, f64, Option<f64>) {
        (self.point.x(), self.point.y(), self.z)
    }
}

/// Payload for a network edge.
#[pyclass(from_py_object)]
pub struct EdgePayload {
    #[pyo3(get)]
    pub start_nd_key_py: Option<Py<PyAny>>, // Made optional
    #[pyo3(get)]
    pub end_nd_key_py: Option<Py<PyAny>>, // Made optional
    pub shared_primal_node_key: Option<String>,
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
    #[pyo3(get)]
    pub geom_wkt: Option<String>,
    pub geom: Option<LineString<f64>>,
    #[pyo3(get)]
    pub is_transport: bool,
}

impl Clone for EdgePayload {
    fn clone(&self) -> Self {
        Python::attach(|py| EdgePayload {
            start_nd_key_py: self.start_nd_key_py.as_ref().map(|k| k.clone_ref(py)),
            end_nd_key_py: self.end_nd_key_py.as_ref().map(|k| k.clone_ref(py)),
            shared_primal_node_key: self.shared_primal_node_key.clone(),
            edge_idx: self.edge_idx,       // usize is Copy
            length: self.length,           // f32 is Copy
            angle_sum: self.angle_sum,     // f32 is Copy
            imp_factor: self.imp_factor,   // f32 is Copy
            in_bearing: self.in_bearing,   // f32 is Copy
            out_bearing: self.out_bearing, // f32 is Copy
            seconds: self.seconds,         // f32 is Copy
            geom_wkt: self.geom_wkt.clone(),
            geom: self.geom.clone(),
            is_transport: self.is_transport, // bool is Copy
        })
    }
}

#[pymethods]
impl EdgePayload {
    /// Validates the payload. Returns Ok(()) if valid, Err(PyValueError) otherwise.
    #[inline]
    pub fn validate(&self, py: Python) -> PyResult<()> {
        // Common validation: imp_factor must be finite and positive
        if !self.imp_factor.is_finite() || self.imp_factor <= 0.0 {
            return Err(exceptions::PyValueError::new_err(format!(
                "Invalid edge payload : imp_factor must be finite and positive (> 0.0). Found {}. Start key: {:?}, End key: {:?}",
                self.imp_factor,
                self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
            )));
        }

        if self.is_transport {
            // Transport edge validation
            if (!self.seconds.is_finite() || self.seconds < 0.0)
                && (!self.length.is_finite() || self.length < 0.0)
            {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid transport edge payload : seconds must be finite and non-negative. Else length must be finite and non-negative. Start key: {:?}, End key: {:?}",
                    self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                    self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
                )));
            }
            if !self.angle_sum.is_nan() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid transport edge payload : angle_sum must be NaN. Found {}. Start key: {:?}, End key: {:?}",
                    self.angle_sum,
                    self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                    self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
                )));
            }
            if !self.in_bearing.is_nan() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid transport edge payload : in_bearing must be NaN. Found {}. Start key: {:?}, End key: {:?}",
                    self.in_bearing,
                    self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                    self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
                )));
            }
            if !self.out_bearing.is_nan() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid transport edge payload : out_bearing must be NaN. Found {}. Start key: {:?}, End key: {:?}",
                    self.out_bearing,
                    self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                    self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
                )));
            }
            if self.geom_wkt.is_some() || self.geom.is_some() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid transport edge payload : geom_wkt and geom must be None. Start key: {:?}, End key: {:?}",
                    self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                    self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
                )));
            }
        } else {
            // Street edge validation
            if !self.seconds.is_nan() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid street edge payload : seconds must be NaN. Found {}. Start key: {:?}, End key: {:?}",
                    self.seconds,
                    self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                    self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
                )));
            }
            if !self.length.is_finite() {
                // length >= 0 is implied by calculation
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid street edge payload : length must be finite. Found {}. Start key: {:?}, End key: {:?}",
                    self.length,
                    self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                    self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
                )));
            }
            if !self.angle_sum.is_finite() {
                // angle_sum >= 0 is implied by calculation
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid street edge payload : angle_sum must be finite. Found {}. Start key: {:?}, End key: {:?}",
                    self.angle_sum,
                    self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                    self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
                )));
            }
            if !self.in_bearing.is_finite() {
                // Bearings can be negative
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid street edge payload : in_bearing must be finite. Found {}. Start key: {:?}, End key: {:?}",
                    self.in_bearing,
                    self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                    self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
                )));
            }
            if !self.out_bearing.is_finite() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid street edge payload : out_bearing must be finite. Found {}. Start key: {:?}, End key: {:?}",
                    self.out_bearing,
                    self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                    self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
                )));
            }
            if self.geom_wkt.is_none() || self.geom.is_none() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid street edge payload : geom_wkt and geom must be Some. Start key: {:?}, End key: {:?}",
                    self.start_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok()),
                    self.end_nd_key_py.as_ref().map(|k| k.bind(py).repr().ok())
                )));
            }
        }
        Ok(())
    }
}

/// Visit state for a node during traversal.
#[pyclass(skip_from_py_object)]
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
    pub origin_seg: Option<usize>,
    #[pyo3(get)]
    pub last_seg: Option<usize>,
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
            origin_seg: None,
            last_seg: None,
            agg_seconds: f32::INFINITY,
        }
    }
}

/// Visit state for an edge during traversal.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Copy)]
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

// Define the type alias for clarity (optional)
// (start_node_idx, end_node_idx, start_node_point, end_node_point, edge_geom)
type EdgeRtreeItem =
    GeomWithData<Rectangle<[f64; 2]>, (usize, usize, Point<f64>, Point<f64>, LineString<f64>)>;

// Define type alias for Barrier R-tree item for clarity
type BarrierRtreeItem = GeomWithData<Rectangle<[f64; 2]>, usize>; // Data is index into barrier_geoms

/// Utility function to compute the bearing (in degrees) between two coordinates.
fn measure_bearing(a: Coord<f64>, b: Coord<f64>) -> f64 {
    // Ensure points are not identical to avoid atan2(0, 0) -> NaN
    if a == b {
        return 0.0; // Or handle as an error/NaN depending on requirements
    }
    (b.y - a.y).atan2(b.x - a.x).to_degrees()
}

/// Measures angle between three coordinate pairs (in degrees).
/// Equivalent to the Python `measure_coords_angle`.
fn measure_coords_angle(a: Coord<f64>, b: Coord<f64>, c: Coord<f64>) -> f64 {
    // Check for coincident points which would make angle undefined or zero
    if a == b || b == c {
        return 0.0; // Angle is zero if points are coincident
    }
    let a1 = measure_bearing(b, a);
    let a2 = measure_bearing(c, b);
    let angle_diff = a2 - a1;
    let normalized_angle = (angle_diff + 180.0).rem_euclid(360.0) - 180.0;
    normalized_angle.abs()
}

/// Measures angle between two segment bearings per indices.
/// Equivalent to the Python `_measure_linestring_angle`.
fn measure_linestring_angle(
    coords: &[Coord<f64>],
    idx_a: usize,
    idx_b: usize,
    idx_c: usize,
) -> f64 {
    // Bounds checks are implicitly handled by measure_cumulative_angle caller, but good practice
    if idx_a >= coords.len() || idx_b >= coords.len() || idx_c >= coords.len() {
        log::error!("Index out of bounds in measure_linestring_angle");
        return f64::NAN; // Indicate error
    }
    let coord_1 = coords[idx_a];
    let coord_2 = coords[idx_b];
    let coord_3 = coords[idx_c];
    measure_coords_angle(coord_1, coord_2, coord_3)
}

// Calculate cumulative angle along the LineString geometry
fn measure_cumulative_angle(coords: &[Coord<f64>]) -> f64 {
    if coords.len() < 3 {
        return 0.0; // No angles to measure if less than 3 points
    }
    let mut angle_sum = 0.0;
    // Iterate through the middle points (vertices) where angles can be formed
    for c_idx in 1..(coords.len() - 1) {
        // Angle at vertex c_idx is formed by points (c_idx-1, c_idx, c_idx+1)
        angle_sum += measure_linestring_angle(coords, c_idx - 1, c_idx, c_idx + 1);
    }
    angle_sum
}

/// Main network structure.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct NetworkStructure {
    pub graph: StableGraph<NodePayload, EdgePayload>,
    pub is_dual: bool,
    pub progress: Arc<AtomicUsize>,
    pub edge_rtree: Option<RTree<EdgeRtreeItem>>,
    pub barrier_geoms: Option<Vec<Geometry<f64>>>,
    pub barrier_rtree: Option<RTree<BarrierRtreeItem>>,
}

#[pymethods]
impl NetworkStructure {
    #[new]
    pub fn new() -> Self {
        Self {
            graph: StableGraph::<NodePayload, EdgePayload>::default(),
            is_dual: false,
            progress: Arc::new(AtomicUsize::new(0)),
            edge_rtree: None,
            barrier_geoms: None,
            barrier_rtree: None,
        }
    }

    #[inline]
    pub fn progress_init(&self) {
        self.progress.store(0, AtomicOrdering::Relaxed);
    }

    #[inline]
    pub fn progress(&self) -> usize {
        self.progress.load(AtomicOrdering::Relaxed)
    }

    #[getter]
    pub fn is_dual(&self) -> bool {
        self.is_dual
    }

    pub fn set_is_dual(&mut self, is_dual: bool) {
        self.is_dual = is_dual;
    }

    #[pyo3(signature = (node_key, x, y, live, weight, z=None))]
    pub fn add_street_node(
        &mut self,
        node_key: Py<PyAny>,
        x: f64,
        y: f64,
        live: bool,
        weight: f32,
        z: Option<f64>,
    ) -> usize {
        let payload = NodePayload {
            node_key,
            point: Point::new(x, y),
            z,
            live,
            weight,
            is_transport: false, // Explicitly false for street nodes
        };
        Python::attach(|py| {
            payload
                .validate(py)
                .expect("Invalid node payload for street node");
        });
        let new_node_idx = self.graph.add_node(payload);
        self.edge_rtree = None; // Invalidate edge R-tree
        new_node_idx.index()
    }

    #[pyo3(signature = (node_key, x, y, linking_radius=None, z=None))]
    pub fn add_transport_node(
        &mut self,
        node_key: Py<PyAny>,
        x: f64,
        y: f64,
        linking_radius: Option<f64>,
        z: Option<f64>,
        py: Python,
    ) -> PyResult<usize> {
        // Bind and clone node_key *before* moving it
        let bound_node_key = node_key.bind(py);
        let transport_key = py_key_to_composite(bound_node_key.clone())?; // Key for assignment search

        // Now move the original node_key into the payload
        let transport_point = Point::new(x, y);
        let new_node_idx = self.graph.add_node(NodePayload {
            node_key, // Original Py<PyAny> moved here
            point: transport_point,
            z,
            live: false, // Transport nodes are typically not "live" destinations themselves
            weight: 0.0, // Transport nodes usually don't have weights
            is_transport: true, // Explicitly true
        });
        let new_node_idx_usize = new_node_idx.index();

        // --- Linking Logic ---
        // Ensure edge R-tree is built before attempting linking
        if self.edge_rtree.is_none() {
            self.build_edge_rtree()?;
        }

        let link_radius = linking_radius.unwrap_or(100.0); // Default linking radius
        let transport_geom: Geometry<f64> = Geometry::Point(transport_point);

        // Find potential street node assignments using the existing logic
        // Note: find_assignments_for_entry uses edge R-tree, barriers, and street intersection checks.
        let potential_assignments = self.find_assignments_for_entry(
            &transport_key,  // Use temporary key for logging/context if needed inside
            &transport_geom, // Use transport node's geometry
            link_radius,     // Use linking radius as max distance
            6,               // No specific street node index
        );

        let mut links_added = 0;
        for (street_node_idx, _data_key, dist) in potential_assignments {
            log::debug!(
                "Linking transport node {} to street node {} with distance {}.",
                new_node_idx_usize,
                street_node_idx,
                dist
            );
            // Create a transport edge payload (bi-directional)
            let link_payload = EdgePayload {
                start_nd_key_py: None, // Linking edges don't have original keys
                end_nd_key_py: None,
                shared_primal_node_key: None,
                edge_idx: 0,
                length: dist as f32, // Store the geometric distance
                angle_sum: f32::NAN, // No geometry
                imp_factor: 1.0,     // Default impedance
                in_bearing: f32::NAN,
                out_bearing: f32::NAN,
                seconds: f32::NAN, // Calculate time based on distance
                geom_wkt: None,
                geom: None,
                is_transport: true, // Mark as a transport link edge
            };
            // Add edges in both directions
            match self._add_edge_internal(
                new_node_idx_usize,
                street_node_idx,
                link_payload.clone(),
                py,
            ) {
                Ok(_) => links_added += 1,
                Err(e) => log::error!(
                    "Failed to add transport link edge {} -> {}: {}",
                    new_node_idx_usize,
                    street_node_idx,
                    e
                ),
            }
            match self._add_edge_internal(street_node_idx, new_node_idx_usize, link_payload, py) {
                // No clone needed for second call
                Ok(_) => links_added += 1, // Count second direction? Or just pairs?
                Err(e) => log::error!(
                    "Failed to add transport link edge {} -> {}: {}",
                    street_node_idx,
                    new_node_idx_usize,
                    e
                ),
            }
        }
        log::debug!(
            "Added transport node {} and created {} link edges.",
            new_node_idx_usize,
            links_added
        );

        Ok(new_node_idx_usize)
    }

    // EXPENSIVE DUE TO PY CLONING OF KEYS - avoid calling in a loop
    pub fn get_node_payload_py(&self, node_idx: usize) -> PyResult<NodePayload> {
        Ok(self
            ._get_node_payload_checked(node_idx, "node_idx")?
            .clone()) // EXPENSIVE!!
    }

    // Unpack node weight directly from the payload to avoid cloning
    pub fn get_node_weight(&self, node_idx: usize) -> PyResult<f32> {
        Ok(self._get_node_payload_checked(node_idx, "node_idx")?.weight)
    }

    // Unpack live directly from the payload to avoid cloning
    pub fn is_node_live(&self, node_idx: usize) -> PyResult<bool> {
        Ok(self._get_node_payload_checked(node_idx, "node_idx")?.live)
    }

    /// Set the live status of a node (e.g. based on a boundary polygon).
    pub fn set_node_live(&mut self, node_idx: usize, live: bool) -> PyResult<()> {
        let ni = NodeIndex::new(node_idx);
        let payload = self.graph.node_weight_mut(ni).ok_or_else(|| {
            exceptions::PyValueError::new_err(format!(
                "Node index {} does not exist in the graph.",
                node_idx
            ))
        })?;
        payload.live = live;
        Ok(())
    }

    /// Returns the total count of all nodes (street and transport).
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns an upper bound on node indices (all valid indices are < node_bound).
    /// Use this instead of node_count() when allocating index-addressed vectors,
    /// because StableGraph may have gaps after node removal.
    pub fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    /// Returns an upper bound on edge indices (all valid indices are < edge_bound).
    pub fn edge_bound(&self) -> usize {
        self.graph.edge_bound()
    }

    /// Returns the count of non-transport (street) nodes.
    pub fn street_node_count(&self) -> usize {
        self.graph
            .node_weights()
            .filter(|p| !p.is_transport)
            .count()
    }

    /// Returns a list of indices for all nodes (street and transport).
    pub fn node_indices(&self) -> Vec<usize> {
        self.graph.node_indices().map(|node| node.index()).collect()
    }

    /// Returns a list of original keys for all nodes (street and transport).
    pub fn node_keys_py(&self, py: Python) -> Vec<Py<PyAny>> {
        self.graph
            .node_weights()
            .map(|payload| payload.node_key.clone_ref(py))
            .collect()
    }

    /// Returns a list of indices for non-transport (street) nodes.
    pub fn street_node_indices(&self) -> Vec<usize> {
        self.graph
            .node_indices()
            .filter(|&ni| {
                self.graph
                    .node_weight(ni)
                    .map_or(false, |p| !p.is_transport)
            })
            .map(|ni| ni.index())
            .collect()
    }

    /// Returns a list of `x` coordinates for all nodes (street and transport).
    #[getter]
    pub fn node_xs(&self) -> Vec<f64> {
        self.graph
            .node_weights()
            .map(|payload| payload.point.x())
            .collect()
    }

    /// Returns a list of `y` coordinates for all nodes (street and transport).
    #[getter]
    pub fn node_ys(&self) -> Vec<f64> {
        self.graph
            .node_weights()
            .map(|payload| payload.point.y())
            .collect()
    }

    /// Returns a list of `(x, y)` coordinates for all nodes (street and transport).
    #[getter]
    pub fn node_xys(&self) -> Vec<(f64, f64)> {
        self.graph
            .node_weights()
            .map(|payload| (payload.point.x(), payload.point.y()))
            .collect()
    }

    /// Returns a list of optional `z` coordinates for all nodes (street and transport).
    #[getter]
    pub fn node_zs(&self) -> Vec<Option<f64>> {
        self.graph.node_weights().map(|payload| payload.z).collect()
    }

    /// Returns a list of `(x, y, z)` coordinates for all nodes (street and transport).
    #[getter]
    pub fn node_xyzs(&self) -> Vec<(f64, f64, Option<f64>)> {
        self.graph
            .node_weights()
            .map(|payload| (payload.point.x(), payload.point.y(), payload.z))
            .collect()
    }

    #[getter]
    pub fn node_lives(&self) -> Vec<bool> {
        self.graph
            .node_weights()
            .map(|payload| payload.live)
            .collect()
    }

    /// Returns a list of `live` status indicators for non-transport (street) nodes.
    #[getter]
    pub fn street_node_lives(&self) -> Vec<bool> {
        self.graph
            .node_weights()
            .filter(|p| !p.is_transport)
            .map(|p| p.live)
            .collect()
    }

    #[getter]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Internal helper to add a pre-constructed EdgePayload.
    fn _add_edge_internal(
        &mut self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        payload: EdgePayload,
        py: Python, // Add Python GIL token
    ) -> PyResult<usize> {
        let start_node_index = self.require_node_exists(start_nd_idx, "start_nd_idx")?;
        let end_node_index = self.require_node_exists(end_nd_idx, "end_nd_idx")?;

        // Validate payload consistency - now returns PyResult<()>
        payload.validate(py)?; // Propagate error if invalid

        let new_edge_idx = self.graph.add_edge(
            start_node_index,
            end_node_index,
            payload, // Payload is moved here
        );
        Ok(new_edge_idx.index())
    }

    /// Adds a street edge with geometry. Calculates length, bearings, and angle sum from WKT.
    /// Sets seconds to NaN.
    #[pyo3(signature = (start_nd_idx, end_nd_idx, edge_idx, start_nd_key_py, end_nd_key_py, geom_wkt, imp_factor=None, shared_primal_node_key=None))]
    pub fn add_street_edge(
        &mut self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
        start_nd_key_py: Py<PyAny>,
        end_nd_key_py: Py<PyAny>,
        geom_wkt: String, // Required geometry
        imp_factor: Option<f32>,
        shared_primal_node_key: Option<String>,
        py: Python, // Add Python GIL token
    ) -> PyResult<usize> {
        // Truncate WKT for error messages if too long
        let wkt_preview: String = if geom_wkt.len() > 200 {
            format!(
                "{}... (truncated, {} chars total)",
                &geom_wkt[..200],
                geom_wkt.len()
            )
        } else {
            geom_wkt.clone()
        };
        let geom = match LineString::try_from_wkt_str(&geom_wkt) {
            Ok(geom) => geom,
            Err(e) => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Failed to parse WKT for street edge (idx {}) between nodes {} and {}.\n\
                     Parse error: {}\n\
                     WKT: {}",
                    edge_idx, start_nd_idx, end_nd_idx, e, wkt_preview
                )));
            }
        };

        let coords_vec: Vec<Coord> = geom.coords().cloned().collect();
        let coords: &[Coord] = &coords_vec;
        let num_coords = coords.len();

        if num_coords < 2 {
            return Err(exceptions::PyValueError::new_err(format!(
                "Street edge geometry (idx {}) between nodes {} and {} must have at least 2 coordinates. Found {}.\n\
                 WKT: {}",
                 edge_idx, start_nd_idx, end_nd_idx, num_coords, wkt_preview
            )));
        }

        let length = Euclidean.length(&geom) as f32;

        // Check for degenerate (zero-length) geometry
        if length < 1e-6 {
            let first_coord = coords[0];
            let last_coord = coords[num_coords - 1];
            log::warn!(
                "Degenerate edge geometry (idx {}) between nodes {} and {}: \
                 length={:.2e}m, num_coords={}, first=({:.4}, {:.4}), last=({:.4}, {:.4})",
                edge_idx,
                start_nd_idx,
                end_nd_idx,
                length,
                num_coords,
                first_coord.x,
                first_coord.y,
                last_coord.x,
                last_coord.y
            );
        }

        // Compute in_bearing: find first distinct coordinate pair from the start
        let in_bearing = {
            let mut bearing = f32::NAN;
            let first = coords[0];
            // Iterate forward to find first distinct coordinate
            for i in 1..num_coords {
                if coords[i] != first {
                    bearing = measure_bearing(first, coords[i]) as f32;
                    if i > 1 {
                        log::debug!(
                            "Edge (idx {}) first {} coords are identical, using coord[{}] for in_bearing",
                            edge_idx, i, i
                        );
                    }
                    break;
                }
            }
            if bearing.is_nan() {
                if length > 1e-6 {
                    // Non-degenerate geometry but all coords identical? Log details for debugging
                    log::warn!(
                        "Edge (idx {}) between nodes {} and {}: length={:.4}m but all {} coords are identical at ({:.4}, {:.4}). \
                         This may indicate a closed ring or corrupt geometry.",
                        edge_idx, start_nd_idx, end_nd_idx, length, num_coords, first.x, first.y
                    );
                }
                // Fallback to 0.0 for degenerate edges to pass finite validation
                bearing = 0.0;
            }
            bearing
        };
        // Compute out_bearing: find first distinct coordinate pair from the end
        let out_bearing = {
            let mut bearing = f32::NAN;
            let last = coords[num_coords - 1];
            // Iterate backward to find first distinct coordinate from end
            for i in (0..num_coords - 1).rev() {
                if coords[i] != last {
                    bearing = measure_bearing(coords[i], last) as f32;
                    if i < num_coords - 2 {
                        log::debug!(
                            "Edge (idx {}) last {} coords are identical, using coord[{}] for out_bearing",
                            edge_idx, num_coords - 1 - i, i
                        );
                    }
                    break;
                }
            }
            if bearing.is_nan() {
                if length > 1e-6 {
                    log::warn!(
                        "Edge (idx {}) between nodes {} and {}: length={:.4}m but all {} coords match last coord ({:.4}, {:.4}). \
                         This may indicate a closed ring or corrupt geometry.",
                        edge_idx, start_nd_idx, end_nd_idx, length, num_coords, last.x, last.y
                    );
                }
                // Fallback to 0.0 for degenerate edges to pass finite validation
                bearing = 0.0;
            }
            bearing
        };

        let angle_sum = measure_cumulative_angle(coords) as f32;

        let imp = imp_factor.unwrap_or(1.0);
        if !imp.is_finite() || imp <= 0.0 {
            return Err(exceptions::PyValueError::new_err(format!(
                "Invalid impedance factor ({}) for edge (idx {}) between nodes {} and {}.\n\
                 Impedance must be finite and positive (> 0.0).\n\
                 Edge length: {:.4}m, num_coords: {}",
                imp, edge_idx, start_nd_idx, end_nd_idx, length, num_coords
            )));
        }

        let payload = EdgePayload {
            start_nd_key_py: Some(start_nd_key_py),
            end_nd_key_py: Some(end_nd_key_py),
            shared_primal_node_key,
            edge_idx,
            length,
            angle_sum,
            imp_factor: imp,
            in_bearing,
            out_bearing,
            seconds: f32::NAN,
            geom_wkt: Some(geom_wkt),
            geom: Some(geom),
            is_transport: false,
        };

        self.edge_rtree = None;

        // Pass Python token to internal helper
        self._add_edge_internal(start_nd_idx, end_nd_idx, payload, py)
    }

    /// Remove a street node and all its connected edges from the StableGraph.
    ///
    /// StableGraph::remove_node() cascades to all edges connected to the node,
    /// and preserves existing indices for other nodes (no swap-and-compact).
    /// This means node indices held externally (e.g. by the QGIS plugin's
    /// `node_idx` dict) remain valid after removal.
    ///
    /// Returns an error if the node does not exist or is a transport node.
    pub fn remove_street_node(&mut self, node_idx: usize) -> PyResult<()> {
        let ni = NodeIndex::new(node_idx);
        let payload = self.graph.node_weight(ni).ok_or_else(|| {
            exceptions::PyValueError::new_err(format!(
                "Node index {} does not exist in the graph.",
                node_idx
            ))
        })?;
        if payload.is_transport {
            return Err(exceptions::PyValueError::new_err(format!(
                "Node index {} is a transport node and cannot be removed with remove_street_node.",
                node_idx
            )));
        }
        self.graph.remove_node(ni);
        self.edge_rtree = None;
        Ok(())
    }

    /// Remove a specific directed edge identified by its start/end node indices and edge_idx.
    /// Other edge indices remain stable after removal (StableGraph guarantee).
    pub fn remove_street_edge(
        &mut self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
    ) -> PyResult<()> {
        let start_ni = self.require_node_exists(start_nd_idx, "start_nd_idx")?;
        let end_ni = self.require_node_exists(end_nd_idx, "end_nd_idx")?;
        let edge_id = self
            .graph
            .edges_connecting(start_ni, end_ni)
            .find(|e| e.weight().edge_idx == edge_idx)
            .map(|e| e.id());
        match edge_id {
            Some(eid) => {
                self.graph.remove_edge(eid);
                self.edge_rtree = None;
                Ok(())
            }
            None => Err(exceptions::PyValueError::new_err(format!(
                "No edge with edge_idx {} found from node {} to node {}.",
                edge_idx, start_nd_idx, end_nd_idx
            ))),
        }
    }

    /// Adds an abstract transport edge defined by travel time (seconds).
    /// Length is set to NaN. Geometry-related fields are NaN/None.
    #[pyo3(signature = (start_nd_idx, end_nd_idx, edge_idx, start_nd_key_py, end_nd_key_py, seconds, imp_factor=None))]
    pub fn add_transport_edge(
        &mut self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
        start_nd_key_py: Py<PyAny>,
        end_nd_key_py: Py<PyAny>,
        seconds: f32, // Required seconds
        imp_factor: Option<f32>,
        py: Python, // Add Python GIL token
    ) -> PyResult<usize> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(exceptions::PyValueError::new_err(format!(
                "Invalid seconds value ({}) for transport edge (idx {}) between nodes {} and {}.",
                seconds, edge_idx, start_nd_idx, end_nd_idx
            )));
        }

        let payload = EdgePayload {
            start_nd_key_py: Some(start_nd_key_py),
            end_nd_key_py: Some(end_nd_key_py),
            shared_primal_node_key: None,
            edge_idx,
            length: f32::NAN,
            angle_sum: f32::NAN,
            imp_factor: imp_factor.unwrap_or(1.0),
            in_bearing: f32::NAN,
            out_bearing: f32::NAN,
            seconds,
            geom_wkt: None,
            geom: None,
            is_transport: true,
        };

        // Pass Python token to internal helper
        self._add_edge_internal(start_nd_idx, end_nd_idx, payload, py)
    }

    pub fn edge_references(&self) -> Vec<(usize, usize, usize)> {
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

    // EXPENSIVE DUE TO PY CLONING OF KEYS - avoid calling in a loop
    pub fn get_edge_payload_py(
        &self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
    ) -> PyResult<EdgePayload> {
        let payload = self
            ._get_edge_payload_checked(start_nd_idx, end_nd_idx, edge_idx)?
            .clone(); // EXPENSIVE!!
        Ok(payload)
    }

    pub fn get_edge_length(
        &self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
    ) -> PyResult<f32> {
        Ok(self
            ._get_edge_payload_checked(start_nd_idx, end_nd_idx, edge_idx)?
            .length)
    }

    pub fn get_edge_impedance(
        &self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
    ) -> PyResult<f32> {
        Ok(self
            ._get_edge_payload_checked(start_nd_idx, end_nd_idx, edge_idx)?
            .imp_factor)
    }

    pub fn validate(&self, py: Python) -> PyResult<()> {
        if self.node_count() == 0 {
            return Err(exceptions::PyValueError::new_err(
                "NetworkStructure contains no nodes.",
            ));
        }
        // Validate all node payloads
        for node_idx in self.graph.node_indices() {
            let node_payload = self
                .graph
                .node_weight(node_idx)
                .expect("Node payload should exist for valid index from node_indices");
            // Call the updated validate method which returns PyResult<()>
            node_payload.validate(py)?; // Propagate error if invalid
        }

        // Validate all edge payloads
        for edge_ref in self.graph.edge_references() {
            let edge_payload = edge_ref.weight();
            // Call the updated validate method which returns PyResult<()>
            edge_payload.validate(py)?; // Propagate error if invalid
        }
        Ok(())
    }

    /// Builds the R-tree for street edge geometries using their bounding boxes.
    /// Deduplicates edges based on sorted node pairs and geometric equality.
    /// Stores (start_node_idx, end_node_idx, start_node_point, end_node_point, edge_geom)
    /// in the R-tree data payload.
    pub fn build_edge_rtree(&mut self) -> PyResult<()> {
        let edge_count = self.graph.edge_count();
        if edge_count == 0 {
            log::warn!("Cannot build R-tree, graph has no edges.");
            self.edge_rtree = None;
            return Ok(());
        }

        let mut rtree_items: Vec<EdgeRtreeItem> = Vec::with_capacity(edge_count / 2);
        let mut seen_node_pair_geoms: HashMap<(usize, usize), Vec<LineString<f64>>> =
            HashMap::new();

        for edge_ref in self.graph.edge_references() {
            let edge_payload = edge_ref.weight();

            if edge_payload.is_transport {
                continue;
            }

            let start_node_idx = edge_ref.source().index();
            let end_node_idx = edge_ref.target().index();

            let start_node_payload = self
                .graph
                .node_weight(edge_ref.source())
                .expect("Start node payload should exist for valid edge");
            let end_node_payload = self
                .graph
                .node_weight(edge_ref.target())
                .expect("End node payload should exist for valid edge");

            let node_pair_key = if start_node_idx < end_node_idx {
                (start_node_idx, end_node_idx)
            } else {
                (end_node_idx, start_node_idx)
            };

            let current_geom = edge_payload
                .geom
                .as_ref()
                .expect("Edge geometry should exist for valid edge");
            let reversed_geom = LineString::new(current_geom.coords().cloned().rev().collect());
            let existing_geoms = seen_node_pair_geoms.entry(node_pair_key).or_default();
            let already_exists = existing_geoms
                .iter()
                .any(|g| g.eq(&current_geom) || g.eq(&reversed_geom));
            if already_exists {
                continue;
            }
            existing_geoms.push(current_geom.clone());

            let rect = current_geom
                .bounding_rect()
                .expect("Geometry should have a bounding rect");
            let min_coord = rect.min();
            let max_coord = rect.max();
            let envelope =
                Rectangle::from_corners([min_coord.x, min_coord.y], [max_coord.x, max_coord.y]);

            rtree_items.push(GeomWithData::new(
                envelope,
                (
                    start_node_idx,
                    end_node_idx,
                    start_node_payload.point,
                    end_node_payload.point,
                    current_geom.clone(),
                ),
            ));
        }

        if rtree_items.is_empty() {
            log::warn!("No valid, non-duplicate street edge geometries found to build R-tree.");
            self.edge_rtree = None;
        } else {
            self.edge_rtree = Some(RTree::bulk_load(rtree_items));
            let built_count = self
                .edge_rtree
                .as_ref()
                .expect("R-tree should exist after successful build")
                .size();
            log::info!("Edge R-tree built successfully with {} items.", built_count,);
        }

        Ok(())
    }

    /// Sets barrier geometries from WKT strings and builds the R-tree.
    /// Replaces any existing barriers.
    #[pyo3(signature = (barriers_wkt))]
    pub fn set_barriers(&mut self, barriers_wkt: Vec<String>) -> PyResult<()> {
        let mut loaded_barriers_vec: Vec<Geometry<f64>> = Vec::with_capacity(barriers_wkt.len());
        let mut rtree_items: Vec<BarrierRtreeItem> = Vec::with_capacity(barriers_wkt.len());
        let mut parse_errors = 0;
        let mut skipped_no_bbox = 0;

        for (index, wkt) in barriers_wkt.into_iter().enumerate() {
            match Geometry::try_from_wkt_str(&wkt) {
                Ok(wkt_geom) => {
                    if let Some(rect) = wkt_geom.bounding_rect() {
                        let envelope = Rectangle::from_corners(
                            [rect.min().x, rect.min().y],
                            [rect.max().x, rect.max().y],
                        );
                        let barrier_index = loaded_barriers_vec.len();
                        loaded_barriers_vec.push(wkt_geom);
                        rtree_items.push(GeomWithData::new(envelope, barrier_index));
                    } else {
                        log::warn!(
                            "Skipping barrier geom (input index {}) with no bounding box: {}",
                            index,
                            wkt
                        );
                        skipped_no_bbox += 1;
                    }
                }
                Err(e) => {
                    log::warn!(
                        "Failed to parse WKT barrier (input index {}): '{}'. Error: {}",
                        index,
                        wkt,
                        e
                    );
                    parse_errors += 1;
                }
            }
        }

        if parse_errors > 0 {
            log::warn!(
                "Encountered {} errors while parsing barrier WKT strings.",
                parse_errors
            );
        }
        if skipped_no_bbox > 0 {
            log::warn!(
                "Skipped {} barriers due to missing bounding boxes.",
                skipped_no_bbox
            );
        }

        if !rtree_items.is_empty() {
            self.barrier_rtree = Some(RTree::bulk_load(rtree_items));
            self.barrier_geoms = Some(loaded_barriers_vec);
            let built_count = self
                .barrier_rtree
                .as_ref()
                .expect("Barrier R-tree should exist after successful build")
                .size();
            log::debug!(
                "Barriers set and R-tree built successfully with {} items.",
                built_count
            );
        } else {
            log::warn!("No valid barriers were loaded. Clearing existing barriers.");
            self.barrier_rtree = None;
            self.barrier_geoms = None;
        }

        Ok(())
    }

    /// Removes all barrier geometries and the associated R-tree.
    pub fn unset_barriers(&mut self) {
        self.barrier_geoms = None;
        self.barrier_rtree = None;
        log::debug!("Barriers unset and R-tree cleared.");
    }
}

impl NetworkStructure {
    #[inline]
    fn _get_node_payload_checked(
        &self,
        node_idx: usize,
        param_name: &str,
    ) -> PyResult<&NodePayload> {
        let ni = self.require_node_exists(node_idx, param_name)?;
        Ok(self
            .graph
            .node_weight(ni)
            .expect("Node payload should exist after require_node_exists"))
    }

    #[inline]
    fn require_node_exists(&self, node_idx: usize, param_name: &str) -> PyResult<NodeIndex> {
        let ni = NodeIndex::new(node_idx);
        if self.graph.node_weight(ni).is_none() {
            return Err(exceptions::PyValueError::new_err(format!(
                "{} {} does not exist in the graph.",
                param_name, node_idx
            )));
        }
        Ok(ni)
    }

    fn _get_edge_payload_checked(
        &self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
    ) -> PyResult<&EdgePayload> {
        let start_node_index = self.require_node_exists(start_nd_idx, "start_nd_idx")?;
        let end_node_index = self.require_node_exists(end_nd_idx, "end_nd_idx")?;

        let edge_ref = self
            .graph
            .edges_connecting(start_node_index, end_node_index)
            .find(|edge_ref| edge_ref.weight().edge_idx == edge_idx);
        edge_ref
            .map(|e| e.weight())
            .ok_or_else(|| exceptions::PyValueError::new_err("Edge not found"))
    }

    fn _get_edge_payload(
        &self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
    ) -> &EdgePayload {
        let start_node_index = NodeIndex::new(start_nd_idx);
        let end_node_index = NodeIndex::new(end_nd_idx);

        let edge_ref = self
            .graph
            .edges_connecting(start_node_index, end_node_index)
            .find(|edge_ref| edge_ref.weight().edge_idx == edge_idx);
        edge_ref.expect("Edge not found").weight()
    }

    #[inline]
    pub(crate) fn get_edge_length_unchecked(
        &self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
    ) -> f32 {
        self._get_edge_payload(start_nd_idx, end_nd_idx, edge_idx)
            .length
    }

    #[inline]
    pub(crate) fn get_edge_impedance_unchecked(
        &self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
    ) -> f32 {
        self._get_edge_payload(start_nd_idx, end_nd_idx, edge_idx)
            .imp_factor
    }

    #[inline]
    pub(crate) fn get_node_weight_unchecked(&self, node_idx: usize) -> f32 {
        self.graph
            .node_weight(NodeIndex::new(node_idx))
            .expect("No payload for requested node index.")
            .weight
    }

    #[inline]
    pub(crate) fn is_node_live_unchecked(&self, node_idx: usize) -> bool {
        self.graph
            .node_weight(NodeIndex::new(node_idx))
            .expect("No payload for requested node index.")
            .live
    }

    #[inline]
    pub(crate) fn validate_dual_for_angular(&self, context: &str) -> PyResult<()> {
        if !self.is_dual {
            return Err(exceptions::PyValueError::new_err(format!(
                "{} requires a dual graph for angular analysis. Convert the graph with cityseer.tools.graphs.nx_to_dual(...) before ingesting it into NetworkStructure.",
                context
            )));
        }
        Ok(())
    }

    /// Finds valid network node assignments for a single data entry.
    /// Checks proximity, max distance, barrier intersections, and street intersections.
    /// Returns Vec<(assigned_node_idx, data_key_clone, assignment_distance)>
    pub fn find_assignments_for_entry(
        &self,
        data_key: &str,
        data_geom: &Geometry<f64>,
        max_assignment_dist: f64,
        n_nearest_candidates: usize,
    ) -> Vec<(usize, String, f64)> {
        let edge_rtree = self.edge_rtree.as_ref().expect("Edge R-tree should exist.");

        let is_point_geom = matches!(data_geom, Geometry::Point(_));
        let data_cent = match data_geom.centroid() {
            Some(c) => c,
            None => {
                log::warn!(
                    "Data entry '{}' has no centroid (empty geometry?), skipping assignment.",
                    data_key
                );
                return Vec::new();
            }
        };
        // Get candidates from the R-tree
        let candidate_edges_rtree = if is_point_geom {
            // if the data geometry is a point, use nearest neighbor search
            edge_rtree
                .nearest_neighbor_iter(&[data_cent.x(), data_cent.y()])
                .take(n_nearest_candidates)
                .collect::<Vec<_>>()
        } else {
            // otherwise, use envelope intersection
            let data_rect = match data_geom.bounding_rect() {
                Some(r) => r,
                None => {
                    log::warn!(
                        "Data entry '{}' has no bounding rect (empty geometry?), skipping assignment.",
                        data_key
                    );
                    return Vec::new();
                }
            };
            let query_aabb = AABB::from_corners(
                [
                    data_rect.min().x - max_assignment_dist,
                    data_rect.min().y - max_assignment_dist,
                ],
                [
                    data_rect.max().x + max_assignment_dist,
                    data_rect.max().y + max_assignment_dist,
                ],
            );
            edge_rtree
                .locate_in_envelope_intersecting(&query_aabb)
                .collect()
        };

        let mut candidates_with_dist: Vec<(f64, &EdgeRtreeItem)> = Vec::new();
        for edge_rtree_item in &candidate_edges_rtree {
            let edge_geom = &edge_rtree_item.data.4;
            let true_edge_dist = Euclidean.distance(data_geom, edge_geom);
            if true_edge_dist <= max_assignment_dist {
                candidates_with_dist.push((true_edge_dist, edge_rtree_item));
            }
        }
        candidates_with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let mut checked_nodes_for_this_entry: HashMap<usize, Option<(usize, f64)>> = HashMap::new();
        let mut nodes_added_for_this_entry: HashSet<usize> = HashSet::new();
        let mut local_assignments: Vec<(usize, String, f64)> = Vec::new();

        let check_node_validity_logic =
            |node_idx: usize, node_point: Point<f64>| -> Option<(usize, f64)> {
                // Use Euclidean distance from node to geometry directly.
                // This correctly returns 0 for nodes inside a polygon,
                // avoiding the closest_point boundary-snap issue where interior
                // nodes would get inflated distances to the polygon boundary.
                let node_dist = Euclidean.distance(&node_point, data_geom);
                if node_dist < 1e-6 {
                    // Node is inside or on the geometry boundary — distance is 0.
                    return Some((node_idx, 0.0));
                }
                // For exterior nodes, find the closest point on the geometry
                // to construct the assignment line for barrier/street checks.
                let closest_point_on_data = match data_geom.closest_point(&node_point) {
                    geo::Closest::SinglePoint(p) => p,
                    geo::Closest::Intersection(p) => p,
                    geo::Closest::Indeterminate => {
                        log::warn!(
                            "Indeterminate closest point for node {} to data '{}', using centroid.",
                            node_idx,
                            data_key
                        );
                        data_cent
                    }
                };
                let assignment_line = Line::new(closest_point_on_data.0, node_point.0);
                if self.line_intersects_barriers(&assignment_line)
                    || self.line_intersects_streets(&assignment_line)
                {
                    return None;
                }
                Some((node_idx, node_dist))
            };

        let mut assignments_found = 0;
        for (_true_edge_dist, edge_rtree_item) in candidates_with_dist {
            // Destructure the tuple from the R-tree item.
            // Primitive types (usize, Point<f64>) are Copy and will be copied.
            // Use `ref` for LineString as it's not Copy and we only have a shared reference.
            let (start_node_idx, end_node_idx, start_node_point, end_node_point, ref _edge_geom) =
                edge_rtree_item.data;

            // Check validity for the start node, caching the result.
            // Dereference the result from the or_insert_with closure as it returns &Option<(usize, f64)>
            let valid_start_node = *checked_nodes_for_this_entry
                .entry(start_node_idx)
                .or_insert_with(|| check_node_validity_logic(start_node_idx, start_node_point));

            // Check validity for the end node, caching the result.
            // Dereference the result from the or_insert_with closure.
            let valid_end_node = *checked_nodes_for_this_entry
                .entry(end_node_idx)
                .or_insert_with(|| check_node_validity_logic(end_node_idx, end_node_point));

            let mut edge_produced_assignment = false;

            // If the start node is valid and hasn't been added yet, add it to assignments.
            if let Some((node_idx, node_dist)) = valid_start_node {
                if nodes_added_for_this_entry.insert(node_idx) {
                    local_assignments.push((node_idx, data_key.to_string(), node_dist));
                    edge_produced_assignment = true;
                }
            }
            // If the end node is valid and hasn't been added yet, add it to assignments.
            if let Some((node_idx, node_dist)) = valid_end_node {
                if nodes_added_for_this_entry.insert(node_idx) {
                    local_assignments.push((node_idx, data_key.to_string(), node_dist));
                    edge_produced_assignment = true;
                }
            }

            // Optimization: If the data geometry is a point and we found an assignment from this edge,
            // we can stop processing further edges for this point.
            if is_point_geom && edge_produced_assignment {
                break;
            }

            assignments_found += 1;
            if assignments_found >= n_nearest_candidates {
                break;
            }
        }

        local_assignments
    }

    /// Checks if a line segment intersects with any street edge geometry in the R-tree,
    /// excluding intersections only at the endpoints of the line or the edge segments.
    #[inline]
    fn line_intersects_streets(&self, line: &Line<f64>) -> bool {
        if let Some(edge_rtree) = self.edge_rtree.as_ref() {
            let line_rect = line.bounding_rect();
            let line_aabb = AABB::from_corners(
                [line_rect.min().x, line_rect.min().y],
                [line_rect.max().x, line_rect.max().y],
            );
            for edge_item in edge_rtree.locate_in_envelope_intersecting(&line_aabb) {
                let edge_geom = &edge_item.data.4;
                for edge_segment in edge_geom.lines() {
                    if let Some(intersection) = line_intersection(*line, edge_segment) {
                        if let LineIntersection::SinglePoint {
                            is_proper: true, ..
                        } = intersection
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Checks if a given line segment intersects with any internal barrier geometry.
    #[inline]
    fn line_intersects_barriers(&self, line: &Line<f64>) -> bool {
        if let (Some(rtree), Some(geoms)) = (&self.barrier_rtree, &self.barrier_geoms) {
            let line_rect = line.bounding_rect();
            let line_aabb = AABB::from_corners(
                [line_rect.min().x, line_rect.min().y],
                [line_rect.max().x, line_rect.max().y],
            );
            for barrier_item in rtree.locate_in_envelope_intersecting(&line_aabb) {
                let barrier_geom = geoms
                    .get(barrier_item.data)
                    .expect("Barrier geometry should exist for valid index.");
                if line.intersects(barrier_geom) {
                    return true;
                }
            }
        }
        false
    }
}
