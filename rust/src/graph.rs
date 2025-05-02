use geo::algorithm::Euclidean;
use geo::geometry::{Coord, LineString};
use geo::BoundingRect;
use geo::Length;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::prelude::*;
use pyo3::exceptions;
use pyo3::prelude::*;
use rstar::primitives::{GeomWithData, Rectangle};
use rstar::RTree;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use wkt::TryFromWkt;

/// Payload for a network node.
#[pyclass]
#[derive(Clone)]
pub struct NodePayload {
    #[pyo3(get)]
    pub node_key: String,
    pub coord: Coord<f64>,
    #[pyo3(get)]
    pub live: bool,
    #[pyo3(get)]
    pub weight: f32,
}

#[pymethods]
impl NodePayload {
    #[inline]
    pub fn validate(&self) -> bool {
        true
    }

    #[getter]
    pub fn coord(&self) -> (f64, f64) {
        (self.coord.x, self.coord.y)
    }
}

/// Payload for a network edge.
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
    #[pyo3(get)]
    pub seconds: f32,
    #[pyo3(get)]
    pub geom_wkt: String,
    pub geom: LineString<f64>,
}

#[pymethods]
impl EdgePayload {
    #[inline]
    pub fn validate(&self) -> bool {
        // If seconds is NaN, all other values must be finite
        if self.seconds.is_nan() {
            self.length.is_finite()
                && self.angle_sum.is_finite()
                && self.imp_factor.is_finite()
                && self.in_bearing.is_finite()
                && self.out_bearing.is_finite()
        } else {
            // If seconds is finite, other values are optional
            self.seconds.is_finite()
                && self.length.is_finite()
                && self.angle_sum.is_finite()
                && self.imp_factor.is_finite()
                && (self.in_bearing.is_finite() || self.in_bearing.is_nan())
                && (self.out_bearing.is_finite() || self.out_bearing.is_nan())
        }
    }
}

/// Visit state for a node during traversal.
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
            cycles: 0.0,
            origin_seg: None,
            last_seg: None,
            out_bearing: f32::NAN,
            agg_seconds: f32::INFINITY,
        }
    }
}

/// Visit state for an edge during traversal.
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

// Define the type alias for clarity (optional)
// (start_node_idx, end_node_idx, start_node_coord, end_node_coord, edge_geom)
type EdgeRtreeItem =
    GeomWithData<Rectangle<[f64; 2]>, (usize, usize, Coord<f64>, Coord<f64>, LineString<f64>)>;

/// Utility function to compute the bearing (in degrees) between two coordinates.
fn measure_bearing(a: Coord<f64>, b: Coord<f64>) -> f64 {
    // Python reverses (y, x) order, so we do the same here.
    let (y1, x1) = (a.y, a.x);
    let (y2, x2) = (b.y, b.x);
    (y2 - y1).atan2(x2 - x1).to_degrees()
}

/// Measures angle between three coordinate pairs (in degrees).
/// Equivalent to the Python `measure_coords_angle`.
fn measure_coords_angle(a: Coord<f64>, b: Coord<f64>, c: Coord<f64>) -> f64 {
    let a1 = measure_bearing(b, a);
    let a2 = measure_bearing(c, b);
    ((a2 - a1 + 180.0).rem_euclid(360.0) - 180.0).abs()
}

/// Measures angle between two segment bearings per indices.
/// Equivalent to the Python `_measure_linestring_angle`.
fn measure_linestring_angle(
    coords: &[Coord<f64>],
    idx_a: usize,
    idx_b: usize,
    idx_c: usize,
) -> f64 {
    let coord_1 = coords[idx_a];
    let coord_2 = coords[idx_b];
    let coord_3 = coords[idx_c];
    measure_coords_angle(coord_1, coord_2, coord_3)
}

// Calculate cumulative angle along the LineString geometry
fn measure_cumulative_angle(coords: &[Coord<f64>]) -> f64 {
    let mut angle_sum = 0.0;
    for c_idx in 0..(coords.len().saturating_sub(2)) {
        angle_sum += measure_linestring_angle(coords, c_idx, c_idx + 1, c_idx + 2);
    }
    angle_sum
}

/// Main network structure.
#[pyclass]
#[derive(Clone)]
pub struct NetworkStructure {
    pub graph: DiGraph<NodePayload, EdgePayload>,
    pub progress: Arc<AtomicUsize>,
    pub edge_rtree: Option<RTree<EdgeRtreeItem>>,
}

#[pymethods]
impl NetworkStructure {
    #[new]
    pub fn new() -> Self {
        Self {
            graph: DiGraph::<NodePayload, EdgePayload>::default(),
            progress: Arc::new(AtomicUsize::new(0)),
            edge_rtree: None,
        }
    }

    #[inline]
    pub fn progress_init(&self) {
        self.progress.store(0, Ordering::Relaxed);
    }

    #[inline]
    pub fn progress(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }

    pub fn add_node(&mut self, node_key: String, x: f64, y: f64, live: bool, weight: f32) -> usize {
        let new_node_idx = self.graph.add_node(NodePayload {
            node_key,
            coord: Coord { x, y },
            live,
            weight,
        });
        self.edge_rtree = None;
        new_node_idx.index()
    }

    pub fn get_node_payload(&self, node_idx: usize) -> PyResult<NodePayload> {
        self.graph
            .node_weight(NodeIndex::new(node_idx))
            .cloned()
            .ok_or_else(|| {
                exceptions::PyValueError::new_err(format!(
                    "No payload for requested node index {}.",
                    node_idx
                ))
            })
    }

    pub fn get_node_weight(&self, node_idx: usize) -> PyResult<f32> {
        self.get_node_payload(node_idx)
            .map(|payload| payload.weight)
    }

    pub fn is_node_live(&self, node_idx: usize) -> PyResult<bool> {
        self.get_node_payload(node_idx).map(|payload| payload.live)
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn node_indices(&self) -> Vec<usize> {
        self.graph.node_indices().map(|node| node.index()).collect()
    }

    #[getter]
    pub fn node_xs(&self) -> Vec<f64> {
        self.graph
            .node_indices()
            .map(|node| self.graph[node].coord.x)
            .collect()
    }

    #[getter]
    pub fn node_ys(&self) -> Vec<f64> {
        self.graph
            .node_indices()
            .map(|node| self.graph[node].coord.y)
            .collect()
    }

    #[getter]
    pub fn node_xys(&self) -> Vec<(f64, f64)> {
        self.graph
            .node_indices()
            .map(|node| (self.graph[node].coord.x, self.graph[node].coord.y))
            .collect()
    }

    #[getter]
    pub fn node_lives(&self) -> Vec<bool> {
        self.graph
            .node_indices()
            .map(|node| self.graph[node].live)
            .collect()
    }

    #[getter]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn add_edge(
        &mut self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
        start_nd_key: String,
        end_nd_key: String,
        geom_wkt: String,
        imp_factor: Option<f32>,
        seconds: Option<f32>,
    ) -> PyResult<usize> {
        let node_idx_a = NodeIndex::new(start_nd_idx);
        let node_idx_b = NodeIndex::new(end_nd_idx);

        let geom = match LineString::try_from_wkt_str(&geom_wkt) {
            Ok(geom) => geom,
            Err(e) => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Failed to parse WKT for edge between nodes '{}'-'{}' (edge_idx {}): {}",
                    start_nd_key, end_nd_key, edge_idx, e
                )));
            }
        };

        // Calculate bearings from geometry
        let coords_vec: Vec<Coord> = geom.coords().cloned().collect();
        let coords: &[Coord] = &coords_vec;
        let num_coords = coords.len();

        let (in_bearing, out_bearing) = if num_coords >= 2 {
            let coord_1 = coords[0];
            let coord_2 = coords[1];
            let calculated_in_bearing = measure_bearing(coord_1, coord_2);

            let coord_n_minus_1 = coords[num_coords - 2];
            let coord_n = coords[num_coords - 1];
            let calculated_out_bearing = measure_bearing(coord_n_minus_1, coord_n);

            (calculated_in_bearing, calculated_out_bearing)
        } else {
            (f64::NAN, f64::NAN)
        };

        let angle_sum = if num_coords >= 3 {
            measure_cumulative_angle(coords)
        } else {
            0.0
        } as f32;

        let new_edge_idx = self.graph.add_edge(
            node_idx_a,
            node_idx_b,
            EdgePayload {
                start_nd_key,
                end_nd_key,
                edge_idx,
                length: Euclidean.length(&geom) as f32, // Calculate length using geo crate
                angle_sum,                              // Use calculated value
                imp_factor: imp_factor.unwrap_or(1.0),  // Keep existing default logic
                in_bearing: in_bearing as f32,          // Use calculated value
                out_bearing: out_bearing as f32,        // Use calculated value
                seconds: seconds.unwrap_or(f32::NAN),   // Keep existing default logic
                geom_wkt,                               // Store original WKT
                geom,                                   // Store parsed geometry
            },
        );
        self.edge_rtree = None; // Invalidate R-tree
        Ok(new_edge_idx.index())
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

    pub fn get_edge_payload(
        &self,
        start_nd_idx: usize,
        end_nd_idx: usize,
        edge_idx: usize,
    ) -> PyResult<EdgePayload> {
        let start_node_index = NodeIndex::new(start_nd_idx);
        let end_node_index = NodeIndex::new(end_nd_idx);
        self.graph
            .edges_connecting(start_node_index, end_node_index)
            .find(|edge_ref| edge_ref.weight().edge_idx == edge_idx)
            .map(|edge_ref| edge_ref.weight().clone())
            .ok_or_else(|| {
                exceptions::PyValueError::new_err(format!(
                    "Edge not found for nodes {}, {}, and idx {}.",
                    start_nd_idx, end_nd_idx, edge_idx
                ))
            })
    }

    pub fn validate(&self) -> PyResult<bool> {
        if self.node_count() == 0 {
            return Err(exceptions::PyValueError::new_err(
                "NetworkStructure contains no nodes.",
            ));
        }
        if self.edge_count() == 0 {
            return Err(exceptions::PyValueError::new_err(
                "NetworkStructure contains no edges.",
            ));
        }
        for node_idx in self.graph.node_indices() {
            let node_payload = self.get_node_payload(node_idx.index())?;
            if !node_payload.validate() {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid node payload for node idx {:?}.",
                    node_idx
                )));
            }
        }
        for edge_ref in self.graph.edge_references() {
            let edge_payload = edge_ref.weight();
            if !edge_payload.validate() {
                let start_node_idx = edge_ref.source().index();
                let end_node_idx = edge_ref.target().index();
                let edge_data_idx = edge_payload.edge_idx;
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid edge payload for edge between nodes {} and {} (edge_idx {}).",
                    start_node_idx, end_node_idx, edge_data_idx
                )));
            }
        }
        Ok(true)
    }

    /// Builds the R-tree for edge geometries using their bounding boxes.
    ///
    /// Stores (start_node_idx, end_node_idx, start_node_coord, end_node_coord, edge_geom)
    /// in the R-tree data payload.
    pub fn build_edge_rtree(&mut self) -> PyResult<()> {
        let edge_count = self.graph.edge_count();
        if edge_count == 0 {
            eprintln!("Warning: Cannot build R-tree, graph has no edges.");
            self.edge_rtree = None; // Ensure it's None if graph is empty
            return Ok(());
        }

        let mut rtree_items: Vec<EdgeRtreeItem> = Vec::with_capacity(edge_count);
        let mut skipped_edges = 0;

        for edge_ref in self.graph.edge_references() {
            let edge_payload = edge_ref.weight();
            let start_node_idx = edge_ref.source().index();
            let end_node_idx = edge_ref.target().index();
            let edge_data_idx = edge_payload.edge_idx; // Use the unique edge index for logging

            // Fetch node payloads ONCE here to get coordinates
            let start_node_payload = match self.get_node_payload(start_node_idx) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!(
                        "Error fetching start node payload for edge {}: {}. Skipping edge.",
                        edge_data_idx, e
                    );
                    skipped_edges += 1;
                    continue;
                }
            };
            let end_node_payload = match self.get_node_payload(end_node_idx) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!(
                        "Error fetching end node payload for edge {}: {}. Skipping edge.",
                        edge_data_idx, e
                    );
                    skipped_edges += 1;
                    continue;
                }
            };

            if let Some(rect) = edge_payload.geom.bounding_rect() {
                let min_coord = rect.min();
                let max_coord = rect.max();

                let rect_geom =
                    Rectangle::from_corners([min_coord.x, min_coord.y], [max_coord.x, max_coord.y]);
                rtree_items.push(GeomWithData::new(
                    rect_geom,
                    // Store tuple including coordinates
                    (
                        start_node_idx,
                        end_node_idx,
                        start_node_payload.coord,  // Store start coord
                        end_node_payload.coord,    // Store end coord
                        edge_payload.geom.clone(), // Store edge geom
                    ),
                ));
            } else {
                eprintln!(
                    "Warning: Skipping edge with no bounding box between nodes {} and {} (edge_idx {}). Geometry might be empty.",
                    start_node_idx, end_node_idx, edge_data_idx
                );
                skipped_edges += 1;
            }
        }

        if rtree_items.is_empty() {
            eprintln!(
                "Warning: No valid, non-degenerate edge geometries found to build R-tree. {} edges were skipped.",
                skipped_edges
            );
            self.edge_rtree = None;
        } else {
            self.edge_rtree = Some(RTree::bulk_load(rtree_items));
            let built_count = self.edge_rtree.as_ref().map_or(0, |r| r.size());
            eprintln!(
                "Edge R-tree built successfully with {} items. {} edges were skipped.",
                built_count, skipped_edges
            );
        }

        Ok(())
    }
}
