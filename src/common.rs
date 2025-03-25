use atomic_float::AtomicF32;
use numpy::borrow::PyReadonlyArray2;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::atomic::Ordering;

/// Minimum threshold weight for distance and beta calculations.
static MIN_THRESH_WT: f32 = 0.01831563888873418;
/// Walking speed in meters per second.
pub static WALKING_SPEED: f32 = 1.33333;

/// Represents a 2D coordinate with `x` and `y` values.
#[pyclass]
#[derive(Clone, Copy)]
pub struct Coord {
    #[pyo3(get)]
    pub x: f32,
    #[pyo3(get)]
    pub y: f32,
}

#[pymethods]
impl Coord {
    /// Creates a new `Coord` instance.
    #[new]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Returns the coordinates as a tuple `(x, y)`.
    pub fn xy(&self) -> (f32, f32) {
        (self.x, self.y)
    }

    /// Validates that the coordinates are finite.
    pub fn validate(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Calculates the Euclidean distance between this coordinate and another.
    pub fn hypot(&self, other_coord: Coord) -> f32 {
        ((self.x - other_coord.x).powi(2) + (self.y - other_coord.y).powi(2)).sqrt()
    }

    /// Computes the difference between this coordinate and another as a vector.
    pub fn difference(&self, other_coord: Coord) -> Coord {
        Coord::new(self.x - other_coord.x, self.y - other_coord.y)
    }
}

/// Holds metric results, including distances and a 2D matrix of atomic floats.
pub struct MetricResult {
    pub distances: Vec<u32>,
    pub metric: Vec<Vec<AtomicF32>>,
}

impl MetricResult {
    /// Initializes a new `MetricResult` with given distances, size, and initial value.
    pub fn new(distances: Vec<u32>, size: usize, init_val: f32) -> Self {
        let metric = distances
            .iter()
            .map(|_| {
                std::iter::repeat_with(|| AtomicF32::new(init_val))
                    .take(size)
                    .collect::<Vec<AtomicF32>>()
            })
            .collect();
        Self { distances, metric }
    }

    /// Converts the atomic floats into a Python-compatible format (`PyArray1<f32>`).
    pub fn load(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.distances
            .iter()
            .enumerate()
            .map(|(i, &dist)| {
                let vec_f32: Vec<f32> = self.metric[i]
                    .iter()
                    .map(|a| a.load(Ordering::SeqCst))
                    .collect();
                let array = Python::with_gil(|py| {
                    vec_f32
                        .into_pyarray(py)
                        .to_owned() // This gives us a PyArray, but wrapped in pyo3::Bound
                        .into() // Convert to the required Py<PyArray> type
                });
                (dist, array)
            })
            .collect()
    }
}
// Manually implement Clone for MetricResult
impl Clone for MetricResult {
    fn clone(&self) -> Self {
        MetricResult {
            distances: self.distances.clone(), // Clone the distances (Vec<u32>)
            metric: self
                .metric
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|atomic_f32| {
                            // Here we clone by reading the value and then creating a new AtomicF32
                            AtomicF32::new(atomic_f32.load(Ordering::SeqCst))
                        })
                        .collect()
                })
                .collect(),
        }
    }
}

/// Calculates the rotation angle between two points relative to the origin.
#[pyfunction]
pub fn calculate_rotation(point_a: Coord, point_b: Coord) -> f32 {
    let ang_a = point_a.y.atan2(point_a.x);
    let ang_b = point_b.y.atan2(point_b.x);
    let rotation = (ang_a - ang_b) % (2.0 * PI);
    rotation.to_degrees()
}

/// Calculates the smallest difference angle between two vectors.
#[pyfunction]
pub fn calculate_rotation_smallest(vec_a: Coord, vec_b: Coord) -> f32 {
    let ang_a = vec_a.y.atan2(vec_a.x).to_degrees();
    let ang_b = vec_b.y.atan2(vec_b.x).to_degrees();
    let diff_angle = (ang_b - ang_a + 180.0) % 360.0 - 180.0;
    diff_angle.abs()
}

/// Validates that all elements in a 2D NumPy array are finite.
#[pyfunction]
pub fn check_numerical_data(data_arr: PyReadonlyArray2<f32>) -> PyResult<()> {
    let data_slice = data_arr.as_array();
    for inner_arr in data_slice.rows() {
        for &num in inner_arr.iter() {
            if !num.is_finite() {
                return Err(PyValueError::new_err(
                    "The numeric data values must be finite.",
                ));
            }
        }
    }
    Ok(())
}

/// Converts beta values to distances using a logarithmic transformation.
#[pyfunction]
#[pyo3(signature = (betas, min_threshold_wt=None))]
pub fn distances_from_betas(betas: Vec<f32>, min_threshold_wt: Option<f32>) -> PyResult<Vec<u32>> {
    if betas.is_empty() {
        return Err(PyValueError::new_err("Empty iterable of betas."));
    }
    let min_threshold_wt = min_threshold_wt.unwrap_or(MIN_THRESH_WT);
    let mut clean: Vec<f32> = Vec::new();
    let mut distances: Vec<u32> = Vec::new();

    for &beta in &betas {
        if beta < 0.0 {
            return Err(PyValueError::new_err(
                "Provide the beta value without the leading negative.",
            ));
        }
        if beta == 0.0 {
            return Err(PyValueError::new_err(
                "Provide a beta value greater than zero.",
            ));
        }
        if clean.contains(&beta) || clean.iter().any(|&x| x < beta) {
            return Err(PyValueError::new_err(
                "Betas must be free of duplicates and sorted in decreasing order.",
            ));
        }
        clean.push(beta);
        distances.push((min_threshold_wt.ln() / -beta).round() as u32);
    }
    Ok(distances)
}

/// Converts distances back to beta values.
#[pyfunction]
#[pyo3(signature = (distances, min_threshold_wt=None))]
pub fn betas_from_distances(
    distances: Vec<u32>,
    min_threshold_wt: Option<f32>,
) -> PyResult<Vec<f32>> {
    if distances.is_empty() {
        return Err(PyValueError::new_err("Empty iterable of distances."));
    }
    let mtw = min_threshold_wt.unwrap_or(MIN_THRESH_WT);
    let mut clean: Vec<u32> = Vec::new();
    let mut betas: Vec<f32> = Vec::new();
    for &distance in &distances {
        if distance == 0 {
            return Err(PyValueError::new_err(
                "Distances must be positive integers.",
            ));
        }
        if clean.contains(&distance) || clean.iter().any(|&x| x > distance) {
            return Err(PyValueError::new_err(
                "Distances must be free of duplicates and sorted in increasing order.",
            ));
        }
        clean.push(distance);
        betas.push((((-(mtw as f64).ln() / distance as f64) * 100000.0).round() / 100000.0) as f32);
    }
    Ok(betas)
}

#[pyfunction]
#[pyo3(signature = (walking_times, speed_m_s=None))]
pub fn distances_from_walking_times(
    walking_times: Vec<f32>,
    speed_m_s: Option<f32>,
) -> PyResult<Vec<u32>> {
    if walking_times.is_empty() {
        return Err(PyValueError::new_err("Empty iterable of walking times."));
    }
    let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
    let mut clean = Vec::new();
    let mut distances = Vec::new();
    for &time in &walking_times {
        if time == 0.0 {
            return Err(PyValueError::new_err("Time must be positive."));
        }
        if clean.contains(&time) || clean.iter().any(|&x| x > time) {
            return Err(PyValueError::new_err(
                "Times must be free of duplicates and sorted in increasing order.",
            ));
        }
        clean.push(time);
        distances.push(((time * 60.0) * speed_m_s).round() as u32)
    }
    Ok(distances)
}

#[pyfunction]
#[pyo3(signature = (distances, speed_m_s=None))]
pub fn walking_times_from_distances(
    distances: Vec<u32>,
    speed_m_s: Option<f32>,
) -> PyResult<Vec<f32>> {
    if distances.is_empty() {
        return Err(PyValueError::new_err("Empty iterable of distances."));
    }
    if distances.iter().any(|&dist| dist <= 0) {
        return Err(PyValueError::new_err(
            "Distances must be positive integers.",
        ));
    }
    let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
    let mut clean: Vec<u32> = Vec::new();
    let mut walking_times: Vec<f32> = Vec::new();
    for &distance in &distances {
        if distance == 0 {
            return Err(PyValueError::new_err(
                "Distances must be positive integers.",
            ));
        }
        if clean.contains(&distance) || clean.iter().any(|&x| x > distance) {
            return Err(PyValueError::new_err(
                "Distances must be free of duplicates and sorted in increasing order.",
            ));
        }
        clean.push(distance);
        walking_times.push((((distance as f32 / speed_m_s) / 60.0) * 100.0).round() / 100.0)
    }
    Ok(walking_times)
}

#[pyfunction]
#[pyo3(signature = (distances=None, betas=None, walking_times=None, min_threshold_wt=None, speed_m_s=None))]
pub fn pair_distances_betas_walking_times(
    distances: Option<Vec<u32>>,
    betas: Option<Vec<f32>>,
    walking_times: Option<Vec<f32>>,
    min_threshold_wt: Option<f32>,
    speed_m_s: Option<f32>,
) -> PyResult<(Vec<u32>, Vec<f32>, Vec<f32>)> {
    let min_threshold_wt = min_threshold_wt.unwrap_or(MIN_THRESH_WT);
    let speed_m_s = speed_m_s.unwrap_or(WALKING_SPEED);
    match (distances, betas, walking_times) {
        (Some(distances), None, None) => {
            let betas = betas_from_distances(distances.clone(), Some(min_threshold_wt))?;
            let walking_times = walking_times_from_distances(distances.clone(), Some(speed_m_s))?;
            Ok((distances, betas, walking_times))
        }
        (None, Some(betas), None) => {
            let distances = distances_from_betas(betas.clone(), Some(min_threshold_wt))?;
            let walking_times = walking_times_from_distances(distances.clone(), Some(speed_m_s))?;
            Ok((distances, betas, walking_times))
        }
        (None, None, Some(walking_times)) => {
            let distances = distances_from_walking_times(walking_times.clone(), Some(speed_m_s))?;
            let betas = betas_from_distances(distances.clone(), Some(min_threshold_wt))?;
            Ok((distances, betas, walking_times))
        }
        _ => Err(PyValueError::new_err(
            "Please provide exactly one of distances, betas, or walking_times.",
        )),
    }
}

/// Computes average distances for given beta values.
#[pyfunction]
#[pyo3(signature = (betas, min_threshold_wt=None))]
pub fn avg_distances_for_betas(
    betas: Vec<f32>,
    min_threshold_wt: Option<f32>,
) -> PyResult<Vec<f32>> {
    if betas.is_empty() {
        return Err(PyValueError::new_err("Empty iterable of betas."));
    }
    let min_threshold_wt = min_threshold_wt.unwrap_or(MIN_THRESH_WT);
    let distances = distances_from_betas(betas.clone(), Some(min_threshold_wt))?;
    let avg_distances: Vec<f32> = betas
        .iter()
        .zip(distances.iter())
        .map(|(&beta, &distance)| {
            if distance == 0 {
                return Err(PyValueError::new_err(
                    "Distances must be positive integers.",
                ));
            }
            let auc = ((-beta * distance as f32).exp() - 1.0) / -beta;
            let wt = auc / distance as f32;
            Ok(-wt.ln() / beta)
        })
        .collect::<PyResult<_>>()?;
    Ok(avg_distances)
}

/// Clips weights based on a spatial tolerance.
#[pyfunction]
pub fn clip_wts_curve(
    distances: Vec<u32>,
    betas: Vec<f32>,
    spatial_tolerance: u32,
) -> PyResult<Vec<f32>> {
    let max_curve_wts: Vec<f32> = distances
        .iter()
        .zip(betas.iter())
        .map(|(&dist, &beta)| {
            if spatial_tolerance > dist {
                return Err(PyValueError::new_err(
                    "Clipping distance cannot be greater than the given distance threshold.",
                ));
            }
            Ok((-beta * spatial_tolerance as f32).exp())
        })
        .collect::<PyResult<_>>()?;
    Ok(max_curve_wts)
}

/// Computes a clipped weight based on a beta value and maximum curve weight.
#[pyfunction]
pub fn clipped_beta_wt(beta: f32, max_curve_wt: f32, data_dist: f32) -> PyResult<f32> {
    if !(0.0..=1.0).contains(&max_curve_wt) {
        return Err(PyValueError::new_err(
            "Max curve weight must be in the range [0, 1].",
        ));
    }
    let raw_wt = (-beta * data_dist).exp();
    let clipped_wt = f32::min(raw_wt, max_curve_wt) / max_curve_wt;
    Ok(clipped_wt)
}
