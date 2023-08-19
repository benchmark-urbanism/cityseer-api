use atomic_float::AtomicF32;
use numpy::borrow::PyReadonlyArray2;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::atomic::Ordering;

static MIN_THRESH_WT: f32 = 0.01831563888873418;

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
    #[new]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    pub fn xy(&self) -> (f32, f32) {
        (self.x, self.y)
    }
    pub fn validate(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
    pub fn hypot(&self, other_coord: Coord) -> f32 {
        ((self.x - other_coord.x).powi(2) + (self.y - other_coord.y).powi(2)).sqrt()
    }
    pub fn difference(&self, other_coord: Coord) -> Coord {
        // using Coord struct as a vector
        let x_diff = self.x - other_coord.x;
        let y_diff = self.y - other_coord.y;
        Coord::new(x_diff, y_diff)
    }
}
pub struct MetricResult {
    pub distances: Vec<u32>,
    pub metric: Vec<Vec<AtomicF32>>,
}
impl MetricResult {
    pub fn new(distances: Vec<u32>, size: usize, init_val: f32) -> Self {
        let mut metric = Vec::new();
        for _d in 0..distances.len() {
            metric.push(
                // tricky to initialise for given size
                std::iter::repeat_with(|| AtomicF32::new(init_val))
                    .take(size)
                    .collect::<Vec<AtomicF32>>(),
            );
        }
        Self { distances, metric }
    }
    pub fn load(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        let mut loaded: HashMap<u32, Py<PyArray1<f32>>> = HashMap::new();
        for i in 0..self.distances.len() {
            let dist = self.distances[i];
            let vec_f32: Vec<f32> = self.metric[i]
                .iter()
                .map(|a| a.load(Ordering::SeqCst))
                .collect();
            let array = Python::with_gil(|py| vec_f32.into_pyarray(py).to_owned());
            loaded.insert(dist, array);
        }
        loaded
    }
}
#[pyfunction]
pub fn calculate_rotation(point_a: Coord, point_b: Coord) -> f32 {
    let ang_a = point_a.y.atan2(point_a.x);
    let ang_b = point_b.y.atan2(point_b.x);
    let rotation = (ang_a - ang_b) % (2.0 * PI);
    rotation.to_degrees()
}
// https://stackoverflow.com/questions/37459121/calculating-angle-between-three-points-but-only-anticlockwise-in-python
// these two points / angles are relative to the origin
// pass in difference between the points and origin as vectors
#[pyfunction]
pub fn calculate_rotation_smallest(vec_a: Coord, vec_b: Coord) -> f32 {
    // Convert angles from radians to degrees and calculate the smallest difference angle
    let ang_a = (vec_a.y.atan2(vec_a.x)).to_degrees();
    let ang_b = (vec_b.y.atan2(vec_b.x)).to_degrees();
    let diff_angle = (ang_b - ang_a + 180.0) % 360.0 - 180.0;
    diff_angle.abs()
}
#[pyfunction]
pub fn check_numerical_data(data_arr: PyReadonlyArray2<f32>) -> PyResult<()> {
    // Check the integrity of numeric data arrays.
    let data_slice = data_arr.as_array();
    for inner_arr in data_slice.rows() {
        for num in inner_arr.iter() {
            let num_val = *num;
            if !num_val.is_finite() {
                return Err(exceptions::PyValueError::new_err(
                    "The numeric data values must be finite.",
                ));
            }
        }
    }
    Ok(())
}

#[pyfunction]
pub fn distances_from_betas(betas: Vec<f32>, min_threshold_wt: Option<f32>) -> PyResult<Vec<u32>> {
    if betas.len() == 0 {
        return Err(exceptions::PyValueError::new_err(
            "Empty iterable of betas.",
        ));
    }
    let min_threshold_wt = min_threshold_wt.unwrap_or(MIN_THRESH_WT);
    let mut clean: Vec<f32> = Vec::new();
    let mut distances: Vec<u32> = Vec::new();
    for beta in betas.iter() {
        if *beta < 0.0 {
            return Err(exceptions::PyValueError::new_err(
                "Provide the beta value without the leading negative.",
            ));
        }
        if *beta == 0.0 {
            return Err(exceptions::PyValueError::new_err(
                "Provide a beta value greater than zero.",
            ));
        }
        if clean.contains(beta) || clean.iter().any(|&x| x < *beta) {
            return Err(exceptions::PyValueError::new_err(
                "Betas must be free of duplicates and sorted in decreasing order.",
            ));
        }
        clean.push(*beta);
        distances.push((min_threshold_wt.ln() / -beta).round() as u32);
    }
    Ok(distances)
}

#[pyfunction]
pub fn betas_from_distances(
    distances: Vec<u32>,
    min_threshold_wt: Option<f32>,
) -> PyResult<Vec<f32>> {
    if distances.len() == 0 {
        return Err(exceptions::PyValueError::new_err(
            "Empty iterable of distances.",
        ));
    }
    let min_threshold_wt = min_threshold_wt.unwrap_or(MIN_THRESH_WT);
    let mut clean: Vec<u32> = Vec::new();
    let mut betas: Vec<f32> = Vec::new();
    for distance in distances.iter() {
        if *distance <= 0 {
            return Err(exceptions::PyValueError::new_err(
                "Distances must be positive integers.",
            ));
        }
        if clean.contains(distance) || clean.iter().any(|&x| x > *distance) {
            return Err(exceptions::PyValueError::new_err(
                "Distances must be free of duplicates and sorted in increasing order.",
            ));
        }
        clean.push(*distance);
        betas.push(-min_threshold_wt.ln() / *distance as f32);
    }
    Ok(betas)
}

#[pyfunction]
pub fn pair_distances_and_betas(
    distances: Option<Vec<u32>>,
    betas: Option<Vec<f32>>,
    min_threshold_wt: Option<f32>,
) -> PyResult<(Vec<u32>, Vec<f32>)> {
    if distances.is_some() && betas.is_some() {
        return Err(exceptions::PyValueError::new_err(
            "Please provide either a distances or betas, not both.",
        ));
    }
    if distances.is_none() && betas.is_none() {
        return Err(exceptions::PyValueError::new_err(
            "Please provide either a distances or betas. Neither has been provided",
        ));
    }
    let betas = if betas.is_some() {
        betas.unwrap()
    } else {
        betas_from_distances(distances.clone().unwrap(), min_threshold_wt)?
    };
    let distances = if distances.is_some() {
        distances.unwrap()
    } else {
        distances_from_betas(betas.clone(), min_threshold_wt)?
    };
    Ok((distances, betas))
}

#[pyfunction]
pub fn avg_distances_for_betas(
    betas: Vec<f32>,
    min_threshold_wt: Option<f32>,
) -> PyResult<Vec<f32>> {
    if betas.len() == 0 {
        return Err(exceptions::PyValueError::new_err(
            "Empty iterable of betas.",
        ));
    }
    let min_threshold_wt = min_threshold_wt.unwrap_or(MIN_THRESH_WT);
    let mut avg_distances: Vec<f32> = Vec::new();
    let distances = distances_from_betas(betas.clone(), Some(min_threshold_wt))?;
    for (beta, distance) in betas.iter().zip(distances.iter()) {
        if *distance <= 0 {
            return Err(exceptions::PyValueError::new_err(
                "Distances must be positive integers.",
            ));
        }
        let auc = ((-beta * *distance as f32).exp() - 1.0) / -beta;
        let wt = auc / *distance as f32;
        let avg_d = -wt.ln() / beta;
        avg_distances.push(avg_d)
    }
    Ok(avg_distances)
}

#[pyfunction]
pub fn clip_wts_curve(
    distances: Vec<u32>,
    betas: Vec<f32>,
    spatial_tolerance: u32,
) -> PyResult<Vec<f32>> {
    let mut max_curve_wts: Vec<f32> = Vec::new();
    for (dist, beta) in distances.iter().zip(betas.iter()) {
        if spatial_tolerance > *dist {
            return Err(exceptions::PyValueError::new_err(
                "Clipping distance cannot be greater than the given distance threshold.",
            ));
        }
        let max_curve_wt = (-beta * spatial_tolerance as f32).exp();
        if max_curve_wt < 0.75 {}
        max_curve_wts.push(max_curve_wt);
    }
    Ok(max_curve_wts)
}

#[pyfunction]
pub fn clipped_beta_wt(beta: f32, max_curve_wt: f32, data_dist: f32) -> PyResult<f32> {
    if beta < 0.0 || beta > 1.0 {
        return Err(exceptions::PyValueError::new_err(
            "Max curve weight must be in a range of 0 - 1.",
        ));
    }
    if max_curve_wt < 0.0 || max_curve_wt > 1.0 {
        return Err(exceptions::PyValueError::new_err(
            "Max curve weight must be in a range of 0 - 1.",
        ));
    }
    // Calculates negative exponential clipped to the max_curve_wt parameter.
    let raw_wt = (-beta * data_dist).exp();
    let clipped_wt = f32::min(raw_wt, max_curve_wt) / max_curve_wt;
    Ok(clipped_wt)
}
