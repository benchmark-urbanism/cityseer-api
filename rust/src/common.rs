use atomic_float::AtomicF32;
use numpy::borrow::PyReadonlyArray2;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

/// Minimum threshold weight for distance and beta calculations.
static MIN_THRESH_WT: f32 = 0.01831563888873418;
/// Walking speed in meters per second.
pub static WALKING_SPEED: f32 = 1.33333;
/// Default interval for progress updates.
pub static PROGRESS_UPDATE_INTERVAL: usize = 100;

/// Holds metric results, including distances and a 2D matrix of atomic floats.
#[derive(Debug)]
pub struct MetricResult {
    pub distances: Vec<u32>,
    pub metric: Vec<Vec<AtomicF32>>,
}

impl MetricResult {
    /// Initializes a new `MetricResult` with given distances, size, and initial value.
    #[inline]
    pub fn new(distances: Vec<u32>, size: usize, init_val: f32) -> Self {
        let metric = distances
            .iter()
            .map(|_| (0..size).map(|_| AtomicF32::new(init_val)).collect())
            .collect();
        Self { distances, metric }
    }

    /// Converts the atomic floats into a Python-compatible format (`PyArray1<f32>`).
    #[inline]
    pub fn load(&self) -> HashMap<u32, Py<PyArray1<f32>>> {
        self.distances
            .iter()
            .zip(&self.metric)
            .map(|(&dist, row)| {
                let vec_f32: Vec<f32> = row.iter().map(|a| a.load(Ordering::Relaxed)).collect();
                let array = Python::with_gil(|py| vec_f32.into_pyarray(py).to_owned().into());
                (dist, array)
            })
            .collect()
    }
}

// Manually implement Clone for MetricResult
impl Clone for MetricResult {
    fn clone(&self) -> Self {
        MetricResult {
            distances: self.distances.clone(),
            metric: self
                .metric
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|atomic_f32| AtomicF32::new(atomic_f32.load(Ordering::Relaxed)))
                        .collect()
                })
                .collect(),
        }
    }
}

/// Validates that all elements in a 2D NumPy array are finite.
#[pyfunction]
pub fn check_numerical_data(data_arr: PyReadonlyArray2<f32>) -> PyResult<()> {
    let data_slice = data_arr.as_array();
    for inner_arr in data_slice.rows() {
        for &num in inner_arr {
            if !num.is_finite() {
                return Err(PyValueError::new_err(
                    "The numeric data values must be finite.",
                ));
            }
        }
    }
    Ok(())
}

/// Helper to generate a composite key from a Python object.
pub fn py_key_to_composite(py_obj: Bound<'_, PyAny>) -> PyResult<String> {
    let type_name = py_obj.get_type().name()?;
    let value_pystr = py_obj.str()?;
    let value_str = value_pystr.to_str()?;
    Ok(format!("{}:{}", type_name, value_str))
}

/// Converts beta values to distances using a logarithmic transformation.
#[pyfunction]
#[pyo3(signature = (betas, min_threshold_wt=None))]
pub fn distances_from_betas(betas: Vec<f32>, min_threshold_wt: Option<f32>) -> PyResult<Vec<u32>> {
    if betas.is_empty() {
        return Err(PyValueError::new_err("Input 'betas' cannot be empty."));
    }
    let min_threshold_wt = min_threshold_wt.unwrap_or(MIN_THRESH_WT);

    // Check strictly decreasing order and uniqueness
    if betas.windows(2).any(|w| w[1] >= w[0]) {
        return Err(PyValueError::new_err(
            "Betas must be unique and sorted in strictly decreasing order.",
        ));
    }

    betas
        .iter()
        .map(|&beta| {
            if beta <= 0.0 {
                return Err(PyValueError::new_err(
                    "Beta values must be greater than zero.",
                ));
            }
            let distance = (min_threshold_wt.ln() / -beta).round();
            if distance <= 0.0 {
                return Err(PyValueError::new_err(
                    "Derived distance must be positive. Check beta values.",
                ));
            }
            Ok(distance as u32)
        })
        .collect()
}

/// Converts distances back to beta values.
#[pyfunction]
#[pyo3(signature = (distances, min_threshold_wt=None))]
pub fn betas_from_distances(
    distances: Vec<u32>,
    min_threshold_wt: Option<f32>,
) -> PyResult<Vec<f32>> {
    if distances.is_empty() {
        return Err(PyValueError::new_err("Input 'distances' cannot be empty."));
    }
    let mtw = min_threshold_wt.unwrap_or(MIN_THRESH_WT);

    // Check strictly increasing order and uniqueness
    if distances.windows(2).any(|w| w[1] <= w[0]) {
        return Err(PyValueError::new_err(
            "Distances must be unique and sorted in strictly increasing order.",
        ));
    }

    distances
        .iter()
        .map(|&distance| {
            if distance == 0 {
                return Err(PyValueError::new_err(
                    "Distances must be positive integers.",
                ));
            }
            let beta = -mtw.ln() / distance as f32;
            let beta_rounded = (beta * 100_000.0).round() / 100_000.0;
            Ok(beta_rounded)
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (seconds, speed_m_s))]
pub fn distances_from_seconds(seconds: Vec<u32>, speed_m_s: f32) -> PyResult<Vec<u32>> {
    if seconds.is_empty() {
        return Err(PyValueError::new_err("Input 'seconds' cannot be empty."));
    }
    if speed_m_s <= 0.0 {
        return Err(PyValueError::new_err("Speed must be positive."));
    }

    // Check strictly increasing order and uniqueness
    if seconds.windows(2).any(|w| w[1] <= w[0]) {
        return Err(PyValueError::new_err(
            "Times must be unique and sorted in strictly increasing order.",
        ));
    }

    seconds
        .iter()
        .map(|&time| {
            if time == 0 {
                return Err(PyValueError::new_err(
                    "Time values must be positive integers.",
                ));
            }
            let distance = (time as f32 * speed_m_s).round();
            if distance <= 0.0 {
                return Err(PyValueError::new_err(
                    "Derived distance must be positive. Check time and speed values.",
                ));
            }
            Ok(distance as u32)
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (distances, speed_m_s))]
pub fn seconds_from_distances(distances: Vec<u32>, speed_m_s: f32) -> PyResult<Vec<u32>> {
    if distances.is_empty() {
        return Err(PyValueError::new_err("Input 'distances' cannot be empty."));
    }
    if speed_m_s <= 0.0 {
        return Err(PyValueError::new_err("Speed must be positive."));
    }

    // Check strictly increasing order and uniqueness
    if distances.windows(2).any(|w| w[1] <= w[0]) {
        return Err(PyValueError::new_err(
            "Distances must be unique and sorted in strictly increasing order.",
        ));
    }

    distances
        .iter()
        .map(|&distance| {
            if distance == 0 {
                return Err(PyValueError::new_err(
                    "Distances must be positive integers.",
                ));
            }
            let time = (distance as f32 / speed_m_s).round();
            if time <= 0.0 {
                return Err(PyValueError::new_err(
                    "Derived time must be positive. Check distance and speed values.",
                ));
            }
            Ok(time as u32)
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (speed_m_s, distances=None, betas=None, minutes=None, min_threshold_wt=None))]
pub fn pair_distances_betas_time(
    speed_m_s: f32,
    distances: Option<Vec<u32>>,
    betas: Option<Vec<f32>>,
    minutes: Option<Vec<f32>>,
    min_threshold_wt: Option<f32>,
) -> PyResult<(Vec<u32>, Vec<f32>, Vec<u32>)> {
    let min_threshold_wt = min_threshold_wt.unwrap_or(MIN_THRESH_WT);
    match (distances, betas, minutes) {
        (Some(distances), None, None) => {
            let betas = betas_from_distances(distances.clone(), Some(min_threshold_wt))?;
            let seconds = seconds_from_distances(distances.clone(), speed_m_s)?;
            Ok((distances, betas, seconds))
        }
        (None, Some(betas), None) => {
            let distances = distances_from_betas(betas.clone(), Some(min_threshold_wt))?;
            let seconds = seconds_from_distances(distances.clone(), speed_m_s)?;
            Ok((distances, betas, seconds))
        }
        (None, None, Some(minutes)) => {
            let seconds: Vec<u32> = minutes.iter().map(|&x| (x * 60.0).round() as u32).collect();
            let distances = distances_from_seconds(seconds.clone(), speed_m_s)?;
            let betas = betas_from_distances(distances.clone(), Some(min_threshold_wt))?;
            Ok((distances, betas, seconds))
        }
        _ => Err(PyValueError::new_err(
            "Please provide exactly one of the following arguments: 'distances', 'betas', or 'minutes'.",
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
        return Err(PyValueError::new_err("Input 'betas' cannot be empty."));
    }
    let min_threshold_wt = min_threshold_wt.unwrap_or(MIN_THRESH_WT);
    let distances = distances_from_betas(betas.clone(), Some(min_threshold_wt))?;

    betas
        .iter()
        .zip(distances.iter())
        .map(|(&beta, &distance)| {
            if beta.abs() < f32::EPSILON {
                return Err(PyValueError::new_err(format!(
                    "Beta ({}) must not be zero during average distance calculation.",
                    beta
                )));
            }
            let auc = ((-beta * distance as f32).exp() - 1.0) / -beta;
            let wt = auc / distance as f32;
            if wt <= 0.0 {
                Err(PyValueError::new_err(format!(
                    "Invalid weight ({}) encountered during average distance calculation.",
                    wt
                )))
            } else {
                Ok(-wt.ln() / beta)
            }
        })
        .collect()
}

/// Clips weights based on a spatial tolerance.
#[pyfunction]
pub fn clip_wts_curve(
    distances: Vec<u32>,
    betas: Vec<f32>,
    spatial_tolerance: u32,
) -> PyResult<Vec<f32>> {
    if distances.len() != betas.len() {
        return Err(PyValueError::new_err(
            "Input 'distances' and 'betas' must have the same length.",
        ));
    }
    distances
        .iter()
        .zip(betas.iter())
        .map(|(&dist, &beta)| {
            if spatial_tolerance > dist {
                return Err(PyValueError::new_err(format!(
                    "Clipping distance ({}) cannot be greater than the distance threshold ({}).",
                    spatial_tolerance, dist
                )));
            }
            Ok((-beta * spatial_tolerance as f32).exp())
        })
        .collect()
}

/// Computes a clipped weight based on a beta value and maximum curve weight.
#[pyfunction]
#[inline(always)]
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
