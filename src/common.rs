use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions;
use pyo3::prelude::*;

static MIN_THRESH_WT: f32 = 0.01831563888873418;

#[pyfunction]
pub fn check_numerical_data(data_arr: PyReadonlyArray1<f32>) -> PyResult<()> {
    if data_arr.ndim() != 2 {
        return Err(exceptions::PyValueError::new_err(
            "The numeric data array must have a dimensionality 2, \
             consisting of the number of respective data arrays x the length of data points.",
        ));
    }
    while let Ok(num) = data_arr.iter() {
        let num_val = num.extract::<f32>()?;
        if num_val.is_infinite() {
            return Err(exceptions::PyValueError::new_err(
                "The numeric data values must consist of either floats or NaNs.",
            ));
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
pub fn clipped_beta_wt(beta: f32, max_curve_wt: f32, data_dist: u32) -> PyResult<f32> {
    let raw_wt = (-beta * data_dist as f32).exp();
    let clipped_wt = f32::min(raw_wt, max_curve_wt) / max_curve_wt;
    Ok(clipped_wt)
}
