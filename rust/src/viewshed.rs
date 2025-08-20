use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[pyclass]
pub struct Viewshed {
    pub progress: Arc<AtomicUsize>,
}

#[inline]
fn line_of_sight_elevation(
    raster: ArrayView2<f32>,
    start_x: usize,
    start_y: usize,
    target_x: usize,
    target_y: usize,
    observer_height: f32,
) -> bool {
    let (height, width) = raster.dim();
    let start_elev = raster[(start_y, start_x)] + observer_height;
    let end_elev = raster[(target_y, target_x)];

    let dx = target_x as isize - start_x as isize;
    let dy = target_y as isize - start_y as isize;

    let sx = if dx > 0 { 1 } else { -1 };
    let sy = if dy > 0 { 1 } else { -1 };

    let dx_abs = dx.abs();
    let dy_abs = dy.abs();

    let mut err = dx_abs - dy_abs;

    let mut x = start_x as isize;
    let mut y = start_y as isize;

    let dist_total_sq = (dx * dx + dy * dy) as f32;
    if dist_total_sq == 0.0 {
        return true; // Target is the same as start
    }
    let slope_target = (end_elev - start_elev) / dist_total_sq;
    let mut max_slope = std::f32::MIN;

    let mut rel_x = 0;
    let mut rel_y = 0;

    // We skip the starting point by advancing once before the loop
    let e2_init = 2 * err;
    if e2_init > -dy_abs {
        err -= dy_abs;
        x += sx;
        rel_x += sx;
    }
    if e2_init < dx_abs {
        err += dx_abs;
        y += sy;
        rel_y += sy;
    }

    while x != target_x as isize || y != target_y as isize {
        if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
            return false; // Out of bounds is considered occluded
        }

        let dist_sq = (rel_x * rel_x + rel_y * rel_y) as f32;
        let current_elev = raster[(y as usize, x as usize)];

        if dist_sq > 0.0 {
            let slope = (current_elev - start_elev) / dist_sq;
            if slope > max_slope {
                max_slope = slope;
            }
        }

        if max_slope > slope_target {
            return false;
        }

        let e2 = 2 * err;
        if e2 > -dy_abs {
            err -= dy_abs;
            x += sx;
            rel_x += sx;
        }
        if e2 < dx_abs {
            err += dx_abs;
            y += sy;
            rel_y += sy;
        }
    }

    true
}

#[inline]
fn calculate_visible_cells(
    raster: ArrayView2<f32>,
    max_res_dist: f32,
    observer_height: f32,
    start_x: usize,
    start_y: usize,
) -> (u32, f32, f32) {
    let (height, width) = raster.dim();
    let mut density: u32 = 0;
    let mut farness: f32 = 0.0;
    let mut harmonic: f32 = 0.0;

    // Calculate bounds relative to start pixel and image edge
    let dist_floor = max_res_dist.floor() as isize;
    let min_y = (start_y as isize - dist_floor).max(0) as usize;
    let max_y = (start_y as isize + dist_floor).min(height as isize - 1) as usize;
    let min_x = (start_x as isize - dist_floor).max(0) as usize;
    let max_x = (start_x as isize + dist_floor).min(width as isize - 1) as usize;

    for target_y in min_y..=max_y {
        for target_x in min_x..=max_x {
            if target_y == start_y && target_x == start_x {
                continue;
            }
            let distance = f32::hypot(
                (target_y as isize - start_y as isize) as f32,
                (target_x as isize - start_x as isize) as f32,
            );
            if distance > max_res_dist {
                continue;
            }
            if line_of_sight_elevation(
                raster,
                start_x,
                start_y,
                target_x,
                target_y,
                observer_height,
            ) {
                if distance > 0.0 {
                    density += 1;
                    farness += distance;
                    harmonic += 1.0 / distance;
                }
            }
        }
    }
    (density, farness, harmonic)
}

#[inline]
fn calculate_viewshed(
    raster: ArrayView2<f32>,
    max_res_dist: f32,
    observer_height: f32,
    start_x: usize,
    start_y: usize,
) -> Vec<u32> {
    let (height, width) = raster.dim();
    let mut visibility = vec![0; height * width];

    // Calculate bounds relative to start pixel and image edge
    let dist_floor = max_res_dist.floor() as isize;
    let min_y = (start_y as isize - dist_floor).max(0) as usize;
    let max_y = (start_y as isize + dist_floor).min(height as isize - 1) as usize;
    let min_x = (start_x as isize - dist_floor).max(0) as usize;
    let max_x = (start_x as isize + dist_floor).min(width as isize - 1) as usize;

    for target_y in min_y..=max_y {
        for target_x in min_x..=max_x {
            if target_y == start_y && target_x == start_x {
                continue;
            }
            let distance = f32::hypot(
                (target_y as isize - start_y as isize) as f32,
                (target_x as isize - start_x as isize) as f32,
            );
            if distance > max_res_dist {
                continue;
            }
            if line_of_sight_elevation(
                raster,
                start_x,
                start_y,
                target_x,
                target_y,
                observer_height,
            ) {
                visibility[target_y * width + target_x] = 1;
            }
        }
    }
    visibility
}

/// Helper to unzip a vector of 3-tuples into three vectors.
/// Used for unpacking results in calculate_visibility.
#[inline]
fn unzip3<T, U, V>(v: Vec<(T, U, V)>) -> (Vec<T>, Vec<U>, Vec<V>) {
    let mut t = Vec::with_capacity(v.len());
    let mut u = Vec::with_capacity(v.len());
    let mut w = Vec::with_capacity(v.len());
    for (a, b, c) in v {
        t.push(a);
        u.push(b);
        w.push(c);
    }
    (t, u, w)
}

#[pymethods]
impl Viewshed {
    #[new]
    fn new() -> Self {
        Self {
            progress: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Reset the progress counter to zero.
    pub fn progress_init(&self) {
        self.progress.store(0, Ordering::Relaxed);
    }

    /// Get the current progress value.
    fn progress(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }

    /// Compute the visibility graph for the given raster and view distance.
    #[pyo3(signature = (bldgs_rast, view_distance, resolution, observer_height, pbar_disabled=None))]
    pub fn visibility(
        &self,
        bldgs_rast: PyReadonlyArray2<f32>,
        view_distance: f32,
        resolution: f32,
        observer_height: f32,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        let raster_array = bldgs_rast.as_array();
        let (height, width) = raster_array.dim();
        if resolution <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Resolution must be greater than zero.",
            ));
        }
        let max_res_dist = view_distance / resolution; // Ensure we don't divide by zero

        let results: Vec<(u32, f32, f32)> = py.allow_threads(move || {
            (0..height * width)
                .into_par_iter()
                .map(|index| {
                    if !pbar_disabled {
                        self.progress.fetch_add(1, Ordering::Relaxed);
                    }
                    let start_y = index / width;
                    let start_x = index % width;
                    calculate_visible_cells(
                        raster_array,
                        max_res_dist,
                        observer_height,
                        start_x,
                        start_y,
                    )
                })
                .collect()
        });
        let (results_u32, results_f32_a, results_f32_b) = unzip3(results);

        let array_u32 = Array2::from_shape_vec((height, width), results_u32)
            .unwrap()
            .into_pyarray(py)
            .to_owned();
        let array_f32_a = Array2::from_shape_vec((height, width), results_f32_a)
            .unwrap()
            .into_pyarray(py)
            .to_owned();
        let array_f32_b = Array2::from_shape_vec((height, width), results_f32_b)
            .unwrap()
            .into_pyarray(py)
            .to_owned();

        Ok((array_u32.into(), array_f32_a.into(), array_f32_b.into()))
    }

    /// Compute the viewshed for a single origin cell.
    #[pyo3(signature = (bldgs_rast, view_distance, resolution, observer_height, origin_x, origin_y))]
    pub fn viewshed(
        &self,
        bldgs_rast: PyReadonlyArray2<f32>,
        view_distance: f32,
        resolution: f32,
        observer_height: f32,
        origin_x: usize,
        origin_y: usize,
        py: Python,
    ) -> PyResult<Py<PyArray2<u32>>> {
        let raster_array = bldgs_rast.as_array();
        let (height, width) = raster_array.dim();
        if resolution <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Resolution must be greater than zero.",
            ));
        }
        let max_res_dist = view_distance / resolution; // Ensure we don't divide by zero
        let visibility = calculate_viewshed(
            raster_array,
            max_res_dist,
            observer_height,
            origin_x,
            origin_y,
        );
        let numpy_array = Array2::from_shape_vec((height, width), visibility)
            .unwrap()
            .into_pyarray(py)
            .to_owned();
        Ok(numpy_array.into())
    }
}

// Optionally, add a test module skeleton for future tests.
#[cfg(test)]
mod tests {
    // use super::*;
    // #[test]
    // fn test_viewshed_basic() {
    //     // Add tests here
    // }
}
