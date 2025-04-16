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
fn line_of_sight(
    raster: ArrayView2<u8>,
    start_x: usize,
    start_y: usize,
    target_x: usize,
    target_y: usize,
) -> bool {
    let dx = start_x.abs_diff(target_x) as isize;
    let dy = -(start_y.abs_diff(target_y) as isize);
    let sx: isize = if start_x < target_x { 1 } else { -1 };
    let sy: isize = if start_y < target_y { 1 } else { -1 };
    let mut err = dx + dy;
    let mut current_x = start_x as isize;
    let mut current_y = start_y as isize;
    loop {
        if raster.get((current_y as usize, current_x as usize)) == Some(&1) {
            return false;
        }
        if current_x == target_x as isize && current_y == target_y as isize {
            return true;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if current_x == target_x as isize {
                break;
            }
            err += dy;
            current_x += sx;
        }
        if e2 <= dx {
            if current_y == target_y as isize {
                break;
            }
            err += dx;
            current_y += sy;
        }
    }
    true
}

#[inline]
fn calculate_visible_cells(
    raster: ArrayView2<u8>,
    start_x: usize,
    start_y: usize,
    max_distance: f32,
) -> (u32, f32, f32) {
    let (height, width) = raster.dim();
    let mut density: u32 = 0;
    let mut farness: f32 = 0.0;
    let mut harmonic: f32 = 0.0;
    let min_y = start_y.saturating_sub(max_distance as usize);
    let max_y = (start_y + max_distance as usize).min(height.saturating_sub(1));
    let min_x = start_x.saturating_sub(max_distance as usize);
    let max_x = (start_x + max_distance as usize).min(width.saturating_sub(1));

    for target_y in min_y..=max_y {
        for target_x in min_x..=max_x {
            if target_y == start_y && target_x == start_x {
                continue;
            }
            let distance = f32::hypot(
                (target_y as isize - start_y as isize) as f32,
                (target_x as isize - start_x as isize) as f32,
            );
            if distance > max_distance {
                continue;
            }
            if line_of_sight(raster, start_x, start_y, target_x, target_y) {
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
    raster: ArrayView2<u8>,
    start_x: usize,
    start_y: usize,
    max_distance: f32,
) -> Vec<u32> {
    let (height, width) = raster.dim();
    let mut visibility = vec![0; height * width];
    let min_y = start_y.saturating_sub(max_distance as usize);
    let max_y = (start_y + max_distance as usize).min(height.saturating_sub(1));
    let min_x = start_x.saturating_sub(max_distance as usize);
    let max_x = (start_x + max_distance as usize).min(width.saturating_sub(1));

    for target_y in min_y..=max_y {
        for target_x in min_x..=max_x {
            if target_y == start_y && target_x == start_x {
                continue;
            }
            let distance = f32::hypot(
                (target_y as isize - start_y as isize) as f32,
                (target_x as isize - start_x as isize) as f32,
            );
            if distance > max_distance {
                continue;
            }
            if line_of_sight(raster, start_x, start_y, target_x, target_y) {
                visibility[target_y * width + target_x] = 1;
            }
        }
    }
    visibility
}

/// Helper to unzip a vector of 3-tuples into three vectors.
/// Used for unpacking results in visibility_graph.
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
    #[pyo3(signature = (bldgs_rast, view_distance, pbar_disabled=None))]
    pub fn visibility_graph(
        &self,
        bldgs_rast: PyReadonlyArray2<u8>,
        view_distance: f32,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        let raster_array = bldgs_rast.as_array();
        let (height, width) = raster_array.dim();
        let results: Vec<(u32, f32, f32)> = py.allow_threads(move || {
            (0..height * width)
                .into_par_iter()
                .map(|index| {
                    if !pbar_disabled {
                        self.progress.fetch_add(1, Ordering::Relaxed);
                    }
                    let start_y = index / width;
                    let start_x = index % width;
                    calculate_visible_cells(raster_array, start_x, start_y, view_distance)
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
    pub fn viewshed(
        &self,
        bldgs_rast: PyReadonlyArray2<u8>,
        view_distance: f32,
        origin_x: usize,
        origin_y: usize,
        py: Python,
    ) -> PyResult<Py<PyArray2<u32>>> {
        let raster_array = bldgs_rast.as_array();
        let (height, width) = raster_array.dim();
        let visibility = calculate_viewshed(raster_array, origin_x, origin_y, view_distance);
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
