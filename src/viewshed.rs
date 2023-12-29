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
fn line_of_sight(
    raster: ArrayView2<u8>,
    start_x: usize,
    start_y: usize,
    target_x: usize,
    target_y: usize,
) -> bool {
    let dx = isize::abs(target_x as isize - start_x as isize);
    let dy = -isize::abs(target_y as isize - start_y as isize);
    let sx: isize = if start_x < target_x { 1 } else { -1 };
    let sy: isize = if start_y < target_y { 1 } else { -1 };
    let mut err = dx + dy; // error term
    let mut current_x = start_x as isize;
    let mut current_y = start_y as isize;
    loop {
        // Check for obstruction
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
    for target_y in (start_y as isize - max_distance as isize).max(0) as usize
        ..(start_y as isize + max_distance as isize).min(height as isize) as usize
    {
        for target_x in (start_x as isize - max_distance as isize).max(0) as usize
            ..(start_x as isize + max_distance as isize).min(width as isize) as usize
        {
            if target_y == start_y && target_x == start_x {
                continue;
            }
            let distance = (((target_y as isize - start_y as isize).pow(2)
                + (target_x as isize - start_x as isize).pow(2)) as f64)
                .sqrt() as f32;
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
fn calculate_viewshed(
    raster: ArrayView2<u8>,
    start_x: usize,
    start_y: usize,
    max_distance: f32,
) -> Vec<u32> {
    let (height, width) = raster.dim();
    let mut visibility = vec![0; height * width];
    for target_y in (start_y as isize - max_distance as isize).max(0) as usize
        ..(start_y as isize + max_distance as isize).min(height as isize) as usize
    {
        for target_x in (start_x as isize - max_distance as isize).max(0) as usize
            ..(start_x as isize + max_distance as isize).min(width as isize) as usize
        {
            if target_y == start_y && target_x == start_x {
                continue;
            }
            let distance = (((target_y as isize - start_y as isize).pow(2)
                + (target_x as isize - start_x as isize).pow(2)) as f64)
                .sqrt() as f32;
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

#[pymethods]
impl Viewshed {
    #[new]
    fn new() -> Self {
        Self {
            progress: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn progress_init(&self) {
        self.progress.store(0, Ordering::Relaxed);
    }
    fn progress(&self) -> usize {
        self.progress.as_ref().load(Ordering::Relaxed)
    }
    pub fn visibility_graph(
        &self,
        bldgs_rast: PyReadonlyArray2<u8>,
        view_distance: f32,
        pbar_disabled: Option<bool>,
        py: Python,
    ) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
        // track progress
        let pbar_disabled = pbar_disabled.unwrap_or(false);
        self.progress_init();
        let raster_array = bldgs_rast.as_array().to_owned();
        let (height, width) = raster_array.dim();
        let viewsheds: Vec<(u32, f32, f32)> = py.allow_threads(move || {
            (0..height * width)
                .into_par_iter()
                .map(|index| {
                    if !pbar_disabled {
                        self.progress.fetch_add(1, Ordering::Relaxed);
                    }
                    let start_y = index / width;
                    let start_x = index % width;
                    let raster_view = raster_array.view();
                    calculate_visible_cells(raster_view, start_x, start_y, view_distance)
                })
                .collect()
        });
        // Unpack the tuples into three separate vectors
        let (results_u32, results_f32_a, results_f32_b) = viewsheds.into_iter().fold(
            (
                Vec::with_capacity(height * width),
                Vec::with_capacity(height * width),
                Vec::with_capacity(height * width),
            ),
            |(mut acc_u32, mut acc_f32_a, mut acc_f32_b), (val_u32, val_f32_a, val_f32_b)| {
                acc_u32.push(val_u32);
                acc_f32_a.push(val_f32_a);
                acc_f32_b.push(val_f32_b);
                (acc_u32, acc_f32_a, acc_f32_b)
            },
        );
        // Convert the results back to NumPy arrays
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

        Ok((array_u32, array_f32_a, array_f32_b))
    }
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
        Ok(numpy_array)
    }
}
