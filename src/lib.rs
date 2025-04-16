//! Cityseer API Rust module: high-performance spatial algorithms for Python via PyO3.

use pyo3::prelude::*;

// Module imports (alphabetical for clarity)
mod centrality;
mod common;
mod data;
mod diversity;
mod graph;
mod viewshed;

/// Cityseer API implementation in Rust for performance-critical algorithms.
/// Exposes network, centrality, and diversity computation functions to Python.
#[pymodule]
fn rustalgos(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes and functions
    py_module.add_class::<common::Coord>()?;
    py_module.add_function(wrap_pyfunction!(common::check_numerical_data, py_module)?)?;
    py_module.add_function(wrap_pyfunction!(common::distances_from_betas, py_module)?)?;
    py_module.add_function(wrap_pyfunction!(common::betas_from_distances, py_module)?)?;
    py_module.add_function(wrap_pyfunction!(common::distances_from_seconds, py_module)?)?;
    py_module.add_function(wrap_pyfunction!(common::seconds_from_distances, py_module)?)?;
    py_module.add_function(wrap_pyfunction!(
        common::pair_distances_betas_time,
        py_module
    )?)?;
    py_module.add_function(wrap_pyfunction!(
        common::avg_distances_for_betas,
        py_module
    )?)?;
    py_module.add_function(wrap_pyfunction!(common::clip_wts_curve, py_module)?)?;
    py_module.add_function(wrap_pyfunction!(common::clipped_beta_wt, py_module)?)?;

    // Register submodules
    register_data_module(py_module)?;
    register_diversity_module(py_module)?;
    register_graph_module(py_module)?;
    register_centrality_module(py_module)?;
    register_viewshed_module(py_module)?;

    py_module.add(
        "__doc__",
        "Cityseer high-performance algorithms implemented in Rust.",
    )?;

    Ok(())
}

/// Registers data-related classes and structures for spatial data analysis.
fn register_data_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "data")?;
    submodule.add(
        "__doc__",
        "Data structures and utilities for spatial data analysis.",
    )?;
    submodule.add_class::<data::DataEntry>()?;
    submodule.add_class::<data::DataMap>()?;
    submodule.add_class::<data::AccessibilityResult>()?;
    submodule.add_class::<data::MixedUsesResult>()?;
    submodule.add_class::<data::StatsResult>()?;
    submodule.add_function(wrap_pyfunction!(data::node_matches_for_coord, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

/// Registers diversity-related functions for spatial diversity metrics.
fn register_diversity_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "diversity")?;
    submodule.add(
        "__doc__",
        "Functions for calculating diversity metrics in spatial analysis.",
    )?;
    submodule.add_function(wrap_pyfunction!(diversity::hill_diversity, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(
        diversity::hill_diversity_branch_distance_wt,
        &submodule
    )?)?;
    submodule.add_function(wrap_pyfunction!(
        diversity::hill_diversity_pairwise_distance_wt,
        &submodule
    )?)?;
    submodule.add_function(wrap_pyfunction!(
        diversity::gini_simpson_diversity,
        &submodule
    )?)?;
    submodule.add_function(wrap_pyfunction!(diversity::shannon_diversity, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(
        diversity::raos_quadratic_diversity,
        &submodule
    )?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

/// Registers graph-related classes for network analysis.
fn register_graph_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "graph")?;
    submodule.add(
        "__doc__",
        "Graph data structures and utilities for network analysis.",
    )?;
    submodule.add_class::<graph::NodePayload>()?;
    submodule.add_class::<graph::EdgePayload>()?;
    submodule.add_class::<graph::NetworkStructure>()?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

/// Registers centrality-related classes for network centrality analysis.
fn register_centrality_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "centrality")?;
    submodule.add(
        "__doc__",
        "Centrality analysis utilities for network structures.",
    )?;
    submodule.add_class::<centrality::CentralityShortestResult>()?;
    submodule.add_class::<centrality::CentralitySimplestResult>()?;
    submodule.add_class::<centrality::CentralitySegmentResult>()?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

/// Registers viewshed-related classes for spatial visibility analysis.
fn register_viewshed_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "viewshed")?;
    submodule.add(
        "__doc__",
        "Viewshed analysis utilities for spatial visibility studies.",
    )?;
    submodule.add_class::<viewshed::Viewshed>()?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}
