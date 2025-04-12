use pyo3::prelude::*;

// Module imports
mod centrality;
mod common;
mod data;
mod diversity;
mod graph;
mod viewshed;

/// Cityseer API implementation in Rust for performance-critical algorithms.
/// Exposes network, centrality, and diversity computation functions to Python.
#[pymodule]
fn rustalgos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes and functions
    m.add_class::<common::Coord>()?;
    m.add_function(wrap_pyfunction!(common::calculate_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(common::calculate_rotation_smallest, m)?)?;
    m.add_function(wrap_pyfunction!(common::check_numerical_data, m)?)?;
    m.add_function(wrap_pyfunction!(common::distances_from_betas, m)?)?;
    m.add_function(wrap_pyfunction!(common::betas_from_distances, m)?)?;
    m.add_function(wrap_pyfunction!(common::distances_from_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(common::seconds_from_distances, m)?)?;
    m.add_function(wrap_pyfunction!(common::pair_distances_betas_time, m)?)?;
    m.add_function(wrap_pyfunction!(common::avg_distances_for_betas, m)?)?;
    m.add_function(wrap_pyfunction!(common::clip_wts_curve, m)?)?;
    m.add_function(wrap_pyfunction!(common::clipped_beta_wt, m)?)?;

    // Register modules
    register_data_module(m)?;
    register_diversity_module(m)?;
    register_graph_module(m)?;
    register_centrality_module(m)?;
    register_viewshed_module(m)?;

    m.add(
        "__doc__",
        "Cityseer high-performance algorithms implemented in Rust.",
    )?;

    Ok(())
}

/// Registers data-related classes.
///
/// This module provides data structures for managing and analyzing spatial data.
fn register_data_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let data_module = PyModule::new(m.py(), "data")?;

    // Add module documentation
    data_module.add(
        "__doc__",
        "Data structures and utilities for spatial data analysis.",
    )?;

    // Register classes
    data_module.add_class::<data::DataEntry>()?;
    data_module.add_class::<data::DataMap>()?;
    data_module.add_class::<data::AccessibilityResult>()?;
    data_module.add_class::<data::MixedUsesResult>()?;
    data_module.add_class::<data::StatsResult>()?;

    // Add the submodule to the parent module
    m.add_submodule(&data_module)?;

    Ok(())
}

/// Registers diversity-related functions.
///
/// This module provides functions for calculating various diversity metrics.
fn register_diversity_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let diversity_module = PyModule::new(m.py(), "diversity")?;

    // Add module documentation
    diversity_module.add(
        "__doc__",
        "Functions for calculating diversity metrics in spatial analysis.",
    )?;

    // Register functions
    diversity_module.add_function(wrap_pyfunction!(
        diversity::hill_diversity,
        &diversity_module
    )?)?;
    diversity_module.add_function(wrap_pyfunction!(
        diversity::hill_diversity_branch_distance_wt,
        &diversity_module
    )?)?;
    diversity_module.add_function(wrap_pyfunction!(
        diversity::hill_diversity_pairwise_distance_wt,
        &diversity_module
    )?)?;
    diversity_module.add_function(wrap_pyfunction!(
        diversity::gini_simpson_diversity,
        &diversity_module
    )?)?;
    diversity_module.add_function(wrap_pyfunction!(
        diversity::shannon_diversity,
        &diversity_module
    )?)?;
    diversity_module.add_function(wrap_pyfunction!(
        diversity::raos_quadratic_diversity,
        &diversity_module
    )?)?;

    // Add the submodule to the parent module
    m.add_submodule(&diversity_module)?;

    Ok(())
}

/// Registers graph-related classes.
///
/// This module provides data structures for representing and analyzing graphs.
fn register_graph_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let graph_module = PyModule::new(m.py(), "graph")?;

    // Add module documentation
    graph_module.add(
        "__doc__",
        "Graph data structures and utilities for network analysis.",
    )?;

    // Register classes
    graph_module.add_class::<graph::NodePayload>()?;
    graph_module.add_class::<graph::EdgePayload>()?;
    graph_module.add_class::<graph::NetworkStructure>()?;

    // Add the submodule to the parent module
    m.add_submodule(&graph_module)?;

    Ok(())
}

/// Registers centrality-related classes.
///
/// This module provides data structures for centrality analysis in networks.
fn register_centrality_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let centrality_module = PyModule::new(m.py(), "centrality")?;

    // Add module documentation
    centrality_module.add(
        "__doc__",
        "Centrality analysis utilities for network structures.",
    )?;

    // Register classes
    centrality_module.add_class::<centrality::CentralityShortestResult>()?;
    centrality_module.add_class::<centrality::CentralitySimplestResult>()?;
    centrality_module.add_class::<centrality::CentralitySegmentResult>()?;

    // Add the submodule to the parent module
    m.add_submodule(&centrality_module)?;

    Ok(())
}

/// Registers viewshed-related classes.
///
/// This module provides data structures for viewshed analysis.
fn register_viewshed_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let viewshed_module = PyModule::new(m.py(), "viewshed")?;

    // Add module documentation
    viewshed_module.add(
        "__doc__",
        "Viewshed analysis utilities for spatial visibility studies.",
    )?;

    // Register classes
    viewshed_module.add_class::<viewshed::Viewshed>()?;

    // Add the submodule to the parent module
    m.add_submodule(&viewshed_module)?;

    Ok(())
}
