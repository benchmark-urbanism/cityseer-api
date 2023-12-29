use pyo3::prelude::*;
mod centrality;
mod common;
mod data;
mod diversity;
mod graph;
mod viewshed;

#[pymodule]
fn rustalgos(_py: Python, m: &PyModule) -> PyResult<()> {
    // let rustalgos = PyModule::new(py, "rustalgos")?;
    // m.add_submodule(rustalgos)?;
    m.add_class::<common::Coord>()?;
    m.add_function(wrap_pyfunction!(common::calculate_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(common::calculate_rotation_smallest, m)?)?;
    m.add_function(wrap_pyfunction!(common::check_numerical_data, m)?)?;
    m.add_function(wrap_pyfunction!(common::distances_from_betas, m)?)?;
    m.add_function(wrap_pyfunction!(common::betas_from_distances, m)?)?;
    m.add_function(wrap_pyfunction!(common::pair_distances_and_betas, m)?)?;
    m.add_function(wrap_pyfunction!(common::avg_distances_for_betas, m)?)?;
    m.add_function(wrap_pyfunction!(common::clip_wts_curve, m)?)?;
    m.add_function(wrap_pyfunction!(common::clipped_beta_wt, m)?)?;
    // DATA
    m.add_class::<data::DataEntry>()?;
    m.add_class::<data::DataMap>()?;
    m.add_class::<data::AccessibilityResult>()?;
    m.add_class::<data::MixedUsesResult>()?;
    m.add_class::<data::StatsResult>()?;
    m.add_function(wrap_pyfunction!(diversity::hill_diversity, m)?)?;
    m.add_function(wrap_pyfunction!(
        diversity::hill_diversity_branch_distance_wt,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        diversity::hill_diversity_pairwise_distance_wt,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(diversity::gini_simpson_diversity, m)?)?;
    m.add_function(wrap_pyfunction!(diversity::shannon_diversity, m)?)?;
    m.add_function(wrap_pyfunction!(diversity::raos_quadratic_diversity, m)?)?;
    // GRAPH
    m.add_class::<graph::NodePayload>()?;
    m.add_class::<graph::EdgePayload>()?;
    m.add_class::<graph::NetworkStructure>()?;
    m.add_class::<centrality::CentralityShortestResult>()?;
    m.add_class::<centrality::CentralitySimplestResult>()?;
    m.add_class::<centrality::CentralitySegmentResult>()?;
    // VIEWSHED
    m.add_class::<viewshed::Viewshed>()?;
    // let sys = PyModule::import(py, "sys")?;
    // sys.getattr("modules")?.set_item("rustalgos", rustalgos)?;
    Ok(())
}
