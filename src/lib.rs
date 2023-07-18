use pyo3::prelude::*;
mod common;
mod data_structure;
mod graph_structure;

#[pyfunction]
fn multiply(a: isize, b: isize) -> PyResult<isize> {
    Ok(a * b)
}

#[pymodule]
fn rustalgos(_py: Python, m: &PyModule) -> PyResult<()> {
    // let rustalgos = PyModule::new(py, "rustalgos")?;
    m.add_wrapped(wrap_pyfunction!(multiply))?;
    // m.add_submodule(rustalgos)?;
    m.add_class::<data_structure::DataEntry>()?;
    m.add_class::<data_structure::DataMap>()?;
    m.add_class::<graph_structure::NodePayload>()?;
    m.add_class::<graph_structure::EdgePayload>()?;
    m.add_class::<graph_structure::NetworkStructure>()?;
    m.add_function(wrap_pyfunction!(common::check_numerical_data, m)?)?;
    m.add_function(wrap_pyfunction!(common::distances_from_betas, m)?)?;
    m.add_function(wrap_pyfunction!(common::betas_from_distances, m)?)?;
    m.add_function(wrap_pyfunction!(common::pair_distances_and_betas, m)?)?;
    m.add_function(wrap_pyfunction!(common::avg_distances_for_betas, m)?)?;
    m.add_function(wrap_pyfunction!(common::clip_wts_curve, m)?)?;
    m.add_function(wrap_pyfunction!(common::clipped_beta_wt, m)?)?;
    // let sys = PyModule::import(py, "sys")?;
    // sys.getattr("modules")?.set_item("rustalgos", rustalgos)?;
    Ok(())
}
