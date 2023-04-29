use pyo3::prelude::*;
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

    // let sys = PyModule::import(py, "sys")?;
    // sys.getattr("modules")?.set_item("rustalgos", rustalgos)?;

    Ok(())
}
