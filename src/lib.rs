extern crate pyo3;
use pyo3::prelude::*;

#[pyfunction]
fn multiply(a: isize, b: isize) -> PyResult<isize> {
    Ok(a * b)
}

#[pymodule]
fn _internal(py: Python, m: &PyModule) -> PyResult<()> {
    
    let rustalgos = PyModule::new(py, "rustalgos")?;
    rustalgos.add_wrapped(wrap_pyfunction!(multiply))?;
    m.add_submodule(rustalgos)?;

    // let sys = PyModule::import(py, "sys")?;
    // sys.getattr("modules")?.set_item("rustalgos", rustalgos)?;
    
    Ok(())
}
