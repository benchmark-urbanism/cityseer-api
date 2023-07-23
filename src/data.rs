use crate::common::Coord;
use pyo3::prelude::*;
use pyo3::types::PyString;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone)]
pub struct DataEntry {
    #[pyo3(get)]
    data_key: String,
    #[pyo3(get)]
    coord: Coord,
    #[pyo3(get)]
    nearest_assign: Option<usize>,
    #[pyo3(get)]
    next_nearest_assign: Option<usize>,
    #[pyo3(get)]
    data_id: String,
}
#[pymethods]
impl DataEntry {
    #[new]
    fn new(data_key: String, x: f32, y: f32, data_id: String) -> DataEntry {
        DataEntry {
            data_key,
            coord: Coord::new(x, y),
            nearest_assign: None,
            next_nearest_assign: None,
            data_id,
        }
    }
    fn is_assigned(&self) -> bool {
        !self.nearest_assign.is_none()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct DataMap {
    #[pyo3(get)]
    entries: HashMap<String, DataEntry>,
}
#[pymethods]
impl DataMap {
    #[new]
    fn new() -> DataMap {
        DataMap {
            entries: HashMap::new(),
        }
    }
    fn insert(&mut self, data_key: String, x: f32, y: f32, data_id: String) {
        self.entries
            .insert(data_key.clone(), DataEntry::new(data_key, x, y, data_id));
    }
    fn entry_keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }
    fn get_entry(&self, data_key: &str) -> Option<DataEntry> {
        let entry = self.entries.get(data_key);
        if entry.is_some() {
            return Some(entry.unwrap().clone());
        }
        None
    }
    fn get_data_coord(&self, data_key: &str) -> Option<Coord> {
        let entry = self.entries.get(data_key);
        if entry.is_some() {
            return Some(entry.unwrap().coord);
        }
        None
    }
    #[getter]
    fn count(&self) -> usize {
        self.entries.len()
    }
    #[getter]
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    fn all_assigned(&self) -> bool {
        for (_item_key, item_val) in &self.entries {
            if item_val.is_assigned() == false {
                return false;
            }
        }
        true
    }
    fn none_assigned(&self) -> bool {
        for (_item_key, item_val) in &self.entries {
            if item_val.is_assigned() {
                return false;
            }
        }
        true
    }
    fn set_nearest_assign(&mut self, data_key: String, assign_idx: usize) {
        if let Some(entry) = self.entries.get_mut(&data_key) {
            entry.nearest_assign = Some(assign_idx);
        }
    }
    fn set_next_nearest_assign(&mut self, data_key: String, assign_idx: usize) {
        if let Some(entry) = self.entries.get_mut(&data_key) {
            entry.next_nearest_assign = Some(assign_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_entry() {
        let entry = DataEntry::new(1, 2);
        assert_eq!(entry.xy(), (1, 2));
        assert_eq!(entry.is_assigned(), false);

        let mut entry2 = DataEntry::new(3, 4);
        assert_eq!(entry2.is_assigned(), false);
        entry2.nearest_assign = 1;
        assert_eq!(entry2.is_assigned(), true);
    }
}
