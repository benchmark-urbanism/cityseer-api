use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct DataEntry {
    #[pyo3(get)]
    x: i32,
    #[pyo3(get)]
    y: i32,
    #[pyo3(get)]
    nearest_assign: i32,
    #[pyo3(get)]
    next_nearest_assign: i32,
}
#[pymethods]
impl DataEntry {
    #[new]
    fn new(x: i32, y: i32) -> DataEntry {
        DataEntry {
            x,
            y,
            nearest_assign: -1,
            next_nearest_assign: -1,
        }
    }
    fn xy(&self) -> (i32, i32) {
        (self.x, self.y)
    }
    fn is_assigned(&self) -> bool {
        self.nearest_assign != -1
    }
}

#[pyclass]
pub struct DataMap {
    #[pyo3(get)]
    entries: Vec<DataEntry>,
}
#[pymethods]
impl DataMap {
    #[new]
    fn new() -> DataMap {
        DataMap {
            entries: Vec::new(),
        }
    }
    fn insert(&mut self, x: i32, y: i32) {
        let entry = DataEntry::new(x, y);
        self.entries.push(entry);
    }
    fn set_nearest_assign(&mut self, idx: usize, assign_idx: i32) {
        let entry = self.entries.get_mut(idx);
        if entry.is_some() {
            entry.unwrap().nearest_assign = assign_idx;
        }
    }
    fn set_next_nearest_assign(&mut self, idx: usize, assign_idx: i32) {
        let entry = self.entries.get_mut(idx);
        if entry.is_some() {
            entry.unwrap().next_nearest_assign = assign_idx;
        }
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    fn all_assigned(&self) -> bool {
        for item_entry in self.entries.iter() {
            if item_entry.is_assigned() == false {
                return false;
            }
        }
        true
    }
    fn none_assigned(&self) -> bool {
        for item_entry in self.entries.iter() {
            if item_entry.is_assigned() {
                return false;
            }
        }
        true
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