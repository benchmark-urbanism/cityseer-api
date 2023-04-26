use std::collections::HashMap;
use petgraph::prelude::*;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::{StableGraph};
use petgraph::visit::GraphBase;

struct NodePayload {
    idx: String,
    x: f32,
    y: f32,
    live: bool,
}
struct EdgePayload {

}
struct NetworkStructure {
    graph: StableGraph<NodePayload, EdgePayload>,
}
fn prep_network_structure() -> NetworkStructure {
    NetworkStructure {
        graph: StableGraph::<NodePayload, EdgePayload>::new(),
    }
}
impl NetworkStructure {
    fn set_node(&mut self, idx: String, x: f32, y: f32, live: bool) -> NodeIndex {
        let payload = NodePayload{
            idx,
            x,
            y,
            live,
        };
        let node_idx = self.graph.add_node(payload);
        return node_idx
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_graph() {

    }

}