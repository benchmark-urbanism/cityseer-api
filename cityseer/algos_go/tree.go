// DTree contains information of all dijkstra shortest paths from a source node.
// It is returned by the ShortestPathTree function.
//export DTree
type DTree struct {
    // the source node from which the source has been instantiated
    srcKey string
    // the maximum distance searched
    maxDist float64
    // visited nodes and edges
    visitedNodes, visitedEdges []bool
    // cycles
    cycles []bool
    // predecessors
    pred []int
    // distances and impedances from the source node
    dist, imps []float64
    // for any to_idx, the origin and last segments of the shortest path 
    originSeg, lastSeg []int
    // in bearings and out bearings for computing live angular changes
    inBearings, outBearings []float64
}
// newDTree instances a new DTree
func newDTree(g Graph, srcKey string, maxDist float64) *DTree {
    nodesLength := len(g.nodes)
    edgesLength := len(g.edges)
    // instantiate
    dT := DTree{ srcKey: srcKey, maxDist: maxDist }
    dT.visitedNodes = make([]bool, nodesLength) // false
    dT.cycles = make([]bool, nodesLength) // 0
    dT.pred = make([]int, nodesLength) // -1
    for idx := range dT.pred {
        dT.pred[idx] = -1
    }
    dT.dist = make([]float64, nodesLength) // Inf
    for idx := range dT.dist {
        dT.dist[idx] = math.Inf(1)
    }
    copy(dT.imps, dT.dist) // Inf
    copy(dT.originSeg, dT.pred) // -1
    copy(dT.lastSeg, dT.pred) // -1
    dT.inBearings = make([]float64, nodesLength) // Nan
    for idx := range dT.inBearings {
        dT.inBearings[idx] = math.NaN()
    }
    copy(dT.outBearings, dT.inBearings) // Nan
    dT.visitedEdges = make([]bool, edgesLength) // false
    return &dT
}

// sliceRemoveAt is a convenience function for removing items from a slice
func sliceRemoveAt(s []int, idx int) []int {
    s[idx] = s[len(s) - 1] // swap the removed item with the last item
    return s[:len(s) - 1] // return sans the last item
}
// ShortestPathTree returns the shortest paths to all nodes reachable from a source node with a max distance.
// Returns a DTree with all reachable nodes, their distances, their predecessors.
// Angular flag triggers checks for sidestepping per angular impedance sidesteps.
func ShortestPathTree(g Graph, srcKey string, maxDist float64, angular bool) *DTree {
    dT := newDTree(g, srcKey, maxDist)
    // the starting node's impedance and distance will be zero
    // TODO: hmm, major structural ramifications - string maps vs. int arrays
    dT.imps[srcKey] = 0
    dT.dist[srcKey] = 0
    // active keeps track of the indices of active nodes
    // TODO: this could be a map
    active := make([]string, 0, len(g.nodes)) // include max capacity
    // add the srcKey to start
    active = append(active, srcKey)
    // continues until all nodes within the max distance have been processed
    for len(active) > 0 {
        // iterate the currently active indices and find the one with the smallest distance
        var activeNodeKey string // selected Graph index
        minImp := math.Inf(1)
        var minArrIdx int // minimum active slice index
        for idx, nodeIdx := range active {
            if dT.imps[nodeIdx] < minImp {
                minImp = dT.imps[nodeIdx]
                activeNodeKey = nodeIdx
                minArrIdx = idx
            }
        }
        // the currently processed node can now be removed from the active list and added to the processed list
        sliceRemoveAt(active, minArrIdx)
        // add to active node
        dT.visitedNodes[activeNodeKey] = true
        // iterate the node's neighbours
        for _, edgeIdx := range g.nodes[activeNodeKey].edges {
            // find the corresponding out edge
            outEdge := g.edges[edgeIdx]
            // find the corresponding neighbour
            nbNodeIdx := outEdge.endNodeIdx
            // don't follow self-loops
            if nbNodeIdx == activeNodeKey {
                continue
            }
            // don't visit predecessor nodes - otherwise successive nodes revisit out-edges to previous (neighbour) nodes
            if nbNodeIdx == dT.pred[activeNodeKey] {
                continue
            }
            // DO ANGULAR BEFORE CYCLES, AND CYCLES BEFORE DISTANCE THRESHOLD
            // it is necessary to check for angular sidestepping if using angular impedances on a dual graph
            // only do this for angular graphs, and if the nb node has already been discovered
            if angular == true && activeNodeKey != srcKey && !(dT.pred[nbNodeIdx] == -1) {
                priorMatch := false
                // get the active node's predecessor
                predNodeIdx := dT.pred[activeNodeKey]
                // check that the new neighbour was not directly accessible from the predecessor's set of neighbours
                for _, predEdgeIdx := range(g.nodeEdges[predNodeIdx].edgeList) {
                    // iterate end nodes corresponding to edges accessible from the predecessor node
                    predEndNode := g.edges[predEdgeIdx].endNodeIdx
                    // check that the previous node's neighbour's node is not equal to the currently new neighbour node
                    // if so, the new neighbour was previously accessible
                    if predEndNode == nbNodeIdx {
                        priorMatch = true
                        break
                    }
                }
                // continue if prior match was found
                if priorMatch == true {
                    continue
                }
            }
            // if a neighbouring node has already been discovered, then it is a cycle
            // do before distance cutoff because this node and the neighbour can respectively be within max distance
            // in some cases all distances are run at once, so keep behaviour consistent by
            // designating the farthest node (but via the shortest distance) as the cycle node
            if dT.pred[nbNodeIdx] != -1 {
                // set the farthest location to True - nb node vs active node
                if dT.dist[nbNodeIdx] > dT.dist[activeNodeKey] {
                    dT.cycles[nbNodeIdx] = true
                } else {
                    dT.cycles[activeNodeKey] = true
                }
            }
            // impedance and distance is previous plus new
            var impedance float64
            if angular != true {
                impedance = dT.imps[activeNodeKey] + outEdge.length * outEdge.impedanceFactor
            } else {
                // angular impedance include two parts:
                // A - turn from prior simplest-path route segment
                // B - angular change across current segment
                turn := 0.0
                if activeNodeKey != srcKey {
                    turn = math.Abs(float64(
                            int(outEdge.inBearing - dT.outBearings[activeNodeKey] + 180) % 360 - 180))
                    }
                impedance = dT.imps[activeNodeKey] + (turn + outEdge.angles) * outEdge.impedanceFactor
            }
            distance := dT.dist[activeNodeKey] + outEdge.length
            // add the neighbour to active if undiscovered but only if less than max threshold
            if dT.pred[nbNodeIdx] != -1 && distance <= maxDist {
                active = append(active, nbNodeIdx)
            }
            // only add edge to active if the neighbour node has not been processed previously
            // (i.e. single direction only)
            if dT.visitedNodes[nbNodeIdx] == false {
                dT.visitedEdges[edgeIdx] = true
            }
            // if impedance less than prior, update
            // this will also happen for the first nodes that overshoot the boundary
            // they will not be explored further because they have not been added to active
            if impedance < dT.imps[nbNodeIdx] {
                dT.imps[nbNodeIdx] = impedance
                dT.dist[nbNodeIdx] = distance
                dT.pred[nbNodeIdx] = activeNodeKey
                dT.outBearings[nbNodeIdx] = outEdge.outBearing
                // chain through origin segs - identifies which segment a particular shortest path originated from
                if activeNodeKey == srcKey {
                    dT.originSeg[nbNodeIdx] = edgeIdx
                } else {
                    dT.originSeg[nbNodeIdx] = dT.originSeg[activeNodeKey]
                }
                // keep track of last seg
                dT.lastSeg[nbNodeIdx] = edgeIdx
            }
        }
    }
    // the returned active edges contain activated edges, but only in direction of shortest-path discovery
    return dT
}
