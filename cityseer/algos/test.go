package main

import "C"

import (
    "math"
)
// NodeDensity calculates node density
//export NodeDensity
func NodeDensity(toDist, toImp, beta float64, cycles int) float64 { return 1 }
// NodeFarness calculates farness
//export NodeFarness
func NodeFarness(toDist, toImp, beta float64, cycles int) float64 { return toDist }
// NodeCycles calculates network cycles
//export NodeCycles
func NodeCycles(toDist, toImp, beta float64, cycles int) int { return cycles }
// NodeHarmonic calculates harmonic closeness
//export NodeHarmonic
func NodeHarmonic(toDist, toImp, beta float64, cycles int) float64 { return 1.0 / toImp }
// NodeBeta calculates the "gravity" index
//export NodeBeta
func NodeBeta(toDist, toImp, beta float64, cycles int) float64 {
    return float64(math.Exp(float64(beta * toDist)))
}
// NodeHarmonicAngular calculates angular harmonic closeness
//export NodeHarmonicAngular
func NodeHarmonicAngular(toDist, toImp, beta float64, cycles int) float64 {
    var a float64 = 1 + (toImp / 180)
    return 1.0 / a
}
// NodeBetweenness calculates node betweenness
//export NodeBetweenness
func NodeBetweenness(toDist float64, beta float64) float64 { return 1 }
// NodeBetweennessBeta calculates node betweenness weighted by beta
//export NodeBetweennessBeta
func NodeBetweennessBeta(toDist float64, beta float64) float64 {
    /*
    distance is based on distance between from and to vertices
    thus potential spatial impedance via between vertex
    */
    return float64(math.Exp(float64(beta * toDist)))
}
// Node contains information about a graph vertex
//export Node
type Node struct {
    // x coordinate
    x float64
    // y coordinate
    y float64
    // whether the node is considered live (i.e. not inside the extended buffer zone)
    live bool
}
// newNode instances a new node
func newNode(x, y float64, live bool) *Node {
    n := Node { x: x, y: y, live: live }
    return &n
}
// Edge contains information about a graph edge
//export Edge
type Edge struct {
    // the start node for the edge
    startNodeIdx int
    // the end node for the edge
    endNodeIdx int
    // the length of the edge in metres
    length float64
    // the sum of angular change along the edge's length
    angles float64
    // the impedance factor of the edge
    impedanceFactor float64
    // the angular bearing when entering an edge
    inBearing float64
    // the angular bearing when exiting an edge
    outBearing float64
}
// newEdge instances a new edge
func newEdge(start, end int, length, angles, impedanceFactor, inBearing, outBearing float64) *Edge {
    e := Edge {
        startNodeIdx: start,
        endNodeIdx: end,
        length: length,
        impedanceFactor: impedanceFactor,
        inBearing: inBearing,
        outBearing: outBearing }
    return &e
}
// NodeEdges maps nodes to out edges
//export NodeEdges
type NodeEdges struct {
     edgeList []int
}
// newNodeEdges instances a new NodeEdges
func newNodeEdges(edgeList []int) *NodeEdges {
    ne := NodeEdges { edgeList: edgeList }
    return &ne
}
// Graph contains nodes and edges that together define a graph.
//export Graph
type Graph struct {
    nodes []Node
    edges []Edge
    nodeEdges []NodeEdges
}
// newGraph instances a new Graph
func newGraph(nodes []Node, edges []Edge, nodeEdges []NodeEdges) *Graph {
    g := Graph{ nodes: nodes, edges: edges, nodeEdges: nodeEdges}
    return &g
}
// DTree contains information of all dijkstra shortest paths from a source node.
// It is returned by the ShortestPathTree function.
//export DTree
type DTree struct {
    // the source node from which the source has been instantiated
    srcIdx int
    // the maximum distance searched
    maxDist float64
    // visited nodes
    visitedNodes []bool
    // cycles
    cycles []bool
    // predecessors
    pred []int
    // distance from the source node
    dist []float64
    // impedance from the source node
    imps []float64
    // for any to_idx, the origin segment of the shortest path 
    originSeg []int
    // last segments - for any to_idx, the last segment of the shortest path
    lastSeg []int
    // in bearings - for computing live angular changes
    inBearings []float64
    // out bearings - for computing live angular changes
    outBearings []float64
    // for keeping track of visited edges
    visitedEdges []bool
}
// newDTree instances a new DTree
func newDTree(g Graph, srcIdx int, maxDist float64) *DTree {
    nodesLength := len(g.nodes)
    edgesLength := len(g.edges)
    // instantiate
    dT := DTree{ srcIdx: srcIdx, maxDist: maxDist }
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
func ShortestPathTree(g Graph, srcIdx int, maxDist float64, angular bool) *DTree {
    dT := newDTree(g, srcIdx, maxDist)
    // the starting node's impedance and distance will be zero
    dT.imps[srcIdx] = 0
    dT.dist[srcIdx] = 0
    // active keeps track of the indices of active nodes
    active := make([]int, 0, len(g.nodes)) // include max capacity
    // add the srcIdx to start
    active = append(active, srcIdx)
    // continues until all nodes within the max distance have been processed
    for len(active) > 0 {
        // iterate the currently active indices and find the one with the smallest distance
        var activeNodeIdx int // selected Graph index
        minImp := math.Inf(1)
        var minArrIdx int // minimum active slice index
        for idx, nodeIdx := range active {
            if dT.imps[nodeIdx] < minImp {
                minImp = dT.imps[nodeIdx]
                activeNodeIdx = nodeIdx
                minArrIdx = idx
            }
        }
        // the currently processed node can now be removed from the active list and added to the processed list
        sliceRemoveAt(active, minArrIdx)
        // add to active node
        dT.visitedNodes[activeNodeIdx] = true
        // iterate the node's neighbours
        for _, edgeIdx := range g.nodeEdges[activeNodeIdx].edgeList {
            // find the corresponding out edge
            outEdge := g.edges[edgeIdx]
            // find the corresponding neighbour
            nbNodeIdx := outEdge.endNodeIdx
            // don't follow self-loops
            if nbNodeIdx == activeNodeIdx {
                continue
            }
            // don't visit predecessor nodes - otherwise successive nodes revisit out-edges to previous (neighbour) nodes
            if nbNodeIdx == dT.pred[activeNodeIdx] {
                continue
            }
            // DO ANGULAR BEFORE CYCLES, AND CYCLES BEFORE DISTANCE THRESHOLD
            // it is necessary to check for angular sidestepping if using angular impedances on a dual graph
            // only do this for angular graphs, and if the nb node has already been discovered
            if angular == true && activeNodeIdx != srcIdx && !(dT.pred[nbNodeIdx] == -1) {
                priorMatch := false
                // get the active node's predecessor
                predNodeIdx := dT.pred[activeNodeIdx]
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
                if dT.dist[nbNodeIdx] > dT.dist[activeNodeIdx] {
                    dT.cycles[nbNodeIdx] = true
                } else {
                    dT.cycles[activeNodeIdx] = true
                }
            }
            // impedance and distance is previous plus new
            var impedance float64
            if angular != true {
                impedance = dT.imps[activeNodeIdx] + outEdge.length * outEdge.impedanceFactor
            } else {
                // angular impedance include two parts:
                // A - turn from prior simplest-path route segment
                // B - angular change across current segment
                turn := 0.0
                if activeNodeIdx != srcIdx {
                    turn = math.Abs(float64(
                            int(outEdge.inBearing - dT.outBearings[activeNodeIdx] + 180) % 360 - 180))
                    }
                impedance = dT.imps[activeNodeIdx] + (turn + outEdge.angles) * outEdge.impedanceFactor
            }
            distance := dT.dist[activeNodeIdx] + outEdge.length
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
                dT.pred[nbNodeIdx] = activeNodeIdx
                dT.outBearings[nbNodeIdx] = outEdge.outBearing
                // chain through origin segs - identifies which segment a particular shortest path originated from
                if activeNodeIdx == srcIdx {
                    dT.originSeg[nbNodeIdx] = edgeIdx
                } else {
                    dT.originSeg[nbNodeIdx] = dT.originSeg[activeNodeIdx]
                }
                // keep track of last seg
                dT.lastSeg[nbNodeIdx] = edgeIdx
            }
        }
    }
    // the returned active edges contain activated edges, but only in direction of shortest-path discovery
    return dT
}

func main() {}