// go build -o test.so -buildmode=c-shared test.go
package main

import (
    "math";
    "fmt"
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
type Node struct {
    // x, y coordinates
    x, y float64
    // whether the node is considered live (i.e. not inside the extended buffer zone)
    live bool
    // out edges
    edges []string
}
// newNode instances a new node
func newNode(x, y float64, live bool, edges []string) *Node {
    n := Node { x: x, y: y, live: live, edges: edges }
    return &n
}
// Edge contains information about a graph edge
type Edge struct {
    // the start and end nodes for the edge
    startNodeIdx, endNodeIdx string
    // the length of the edge in metres
    // the sum of angular change along the edge's length
    length, angle float64
    // the impedance factor of the edge
    impedanceFactor float64
    // the angular in and out bearings for an edge
    inBearing, outBearing float64
}
// newEdge instances a new edge
func newEdge(start, end string, length, angle, impedanceFactor, inBearing, outBearing float64) Edge {
    e := Edge {
        startNodeIdx: start,
        endNodeIdx: end,
        length: length,
        angle: angle,
        impedanceFactor: impedanceFactor,
        inBearing: inBearing,
        outBearing: outBearing }
    return e
}
// Graph contains nodes and edges that together define a graph.
type Graph struct {
    nodes map[string]*Node
    edges map[string]Edge
}
// newGraph instances a new Graph
func newGraph() Graph {
    g := Graph{ nodes: make(map[string]*Node), edges: make(map[string]Edge) }
    return g
}
var g Graph
func MakeGraph(edgeA Edge) {
    g = newGraph()
    g.nodes["n1"] = newNode(100, 100, true, []string{"e1"})
    g.nodes["n2"] = newNode(100, 200, true, []string{"e1"})
    g.edges["e1"] = edgeA
}
//export PrintGraph
func PrintGraph() {
    fmt.Println(g)
    fmt.Println(g.edges["e1"])
}
//export Test
func Test(e Edge) {
    fmt.Println("boo")
}
// main
func main() {
    MakeGraph(newEdge("n1", "n2", 100.0, 0, 1, 0, 0))
    PrintGraph()
}