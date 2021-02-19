# Table of Contents

* [cityseer.tools.mock](#cityseer.tools.mock)
  * [mock\_graph](#cityseer.tools.mock.mock_graph)
  * [diamond\_graph](#cityseer.tools.mock.diamond_graph)

---
sidebar_label: mock
title: cityseer.tools.mock
---

Generate a graph for testing and documentation purposes.

<a name="cityseer.tools.mock.mock_graph"></a>
#### mock\_graph

<FuncSignature>

mock_graph(wgs84_coords = False) -> nx.MultiGraph

</FuncSignature>

Prepares a Tutte graph per https://en.wikipedia.org/wiki/Tutte_graph
:return: NetworkX graph

<a name="cityseer.tools.mock.diamond_graph"></a>
#### diamond\_graph

<FuncSignature>

@pytest.fixture
diamond_graph()

</FuncSignature>

For manual checks of all node and segmentised methods
3
/ \
/   \
/  a  \
1-------2
\  |  /
\ |b/ c
\|/
0
a = 100m = 2 * 50m
b = 86.60254m
c = 100m
all inner angles = 60º

