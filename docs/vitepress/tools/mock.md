# cityseer.tools.mock

Generate a graph for testing and documentation purposes.

### mock\_graph

<FuncSignature>
<pre>
mock_graph(wgs84_coords = False) -> nx.MultiGraph
</pre>
</FuncSignature>

Prepares a Tutte graph per https://en.wikipedia.org/wiki/Tutte_graph
:return: NetworkX graph

### diamond\_graph

<FuncSignature>
<pre>
@pytest.fixture
diamond_graph()
</pre>
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

