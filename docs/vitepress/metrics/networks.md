# cityseer.metrics.networks

Centrality methods

## Network\_Layer

class Network_Layer()### Network\_Layer.\_\_init\_\_

 | <FuncSignature>
 | <pre>
 | __init__(node_uids,
 |          node_data,
 |          edge_data,
 |          node_edge_map,
 |          distances = None,
 |          betas = None,
 |          min_threshold_wt = checks.def_min_thresh_wt)
 | </pre>
 | </FuncSignature>
 | 
 | NODE MAP:
0 - x
1 - y
2 - live

EDGE MAP:
0 - start node
1 - end node
2 - length in metres
3 - sum of angular travel along length
4 - impedance factor
5 - in bearing
6 - out bearing

### Network\_Layer.metrics\_to\_dict

 | <FuncSignature>
 | <pre>
 | metrics_to_dict()
 | </pre>
 | </FuncSignature>
 | 
 | metrics are stored in arrays, this method unpacks per uid

