# Table of Contents

* [cityseer.metrics.networks](#cityseer.metrics.networks)
  * [Network\_Layer](#cityseer.metrics.networks.Network_Layer)
    * [\_\_init\_\_](#cityseer.metrics.networks.Network_Layer.__init__)
    * [metrics\_to\_dict](#cityseer.metrics.networks.Network_Layer.metrics_to_dict)

---
sidebar_label: networks
title: cityseer.metrics.networks
---

Centrality methods

<a name="cityseer.metrics.networks.Network_Layer"></a>
## Network\_Layer

class Network_Layer()<a name="cityseer.metrics.networks.Network_Layer.__init__"></a>
#### Network\_Layer.\_\_init\_\_

 | <FuncSignature>
 | 
 | __init__(node_uids = checks.def_min_thresh_wt)
 | 
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

<a name="cityseer.metrics.networks.Network_Layer.metrics_to_dict"></a>
#### Network\_Layer.metrics\_to\_dict

 | <FuncSignature>
 | 
 | metrics_to_dict()
 | 
 | </FuncSignature>
 | 
 | metrics are stored in arrays, this method unpacks per uid

