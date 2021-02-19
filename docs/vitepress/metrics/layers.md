# Table of Contents

* [cityseer.metrics.layers](#cityseer.metrics.layers)
  * [data\_map\_from\_dict](#cityseer.metrics.layers.data_map_from_dict)
  * [Data\_Layer](#cityseer.metrics.layers.Data_Layer)
    * [compute\_aggregated](#cityseer.metrics.layers.Data_Layer.compute_aggregated)

---
sidebar_label: layers
title: cityseer.metrics.layers
---

<a name="cityseer.metrics.layers.data_map_from_dict"></a>
#### data\_map\_from\_dict

<FuncSignature>

data_map_from_dict(data_dict: dict) -> Tuple[tuple, np.ndarray]

</FuncSignature>

DATA MAP:
0 - x
1 - y
2 - assigned network index - nearest
3 - assigned network index - next-nearest

<a name="cityseer.metrics.layers.Data_Layer"></a>
## Data\_Layer

class Data_Layer()DATA MAP:
0 - x
1 - y
2 - assigned network index - nearest
3 - assigned network index - next-nearest

<a name="cityseer.metrics.layers.Data_Layer.compute_aggregated"></a>
#### Data\_Layer.compute\_aggregated

 | <FuncSignature>
 | 
 | compute_aggregated(landuse_labels = None,
 |                    stats_data_arrs,
 |                   
 |                                                   Tuple[Union[list,
 |                    tuple,
 |                    np.ndarray]],
 |                   
 |                                                   np.ndarray] = None,
 |                    angular = False)
 | 
 | </FuncSignature>
 | 
 | This method provides full access to the underlying diversity.local_landuses method

Try not to duplicate underlying type or signature checks here

