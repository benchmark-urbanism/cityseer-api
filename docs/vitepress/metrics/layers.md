# cityseer.metrics.layers

### data\_map\_from\_dict

<FuncSignature>
<pre>
data_map_from_dict(data_dict) -> Tuple[tuple, np.ndarray]
</pre>
</FuncSignature>

DATA MAP:
0 - x
1 - y
2 - assigned network index - nearest
3 - assigned network index - next-nearest

## Data\_Layer

class Data_Layer()DATA MAP:
0 - x
1 - y
2 - assigned network index - nearest
3 - assigned network index - next-nearest

### Data\_Layer.compute\_aggregated

 | <FuncSignature>
 | <pre>
 | compute_aggregated(landuse_labels = None,
 |                    mixed_use_keys = None,
 |                    accessibility_keys = None,
 |                    cl_disparity_wt_matrix = None,
 |                    qs = None,
 |                    stats_keys = None,
 |                    stats_data_arrs = None,
 |                    angular = False)
 | </pre>
 | </FuncSignature>
 |
 | This method provides full access to the underlying diversity.local_landuses method

Try not to duplicate underlying type or signature checks here
