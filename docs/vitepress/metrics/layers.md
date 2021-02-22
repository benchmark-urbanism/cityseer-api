# cityseer.metrics.layers

### data\_map\_from\_dict

<FuncSignature>

data_map_from_dict(data_dict: dict) -> Tuple[tuple, np.ndarray]

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
