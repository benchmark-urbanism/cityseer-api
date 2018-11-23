'''

'''
import logging
from typing import Tuple
from tqdm import tqdm
import utm
import numpy as np
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dict_wgs_to_utm(data_dict:dict) -> dict:

    if not isinstance(data_dict, dict):
        raise TypeError('This method requires dictionary object.')

    logger.info('Converting data dictionary from WGS to UTM.')
    data_dict_copy = data_dict.copy()

    logger.info('Processing node x, y coordinates.')
    for k, v in tqdm(data_dict_copy.items()):
        # x coordinate
        if 'x' not in v:
            raise AttributeError(f'Encountered node missing "x" coordinate attribute at data dictionary key {k}.')
        x = v['x']
        # y coordinate
        if 'y' not in v:
            raise AttributeError(f'Encountered node missing "y" coordinate attribute at data dictionary key {k}.')
        y = v['y']
        # check for unintentional use of conversion
        if x > 180 or y > 90:
            raise AttributeError('x, y coordinates exceed WGS bounds. Please check your coordinate system.')
        # remember - accepts and returns in y, x order
        y, x = utm.from_latlon(y, x)[:2]
        # write back to graph
        data_dict_copy[k]['x'] = x
        data_dict_copy[k]['y'] = y

    return data_dict_copy


def dict_to_data_map(data_dict:dict) -> Tuple[tuple, np.ndarray, tuple]:

    if not isinstance(data_dict, dict):
        raise TypeError('This method requires dictionary object.')

    # extract data classes, if present, and convert to ints
    logger.info('Extracting data classes, if present')
    classes_raw = set([v['class'] for v in data_dict.values() if 'class' in v])
    # use sklearn's label encoder
    le = LabelEncoder()
    le.fit(list(classes_raw))
    # map the int encodings to the respective classes
    classes_raw = le.inverse_transform(range(len(classes_raw)))

    data_labels = []
    data_map = np.full((len(data_dict), 6), np.nan)
    for i, (k, v) in enumerate(data_dict.items()):
        # set key to data labels
        data_labels.append(k)
        # DATA MAP INDEX POSITION 0 = x coordinate
        if 'x' not in v:
            raise AttributeError(f'Encountered entry missing "x" coordinate attribute at index {i}.')
        data_map[i][0] = v['x']
        # DATA MAP INDEX POSITION 1 = y coordinate
        if 'y' not in v:
            raise AttributeError(f'Encountered entry missing "y" coordinate attribute at index {i}.')
        data_map[i][1] = v['y']
        # DATA MAP INDEX POSITION 2 = live or not
        if 'live' in v:
            data_map[i][2] = v['live']
        else:
            data_map[i][2] = True
        # DATA MAP INDEX POSITION 3 = optional data class - leave as np.nan if not present
        if 'class' in v:
            data_map[i][3] = le.transform([v['class']])[0]
        # DATA MAP INDEX POSITION 4 = assigned network index - leave as default np.nan
        # pass
        # DATA MAP INDEX POSITION 5 = distance from assigned network index - leave as default np.nan
        # pass

    return tuple(data_labels), data_map, tuple(classes_raw)
