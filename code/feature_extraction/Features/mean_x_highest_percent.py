from feature import Feature
import numpy as np
import math

class MeanXHighestPercent(Feature):

    def __init__(self):
        self._name = 'MeanXHighestPercent'

    def get_name(self, m_type: int) -> str:
        loc = ['_nucleus', '_cell', '_membrane']
        return self._name + loc[m_type]

    def measure(self, img, m_type: int) -> float:
        mask = img.get_mask(m_type)
        raw = img.get_raw(False)

        x = 0.2
        mask_coordinates = np.where(mask != 0)
        if raw[mask_coordinates].sum() == 0:
            value = 0
        else:
            foreground_pixel = mask_coordinates[0].size
            start_index = math.floor(foreground_pixel * (1 - x))
            temp = sorted(raw[mask_coordinates])
            value = np.mean(temp[start_index:])
        return value

