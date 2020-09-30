from feature import Feature
import numpy as np


class MeanBackgroundIntensity(Feature):

    def __init__(self):
        self._name = 'Mean BG Intensity'

    def get_name(self, m_type: int) -> str:
        loc = ['_nucleus', '_cell', '_membrane']
        return self._name + loc[m_type]

    def measure(self, img, m_type: int) -> float:
        mask = img.get_mask(m_type)
        raw = img.get_raw(False)

        no_mask_coordinates = np.where(mask == 0)
        value = (raw[no_mask_coordinates].sum() / no_mask_coordinates[0].size)
        return value
