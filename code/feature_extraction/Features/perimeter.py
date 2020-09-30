from feature import Feature
from skimage.measure import perimeter


class Perimeter(Feature):

    def __init__(self):
        self._name = 'Perimeter'

    def get_name(self, m_type: int) -> str:
        loc = ['_nucleus', '_cell', '_membrane']
        return self._name + loc[m_type]

    def measure(self, img, m_type: int) -> float:
        mask = img.get_mask(m_type)
        mask_one = mask/mask.max()
        value = perimeter(mask_one)
        value = float(value)
        return value

