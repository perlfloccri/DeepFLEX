from feature import Feature
from skimage.measure import perimeter, regionprops, label
import math


class Roundness(Feature):

    def __init__(self):
        self._name = 'Roundness'

    def get_name(self, m_type: int) -> str:
        loc = ['_nucleus', '_cell', '_membrane']
        return self._name + loc[m_type]

    def measure(self, img, m_type: int) -> float:
        mask = img.get_mask(m_type)
        mask_one = mask/mask.max()
        label_img = label(mask_one)
        region = regionprops(label_img)
        area = region[0].area
        per = perimeter(mask_one)
        value = (4*math.pi*area)/(per**2)
        return value
