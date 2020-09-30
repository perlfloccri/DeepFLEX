import numpy as np
from Features.mean_intensity import MeanIntensity
from Features.mean_background_intensity import MeanBackgroundIntensity
from Features.total_intensity import TotalIntensity
from Features.size import Size
from Features.solidity import Solidity
from Features.perimeter import Perimeter
from Features.roundness import Roundness
from Features.mean_x_highest_percent import MeanXHighestPercent


class Image(object):

    def __init__(self, raw: np.ndarray, mask: np.ndarray, marker: str, cytoplasm: bool, single_cell: bool):

        self._raw = raw
        self._cytoplasm = cytoplasm
        self._mask = mask
        self._marker = marker
        self._single_cell = single_cell

        self._intensity_features = []
        self._morphology_features = []

        if not single_cell:
            self._intensity_features.append(MeanIntensity())
            self._intensity_features.append(MeanBackgroundIntensity())
            self._intensity_features.append(TotalIntensity())

            self._morphology_features.append(Size())

        else:
            self._intensity_features.append(MeanIntensity())
            self._intensity_features.append(TotalIntensity())
            self._intensity_features.append(MeanXHighestPercent())

            self._morphology_features.append(Size())
            self._morphology_features.append(Solidity())
            self._morphology_features.append(Perimeter())
            self._morphology_features.append(Roundness())

    def get_raw(self, vector: bool) -> np.ndarray:

        if not vector:
            return self._raw

        else:
            return np.ravel(self._raw)

    def get_mask(self, mask_type: int) -> np.ndarray:
        # mask_type: 0...nuclear mask, 1...cell mask, 2...membrane mask
        return self._mask[:, :, mask_type]

    def get_marker(self) -> str:
        return self._marker

    def get_cytoplasm(self) -> bool:
        return self._cytoplasm

    def get_feature_vector(self) -> np.ndarray:
        feat_vector = []

        if self._cytoplasm:
            for m in range(3):
                for feat in self._intensity_features:
                    feat_vector.append(feat.measure(self, m_type=m))
        else:
            for feat in self._intensity_features:
                feat_vector.append(feat.measure(self, m_type=0))
            for m in range(2):
                for feat in self._morphology_features:
                    feat_vector.append(feat.measure(self, m_type=m))

        return feat_vector

    def get_feature_names(self) -> np.ndarray:
        feature_names = []

        if self._cytoplasm:
            for m in range(3):
                for feat in self._intensity_features:
                    feature_names.append(self.get_marker() + '_' + feat.get_name(m_type=m))
        else:
            for feat in self._intensity_features:
                feature_names.append(self.get_marker() + '_' + feat.get_name(m_type=0))
            for m in range(2):
                for feat in self._morphology_features:
                    feature_names.append(self.get_marker() + '_' + feat.get_name(m_type=m))

        return feature_names

