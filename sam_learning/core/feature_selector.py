"""Module to enable feature selection for the actions' numeric preconditions."""
import pandas as pd
from typing import List


from sklearn.feature_selection import SequentialFeatureSelector

class FeatureSelector:
    samples_dataset: pd.DataFrame
    labels_dataset: pd.DataFrame

    def __init__(self, column_names: List[str]):
        self.samples_dataset = pd.DataFrame(columns=column_names)
        self.labels_dataset = pd.DataFrame(columns=["label"])


    def add_sample(self, sample: List[float], label: int) -> None:
        """Adds a sample to the dataset.

        :param sample: the sample to add to the dataset.
        :param label: the label of the sample.
        """
        self.samples_dataset.loc[len(self.samples_dataset)] = sample
        self.labels_dataset.loc[len(self.labels_dataset)] = label


    def get_selected_features(self, threshold: float) -> List[str]:
        pass