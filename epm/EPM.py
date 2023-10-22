import logging

from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from epm.util import downsize
from epm.subgroup import Subgroup
from epm.description import Description
from epm.metrics import metrics
from epm.preference_matrix import ranking_to_preference_matrix, aggregate_preference_matrix, distance_matrix
from epm.algorithm import Algorithm

class EPM:
    def __init__(self, depth: int, evaluation_metric: str, evaluation_threshold: float = None, frequency_threshold: float = None,
                 width: int = None, bin_subgroups = 'both', candidate_size: int = None, algorithm: str = 'apriori',
                 n_bins: int = 8, bin_strategy: Optional[str] = 'equidepth', log_level=50):
        logging.basicConfig(filename=None, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        try:
            self.evaluation_function = metrics[evaluation_metric]
        except KeyError:
            raise ValueError(f"No such metric: {evaluation_metric}")
        
        if evaluation_metric == 'rw_cov':
            strategy = 'minimize'
            if evaluation_threshold is not None and evaluation_threshold > 0:
                raise ValueError("Evaluation threshold should be less than 0 for rw_cov")
        else:
            strategy = 'maximize'
            if evaluation_threshold is not None and evaluation_threshold < 0:
                raise ValueError("Evaluation threshold should be greater than 0 when not using rw_cov")

        if evaluation_metric == 'rw_norm_mode':
            aggregate_technique = 'mode'
        else:
            aggregate_technique = 'mean'

        if algorithm != 'apriori' and algorithm != 'best_first':
            raise ValueError(f"Invalid algorithm: {algorithm}")
        
        if algorithm == 'apriori' and (candidate_size is not None or width is not None):
            raise ValueError("A limit on candidate size and width are not allowed for apriori algorithm")
        elif algorithm == 'apriori' and (evaluation_threshold is None or frequency_threshold is None):
            raise ValueError("Evaluation and frequency threshold should be specified for apriori algorithm")
        elif algorithm == 'best_first' and frequency_threshold is not None:
            raise ValueError("Frequency threshold is not allowed for best_first algorithm")
        elif algorithm == 'best_first' and width is None and evaluation_threshold is None:
            raise ValueError("Either width or evaluation_threshold should be specified for best_first algorithm")

        self.settings = dict(
            strategy=strategy,
            width=width,
            evaluation_threshold=evaluation_threshold,
            frequency_threshold=frequency_threshold,
            algorithm=algorithm,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            bin_subgroups=bin_subgroups,
            candidate_size=candidate_size,
            aggregate_technique=aggregate_technique,
            depth=depth
        )

        self.dataset = None
        self.algorithm = None
        self.unique_labels = None

    def load_data(self, data: pd.DataFrame):
        logging.info("Loading data...")
        df, translations = downsize(data.copy(deep=True))
        self.settings['object_cols'] = translations

        # Generate preference matrices
        self.unique_labels = sorted(set(filter(str.isalpha, df['ranking'].iloc[0]))) # Assuming each row contains all labels
        df['PM'] = df['ranking'].apply(lambda x: ranking_to_preference_matrix(x, self.unique_labels))

        matrix_d = aggregate_preference_matrix(list(df['PM']), self.settings['aggregate_technique'])

        self.dataset = Subgroup(data=df, description=Description('all'), pm=matrix_d)
        self.algorithm = Algorithm(self.settings, self.dataset, self.evaluation_function)

    def search(self, descriptive_cols: List[str] = None):
        logging.info("Start")
        if descriptive_cols is None:
            descriptive_cols = [c for c in self.dataset.data.columns if c != 'ranking' and c != 'PM']
        if any(c not in self.dataset.data.columns for c in descriptive_cols):
            raise ValueError("All specified descriptive columns should be present in the dataset")
        self.algorithm.run(descriptive_cols)
        self.algorithm.decrypt_descriptions(self.settings['object_cols'])
        self.algorithm.print()

    def visualise(self, subgroups_amount: int = None):
        if subgroups_amount is None:
            subgroups_amount = len(self.algorithm.subgroups)
        for subgroup in self.algorithm.subgroups[:subgroups_amount]:
            cmap = LinearSegmentedColormap.from_list(
                'custom', [(0, 'red'), (0.5, 'white'), (1, 'green')])
            fig, ax = plt.subplots(1, 3, figsize=(12, 5))

            title = subgroup.to_string()
            fig.suptitle(title, fontsize=16)
            fig.canvas.manager.set_window_title(title)

            ax[0].imshow(self.dataset.pm.pm, cmap=cmap, vmin=-1, vmax=1)
            ax[0].set_title('Base Matrix')
            ax[0].set_xticks(np.arange(len(self.unique_labels)))
            ax[0].set_yticks(np.arange(len(self.unique_labels)))
            ax[0].set_xticklabels(self.unique_labels)
            ax[0].set_yticklabels(self.unique_labels)

            ax[1].imshow(subgroup.pm.pm, cmap=cmap, vmin=-1, vmax=1)
            ax[1].set_title('Subgroup Matrix')
            ax[1].set_xticks(np.arange(len(self.unique_labels)))
            ax[1].set_yticks(np.arange(len(self.unique_labels)))
            ax[1].set_xticklabels(self.unique_labels)
            ax[1].set_yticklabels(self.unique_labels)

            ax[2].imshow(distance_matrix(self.dataset.pm, subgroup.pm), cmap=cmap, vmin=-1, vmax=1)
            ax[2].set_title('Distance Matrix')
            ax[2].set_xticks(np.arange(len(self.unique_labels)))
            ax[2].set_yticks(np.arange(len(self.unique_labels)))
            ax[2].set_xticklabels(self.unique_labels)
            ax[2].set_yticklabels(self.unique_labels)
            
            plt.tight_layout()

            plt.show()