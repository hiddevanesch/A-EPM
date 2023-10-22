from copy import deepcopy
import pandas as pd

import logging

from typing import List

from epm.subgroup import Subgroup
from epm.description import Description
from epm.preference_matrix import aggregate_preference_matrix

class Algorithm:
    def __init__(self, settings, dataset, evaluation_function):
        self.settings = settings
        self.dataset = dataset
        self.evaluation_function = evaluation_function
        self.subgroups = [dataset]
        self.candidates = []
        self.items = 1
        self.evaluation_threshold = settings['evaluation_threshold']
        self.frequency_threshold = settings['frequency_threshold']
        self.width = settings['width']
        self.algorithm = settings['algorithm']
        try:
            self.candidate_size = int(settings['candidate_size'])
        except (KeyError, TypeError):
            if self.width is None:
                self.candidate_size = None
            else:
                self.candidate_size = self.width ** 2
        self.strategy = settings['strategy']
        self.worst_score = None
        self.scores = []
        self.depth = settings['depth']
        self.constructed_descriptions = []
        self.current_depth = None
        self.frequent_itemset = []
        self.intervals_list = []

    def run(self, descriptive_cols: List[str]):
        self.current_depth = 0
        while self.current_depth < self.depth:
            self.increase_depth(descriptive_cols)
            self.current_depth += 1

    def increase_depth(self, descriptive_cols: List[str]):
        if len(self.subgroups) == 0:
                return
        if self.algorithm == 'best_first':
            for subgroup in self.subgroups:
                for col in descriptive_cols:
                    self.create_subgroups(subgroup, col)
            self.select_candidates()
        elif self.algorithm == 'apriori':
            if self.current_depth == 0:
                for subgroup in self.subgroups:
                    for col in descriptive_cols:
                        self.create_subgroups(subgroup, col)
            else:
                previous_freq_itemset = deepcopy(self.frequent_itemset)
                self.frequent_itemset = []
                for i, item1 in enumerate(previous_freq_itemset):
                    for j in range(i + 1, len(previous_freq_itemset)):
                        item2 = previous_freq_itemset[j]
                        self.merge_subgroups(item1, item2)
            # If in the last iteration, select the candidates (candidates -> subgroups)
            if self.current_depth == self.depth - 1:
                self.select_candidates()
    
    def merge_subgroups(self, subgroup1: Subgroup, subgroup2: Subgroup):
        # Check whether subgroup1.description and subgroup2.description do not overlap
        overlap = set(subgroup1.description.description.keys()) & set(subgroup2.description.description.keys())

        for col in overlap:
            if subgroup1.description.description[col] != subgroup2.description.description[col]:
                return
        
        if len(overlap) != (self.current_depth - 1):
            return

        subset = pd.merge(subgroup1.data, subgroup2.data, how='inner')
        if len(subset) == 0:
            return
        
        new_desc = Description(None, dictionary=subgroup1.description.description)
        new_desc.merge(subgroup2.description)
        self.check_for_duplicates_and_add(new_desc, subset)

    def create_subgroups(self, subgroup: Subgroup, column: str):
        if column in subgroup.description:
            return
        data = subgroup.data
        values = list(data[column].unique())
        if len(values) == 1:  # No need to make a split for a single value
            return
        if column in self.settings['object_cols'] or len(values) < self.settings['n_bins']:
            self.create_subgroups_categorical(subgroup, column, data, values)
        else:  # Float or Int
            self.create_subgroups_numerical(subgroup, column, data)

    def create_subgroups_categorical(self, subgroup, column, data, values):
        while len(values) > 0: # Reasonable size to keep in the beam
            value = values.pop(0)
            subset = data[data[column] == value]
            if len(subset) == 0:
                continue
            new_desc = deepcopy(subgroup.description).extend(column, value)
            self.check_for_duplicates_and_add(new_desc, subset)

    def create_subgroups_numerical(self, subgroup, column, data):
        if (self.intervals_list == []):
            if self.settings['bin_strategy'] == 'equidepth':
                _, intervals = pd.qcut(data[column].tolist(), q=self.settings['n_bins'],
                                        duplicates='drop', retbins=True)
            elif self.settings['bin_strategy'] == 'equiwidth':
                _, intervals = pd.cut(data[column].tolist(), bins=self.settings['n_bins'],
                                        duplicates='drop', retbins=True)
            else:
                raise ValueError(f"Invalid bin strategy `{self.settings['strategy']}`")
            
            self.intervals_list = list(intervals)

        if self.settings['bin_subgroups'] == 'both' or self.settings['bin_subgroups'] == 'per_bin':
            self.create_subgroups_bins(subgroup, column, data, self.intervals_list)
        if self.settings['bin_subgroups'] == 'both' or self.settings['bin_subgroups'] == 'per_split':
            self.create_subgroups_splits(subgroup, column, data, self.intervals_list)

    def create_subgroups_bins(self, subgroup, column, data, intervals_list):
        intervals = deepcopy(intervals_list)
        if self.settings['bin_subgroups'] == 'both':
            intervals.pop(0)
            intervals.pop(-1)
        lower_bound = intervals.pop(0)
        while len(intervals) > 0:
            upper_bound = intervals.pop(0)
            subset = data[(data[column] > lower_bound) & (data[column] <= upper_bound)]
            if len(subset) == 0:
                continue
            new_desc = deepcopy(subgroup.description).extend(column, [lower_bound, upper_bound])
            self.check_for_duplicates_and_add(new_desc, subset)
            lower_bound = upper_bound

    def create_subgroups_splits(self, subgroup, column, data, intervals_list):
        intervals = deepcopy(intervals_list)
        for interval in intervals[1:-1]:
            subset = data[data[column] >= interval]
            if len(subset) != 0:
                new_desc = deepcopy(subgroup.description).extend(column, [interval, None])
                self.check_for_duplicates_and_add(new_desc, subset)
            subset = data[data[column] <= interval]
            if len(subset) != 0:
                new_desc = deepcopy(subgroup.description).extend(column, [None, interval])
                self.check_for_duplicates_and_add(new_desc, subset)

    def check_for_duplicates_and_add(self, new_desc, subset):
        try:
            if not new_desc.description in self.constructed_descriptions:
                self.constructed_descriptions.append(new_desc.description)
                if self.algorithm== 'best_first':
                    pm = aggregate_preference_matrix(list(subset['PM']), self.settings['aggregate_technique'])
                    subgroup = Subgroup(subset, new_desc, pm=pm)
                    subgroup.score = self.evaluation_function(self.dataset, subgroup)
                elif self.algorithm == 'apriori':
                    coverage = len(subset) / len(self.dataset.data)
                    if coverage < self.frequency_threshold:
                        return
                    pm = aggregate_preference_matrix(list(subset['PM']), self.settings['aggregate_technique'])
                    subgroup = Subgroup(subset, new_desc, pm=pm, coverage=coverage)
                    subgroup.score = self.evaluation_function(self.dataset, subgroup)
                    self.frequent_itemset.append(subgroup)
                if self.evaluation_threshold is not None:
                    if (self.strategy == 'maximize' and subgroup.score > self.evaluation_threshold) or \
                    (self.strategy == 'minimize' and subgroup.score < self.evaluation_threshold):
                        self.add_subgroup(subgroup)
        except:
            logging.debug(f"Skipping subgroup with description: {str(new_desc)} due to comparison error")

    def add_subgroup(self, subgroup: Subgroup):
        if self.algorithm == 'best_first':
            if self.candidate_size is None or len(self.candidates) < self.candidate_size:
                self.candidates.append(subgroup)
                self.scores.append(subgroup.score)
                self.worst_score = min(self.scores) if self.strategy == 'maximize' else max(self.scores)
            elif (self.strategy == 'maximize' and subgroup.score > self.worst_score) or \
                    (self.strategy == 'minimize' and subgroup.score < self.worst_score):
                idx = self.scores.index(self.worst_score)
                del self.scores[idx]
                del self.candidates[idx]
                self.candidates.append(subgroup)
                self.scores.append(subgroup.score)
                self.worst_score = min(self.scores) if self.strategy == 'maximize' else max(self.scores)
        elif self.algorithm == 'apriori':
            self.candidates.append(subgroup)

    def select_candidates(self):
        self.candidates.sort(key=lambda x: x.score, reverse=(self.strategy == 'maximize'))
        if self.width is not None:
            self.candidates = self.candidates[:min(self.width, len(self.candidates))]
        self.subgroups = deepcopy(self.candidates)
        if len(self.subgroups) > 0:
            self.scores = [s.score for s in self.subgroups]
            self.worst_score = min(self.scores) if self.strategy == 'maximize' else max(self.scores)
        else:
            self.scores = []
            self.worst_score = 0

    def decrypt_descriptions(self, translation):
        for s in self.subgroups:
            s.decrypt_description(translation)

    def print(self):
        if len(self.subgroups) > 0:
            logging.debug("-" * 20)
            for s in self.subgroups:
                s.print()
        else:
            logging.debug("No subgroups found, try different values for the width or the evaluation threshold")