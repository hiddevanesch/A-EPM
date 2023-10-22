import logging
import pandas as pd

from epm.description import Description
from epm.preference_matrix import PM

class Subgroup:

    def __init__(self, data: pd.DataFrame, description: Description, pm: PM = None, coverage: float = None):
        self.data = data
        self.description = description
        self.pm = pm
        self.coverage = coverage
        self.score = None

    def decrypt_description(self, translation):
        self.description.decrypt(translation)

    @property
    def size(self):
        return len(self.data)
    
    def to_string(self):
        descriptors = str(sorted(str(self.description).split(' && '))).replace(',', ' &&').replace('[', '').replace(']', '').replace("'", '')
        return f"{descriptors} | score: {round(self.score, 3)} | size: {self.size}"

    def print(self):
        logging.debug(self.to_string())
