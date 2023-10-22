from typing import Union
from itertools import chain
from copy import deepcopy


class Description:

    def __init__(self, attribute: str, value: Union[str, float, int, bool] = None, dictionary: dict = None):
        if dictionary is not None:
            self.description = deepcopy(dictionary)
        else:
            if attribute == 'all':
                value = 'all'
            self.description = dict()
            self.description[attribute] = value

    def __contains__(self, col):
        return col in self.description

    def extend(self, attribute, value):
        if 'all' in self.description:
            self.description = dict()
            self.description[attribute] = value
        else:
            self.description[attribute] = value
        return self
    
    def merge(self, other: 'Description'):
        self.description.update(other.description)

    def decrypt(self, translation):
        for key, value in self.description.items():
            if key in translation:
                self.description[key] = translation[key][value]

    def __str__(self):
        if 'all' in self.description:
            return 'all'
        else:
            result = []
            for key, value in self.description.items():
                if isinstance(value, list):
                    if value[1] is None:
                        result.append(f"{key} >= {round(value[0], 3)}")
                    elif value[0] is None:
                        result.append(f"{key} <= {round(value[1], 3)}")
                    else:
                        result.append(f"{int(value[0])} < {key} <= {int(value[1])}")
                else:
                    result.append(f"{key} = {value}")
            return " && ".join(result)
