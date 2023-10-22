# Apriori Exceptional Preferences Mining (A-EPM)

This is a heavily modified version of the Python Exceptional Model Mining implemenation [emm](https://github.com/MathynS/emm) by [MathynS](https://github.com/MathynS).

## 📦 Installation
Either run `.\setup.ps1` when [pyenv-windows](https://github.com/pyenv-win/pyenv-win) is installed on your machine with **Python 3.12.0** set as local or global pyenv, or install the necessary packages (see `/requirements.txt`) in your **Python 3.12.0** installation.

## 💻 Usage

### 🏃‍♂️ Run
A good example on how to use **A-EPM** is the executable file `/main.py`.

### ⚙️ Options

#### 📅 `EPM` Class

| Attribute | Type | Default | Notes | Options | Description |
|---|---|---|---|---|---|
| algorithm | str | 'apriori' | - | ('apriori', 'best_first') | Decide what algorithm to use to construct the subgroups |
| width | int | None | When algorithm 'depth_first' is used: this, evaluation_threshold or both required. | - | Width parameter of the search: amount of subgroups to keep before moving on to the next depth step |
| depth | int | - | Required | - | Depth parameter of the search: amount of iterations in the process, subgroups are described by at most depth attributes |
| evaluation_metric | str or callable | - | Required | ('rw_norm', 'rw_norm_mode', 'rw_cov', 'lw_norm', 'pw_max') | Function to evaluate the subgroups with. You can choose one of the from our paper as a string or create your own evaluation function |
| evaluation_threshold | float | None | Should be a positive float, except when using 'rc_cov' metric. When algorithm 'depth_first' is used: this, width or both required. | - | Quality metric threshold used to prune subgroups after each depth step. |
| frequency_threshold | float | None | Required when algorithm 'apriori' is used. | - | Frequency threshold used to prune subgroups after each depth step for the 'apriori' algorithm. Example: a frequency threshold of 0.2 means that the subgroup should cover at least 20% of the dataset |
| n_bins | int | 8 | Each depth step new bins are created | - | For int or float columns of the dataset not all options are used to create subgroups. Values are divided into bins for which the amount of bins can be specified |
| bin_strategy | str | 'equidepth' | - | ('equidepth', 'equiwidth') | Method to create bins for int and float columns |
| bin_subgroups | str | 'both' | - | ('both', 'per_bin', 'per_split') | When creating subgroups of bins, decide whether to make a subgroup on a split (e.g. x <= 5), a bin (e.g. 3 < x <= 5) or both. |
| candidate_size | int | width^2 | - | - | Amount of subgroups to keep in memory each depth step while using the 'best_first' algorithm |
| log_level | int | 50 | - | - | Choose the logging log level. When using a log_level of 0, the found subgroups will be shown in the console |

#### ⌛ `load_data()` method
This method only requires a single `df` argument with a dataset containing a `ranking` column.

#### 🔍 `search()` method

| Attribute | Type | Default | Description |
| --- | --- | --- | --- |
| descriptive_cols | str or list | All columns except `ranking` column | Single column or list of columns that can be used to create subgroups with |

#### 👁️ `visualise()` method
This method has a single optional `subgroups_amount` argument expecting an `int`. When this method is called (after calling `load_data()` and `search()`), this will visualise the minimum of (`subgroups_amount`, #subgroups) best subgroups.

⚠️ **Warning!** If no amount is given for `subgroups_amount`, all subgroups will be visualised.