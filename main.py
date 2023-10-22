import pandas as pd

from epm.EPM import EPM

if __name__ == '__main__':
    DATASET = 'paper_example'
        
    path_data = f'datasets/{DATASET}.txt'

    df = pd.read_csv(path_data)
    clf = EPM(depth=2, evaluation_metric='rw_norm', bin_strategy='equiwidth', algorithm='best_first', evaluation_threshold=0.2)
    clf.load_data(df)
    clf.search()
    
    clf.visualise(2)
