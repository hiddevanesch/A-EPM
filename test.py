import pandas as pd

from epm.EPM import EPM

def generateResults(dataset : str, eval_metric : str, depth : int, eval_threshold : int, frequency_threshold : int):

    path_data = f'datasets/{dataset}.txt'
    df = pd.read_csv(path_data)

    num_descriptors = len(df) - 1

    clf_apriori = EPM(depth = num_descriptors, evaluation_metric = eval_metric, bin_strategy = 'equiwidth', algorithm = 'apriori', 
                    evaluation_threshold = eval_threshold, frequency_threshold = frequency_threshold)
    clf_apriori.load_data(df)
    clf_apriori.search()


    apriori_results = []
    for i in clf_apriori.algorithm.subgroups:
        apriori_results.append(i.to_string())
    clf_bestfirst = EPM(depth = depth, evaluation_metric = eval_metric, bin_strategy = 'equiwidth', algorithm = 'best_first', 
                        evaluation_threshold = eval_threshold)
    clf_bestfirst.load_data(df)
    clf_bestfirst.search()

    f_name_bestfirst = f"results/best_first_{dataset}_{eval_metric}_{eval_threshold}_{frequency_threshold}"
    f_bestfirst= open(f_name_bestfirst, "w")

    for i in clf_bestfirst.algorithm.subgroups:
        # Add to file
        f_bestfirst.write(i.to_string() + "\n")
        # Check if already in apriori results then remove
        if i.to_string() in apriori_results:
            apriori_results.remove(i.to_string())

    f_name_apriori = f"results/apriori_not_best_first_{dataset}_{eval_metric}_{eval_threshold}_{frequency_threshold}"

    f_results = open(f_name_apriori, "w")
    for i in apriori_results:
        f_results.write(i + "\n")
        
        
if __name__ == '__main__':
    generateResults(dataset="sushi", eval_metric="rw_norm", depth=3, eval_threshold=0.2, frequency_threshold=0.2)    
