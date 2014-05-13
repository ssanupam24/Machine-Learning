from prettytable import PrettyTable
def print_accuracy_score(training_size, algorithm, dataset, score):
    table = PrettyTable(["training_size", "algorithm", "datset","metric", "score"])
    table.add_row([training_size, algorithm, dataset, "accuracy_score", score])
    print table

def print_precision_recall_fscore(algorithm, dataset, score):
    pass
    #print score).format('centered')

def print_cv_scores(algorithm, dataset, training_size, cv_results):
    table = PrettyTable(["parameters", "metric", "mean", "std"])
    print "Algorithm: %s, Dataset: %s, Training Size: %s" % (algorithm, dataset, training_size)
    for result in cv_results:
        table.add_row(result[:-1])
    print table
