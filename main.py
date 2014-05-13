from datasets import Datasets
import argparse
import importlib
import logging
import yaml
import os
import sys
import importlib
import print_score
import shutil
from plot import plot_data, plot_metric, plot_training_results
import tabulate
dt = Datasets()
def generate_pdf(metric_tuples):
    pass

def dump_results(algorithm, metric_tuples):
    file = open(os.path.join("../dumps", "%s.txt" % algorithm), 'w+')
    headers = ["Algorithm", "Dataset", "Training Size", "Metric", "Score"]
    table = []
    for tup in metric_tuples:
        (algorithm, dataset, training_size, metric, score) = (tup[0], tup[1], tup[2], tup[3][0][0], tup[3][0][2])
        algorithm = algorithm.replace("_", " ").title()
        dataset = dataset.replace("_", " ").title()
        metric = metric.replace("_", " ").title()
        table.append([algorithm, dataset,  training_size, metric, score])
    file.write("\n%s\n" % tabulate.tabulate(table, headers, tablefmt='latex'))
    file.close()



def print_results(training_size, algorithm, dataset, metric_tuples):
    #print "\nFor Algorithm::\t%s" % algorithm
    #print "For Dataset::\t%s\n" % dataset
        for met_tup in metric_tuples:
            func = getattr(print_score, "print_%s" % met_tup[0])
            func(training_size, algorithm, dataset, met_tup[2])

def _get_algorithm_class(algorithm_name):
    module = importlib.import_module("%s" % algorithm_name)

    if not module:
        logging.error("Module %s not found" % algorithm_name)

    class_name = algorithm_name.replace("_"," ").title().replace(" ","")
    logging.info("Algorithm %s loaded from module %s" % (class_name, algorithm_name))
    return getattr(module, class_name)

def _load_dataset(dataset, dt):
    load_function = getattr(dt, "load_%s" % dataset)
    if not load_function:
        logging.error("Dataset %s couldn't be loaded" % dataset)
        sys.exit(0)

    return load_function()


def run_algorithms(algorithms, datasets, metrics, output, conf):
    dts = Datasets()
    shall_plot = conf.get("plot_data")
    if shall_plot:
        plot_dir = conf.get("plot_dir", "../plots")

        tmp_plot_dir = "../plots_1"
        if os.path.exists(tmp_plot_dir):
            shutil.rmtree(tmp_plot_dir)

        os.mkdir(tmp_plot_dir)

        orig_data_dir = os.path.join(tmp_plot_dir, "original")
        os.mkdir(orig_data_dir)
        for dataset in datasets:
            plot_data(os.path.join(orig_data_dir, "%s-orig.png"  % dataset), "%s-orig" % dataset, dataset)

    if output == 'dump_text' and not os.path.exists("../dumps"):
        os.mkdir("../dumps")

    for algorithm in algorithms:

        if shall_plot:
            algo_dir = os.path.join(tmp_plot_dir, algorithm)
            os.mkdir(algo_dir)

        algo_conf = conf["algorithms"].get(algorithm, None)

        if not algo_conf:
            logging.error("Algorithm %s not found in conf file" % algorithm)
            sys.exit(0)

        algo_conf['name'] = algorithm
        learn_class = _get_algorithm_class(algorithm)
        learn = learn_class(**algo_conf)
        learn._set_cross_validation(conf.get("cv_method", None), conf.get("cv_metric", None), conf.get("cv_params", None))
        results = []
        for dataset in datasets:
            if dataset not in conf["datasets"]:
                logging.error("Dataset %s not found" % dataset)
                sys.exit(0)

            cv_dir = None
            if shall_plot:
                dataset_dir = os.path.join(algo_dir, dataset)
                os.mkdir(dataset_dir)

                if algo_conf.get("cross_validate", True):
                    cv_dir = os.path.join(dataset_dir, "cv")
                    os.mkdir(cv_dir)

            training_sizes = conf.get("training_size", [0.40])
            scores = []
            for training_size in training_sizes:
                data = dts.load_dataset(dataset, training_size)

                learn.set_dataset(dataset, training_size*100, cv_dir)
                if learn.check_type(data["type"]):
                    eval_metrics = []
                    if metrics:
                        eval_metrics.extend(metrics)
                    else:
                        eval_metrics.extend(algo_conf["allowed_metrics"])

                    learn.train(data["x_train"], data["y_train"])
                    result_tups = learn.evaluate(data["x_test"], data["y_test"], eval_metrics)

                    print_results(training_size, algorithm, dataset, result_tups)
                    results.append((algorithm, dataset, training_size, result_tups))

                    if shall_plot:
                        decision_plot_path = os.path.join(dataset_dir, "decision-%s_%s_size_%d.png" % (dataset, algorithm, training_size * 100))
                        learn.plot_results(decision_plot_path, dataset, training_size, data['x_train'], data['x_test'], data['y_train'], data['y_test'])

                        for metric, y_test, score in result_tups:
                            metric_plot_path = os.path.join(dataset_dir, "metric-%s-%s_%s_size_%d.png" % (metric, dataset, algorithm, training_size * 100))
                            plot_metric(metric_plot_path, data['type'], y_test, data['y_test'], dataset, algorithm, training_size * 100)
                    scores.append(result_tups[0][2])
            if shall_plot:
                train_plot_path = os.path.join(dataset_dir, "train_vs_acc-%s_%s.png" % (algorithm, dataset))
                plot_training_results(train_plot_path, [train_size * 100 for train_size in training_sizes], scores)

        if output == "pdf":
            generate_pdf(results)
        elif output == "dump_text":
            dump_results(algorithm, results)
    if conf.get("plot_data", False):
        shutil.rmtree(plot_dir)
        shutil.move(tmp_plot_dir, plot_dir)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    FORMAT = '%(levelname)s %(asctime)s %(name)s: %(message)s'

    parser.add_argument("-c", "--config", type=str, default="./classification.json",
            help="Config file path")
    parser.add_argument("-a", "--algorithms", type=str, nargs='+',
            help="""Algorithm to run(ones mentioned in classification.json):\n
                        values: all or specific one from json file\n
                """, default=['all'])


    parser.add_argument("-d", "--datasets", type=str, nargs='+',
            help="""Datasets to run(mentioned in classification.json):\n
                        values: all or specific one from json file\n
                """, default=['all'])

    parser.add_argument("-m", "--metrics", type=str, nargs='+',
            help="""Metrics to evaluate(mentioned in classification.json):\n
                        values:: all allowed for a given algorithm(mentioned in json file) or specific ones\n
                """, default=['all'])

    parser.add_argument("-o", "--output", type=str,
            help="""Output Pattern values: [pdf(Generate a pdf report), print(Print to screen), dump(Dump in text file ./output.txt)
                """, default="print")

    logging.basicConfig(format=FORMAT, level=logging.INFO)
    args = parser.parse_args()

    logging.info("Main::Running algorithm(s): %s" % args.algorithms)
    logging.info("Main::Running on dataset(s): %s" % args.datasets)
    logging.info("Main::Evaluate metric(s): %s" % args.metrics)
    logging.info("Main::Output Mode: %s" % args.output)

    conf_file = open(os.path.abspath(args.config))
    conf_json = yaml.load(conf_file)

    algorithms = []
    datasets = []
    metrics = []

    if len(args.algorithms) and args.algorithms[0] == 'all':
        algorithms.extend(conf_json["algorithms"].keys())
    else:
        algorithms.extend(args.algorithms)

    if len(args.datasets) and args.datasets[0] == 'all':
        datasets.extend(conf_json["datasets"])
    else:
        datasets.extend(args.datasets)

    if len(args.metrics) > 1 and 'all' not in args.metrics:
        metrics.extend(args.metrics)
    run_algorithms(algorithms, datasets, metrics, args.output, conf_json)
