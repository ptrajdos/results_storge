from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import RepeatedStratifiedKFold
import os
import pickle
from results_storage.results_storage import ResultsStorage
import xarray as xr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from results_storage_experiments.settings import EXPERIMENTS_RESULTS_PATH

def get_metrics():
    return {
        "ACC": accuracy_score,
        "BAC": balanced_accuracy_score,
        "Kappa":cohen_kappa_score
    }


def get_classifiers1():
    return {
        "RF": RandomForestClassifier(n_estimators=10, random_state=12),
        "NB": GaussianNB(),
    }

def get_classifiers2():
    return {
        "RF": RandomForestClassifier(n_estimators=10,random_state=12),
        "NB": GaussianNB(),
        "Tree": DecisionTreeClassifier(random_state=12)
    }


def get_preprocessors1():
    return {
        "PCA2": PCA(n_components=2,random_state=12),
        "PCA3": PCA(n_components=3,random_state=12),
    }

def get_preprocessors2():
    return {
        "PCA2": PCA(n_components=2,random_state=12),
        "PCA3": PCA(n_components=3,random_state=12),
        "PCA4": PCA(n_components=2,random_state=12),
        "PCA5": PCA(n_components=3,random_state=12),
    }


result_dir_path = os.path.join(EXPERIMENTS_RESULTS_PATH,"exp1")
result_file_path = os.path.join(result_dir_path,"iris.pickle")

def run_experiment(n_splits=5, n_repeats=3):
    X, y = load_iris(return_X_y=True)

    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=12)
    n_folds = skf.get_n_splits()
    fold_names = ["f{}".format(i) for i in range(n_folds)]

    classifiers = get_classifiers2() if os.path.exists(result_file_path) else get_classifiers1()
    preprocessors = get_preprocessors2() if os.path.exists(result_file_path) else get_preprocessors1()
    metrics = get_metrics()

    result_holder = ResultsStorage.init_coords(
                coords={
                    "Metric":[k for k in metrics],
                    "Pre":[k for k in preprocessors],
                    "Estim": [k for k in classifiers],
                    "Fold":[k for k in fold_names]
                },
                name="h",
                fill_value=np.nan,
            )
    if os.path.exists(result_file_path):
        print("{} exists".format(result_file_path))

        with open(result_file_path,"rb") as fh:
            
            loaded_holder = pickle.load(fh)
            loaded_holder.name = 'loaded'
            result_holder = xr.merge((loaded_holder, result_holder))['loaded']
            
    else:
        print("{} not exist".format(result_file_path))

    def compute(fold_idx,train_idx, test_idx):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        fold_res = []

        for pre_name in ResultsStorage.coords_need_recalc(result_holder,"Pre"):
            print("{} recalc!".format(pre_name))
            preprocessor = preprocessors[pre_name]
            X_train_pre = preprocessor.fit_transform(X_train,y_train)
            X_test_pre = preprocessor.transform(X_test)

            for estim_name in ResultsStorage.coords_need_recalc(result_holder.sel(Pre=pre_name),"Estim"):
                print("{} recalc!".format(estim_name))
                estimator = classifiers[estim_name]
                estimator.fit(X_train_pre, y_train)
                y_pred = estimator.predict(X_test_pre)

                for metric_name in result_holder["Metric"].values:
                    metric = metrics[metric_name]
                    metric_value = metric(y_pred, y_test)

                    fold_res.append( ( metric_name, pre_name, estim_name, fold_idx , metric_value))

        return fold_res

    rets = Parallel(n_jobs=None)\
        (delayed(compute)(fold_idx, train_idx, test_idx) for fold_idx, (train_idx, test_idx) in enumerate( skf.split(X,y)))
    
    for fold_list in rets:
        for metric_name, pre_name, estim_name,fold_idx, metric_value in fold_list:
            result_holder.loc[{ "Metric":metric_name, "Pre":pre_name, "Estim":estim_name, "Fold":"f{}".format(fold_idx) }] = metric_value


    with open(result_file_path, "wb") as fh:
        pickle.dump(result_holder,file=fh)


def analyse_results(results_directory, output_directory,v=1):
    
    result_files = [f for f in os.listdir(results_directory) if f.endswith(".pickle") ]

    for result_file in result_files:
        result_file_basename = os.path.splitext(result_file)[0]
        result_file_path = os.path.join(results_directory, result_file)

        result_holder = pickle.load(open(result_file_path,"rb"))

        pdf_file_path = os.path.join(output_directory, "{}_{}.pdf".format(result_file_basename,v))
        print("RE: ", result_holder)
        
        with PdfPages(pdf_file_path) as pdf:
        
            for metric_name in result_holder["Metric"].values:
                for estim_name in result_holder["Estim"].values: # orders in new and old may differ
                    #metric x pre x estim x fold
                    # pre x fold
                    sub_results = result_holder.loc[{"Metric":metric_name, "Estim":estim_name}].to_numpy()
                    plt.boxplot(sub_results.transpose())
                    plt.title("{}, {} ".format(metric_name, estim_name))
                    pre_names  = result_holder["Pre"].values 
                    plt.xticks(range(1,len(pre_names) +1), pre_names)
                    pdf.savefig()
                    plt.close()
                    

if __name__ == '__main__':

    os.makedirs(os.path.dirname(result_file_path) ,exist_ok=True)
    if os.path.exists(result_file_path):
        os.remove(result_file_path)
    run_experiment()
    analyse_results(result_dir_path, result_dir_path,v=1)
    run_experiment()
    analyse_results(result_dir_path, result_dir_path,v=2)