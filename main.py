import argparse
from monks import run_monks
from utils import read_csv_results, save_obj, LOSSES, EVALUATION_METRICS
from cup import run_cup, best_models_assessment
from sklearn.model_selection import ParameterGrid
from cup import load_dev_set_cup, load_blind_test_cup, load_internal_test_cup
from validation import grid_search_cv
import json
import pickle
import pandas as pd
from ensemble import Ensemble

def main(experiment_name):
    if (experiment_name in AVAILABLE_EXPERIMENTS[:4]):
        with open("jsons/monks.json") as json_monks:
            monks_config = json.load(json_monks).get(experiment_name)
            monks_config['name'] = experiment_name
            run_monks(monks_config)
    
    elif (experiment_name == "cup-best_single"):
        best_gs_config = read_csv_results("csvs/fine_gs.csv")['params'][0]
        run_cup(best_gs_config)
    
    elif (experiment_name == "cup-custom"):
        custom_config = json.load("jsons/cup_custom_config.json")
        run_cup(custom_config)
    
    elif (experiment_name == "cup-best_models_assessment"):
        configs = read_csv_results("csvs/fine_gs.csv")['params'][:2]
        best_models_assessment(configs)
    
    elif (experiment_name == "cup-ensemble_blind_pred"): #TODO: test predictions
        X_blind = load_blind_test_cup()
        with open('pkls/ens.pkl', 'rb') as pkl_file:
            ens = pickle.load(pkl_file)
        preds = ens.predict(X_blind)
        df = pd.DataFrame(preds)
        f = open('TheEnsemblers_ML-CUP22-TS.csv', 'a')
        f.write('# Giulia Ghisolfi	Lorenzo Leuzzi	Irene Testa\n')
        f.write('# TheEnsemblers\n')
        f.write('# ML-CUP22\n')
        f.write('# 23/01/2023\n')
        df.to_csv(f, header=False, lineterminator='\r')
        f.close()

    elif (experiment_name == "cup-ensemble_assessment"):
        params = read_csv_results("csvs/fine_gs.csv")['params'][:10]
        X_dev, y_dev = load_dev_set_cup()
        X_test, y_test = load_internal_test_cup()

        ens = Ensemble(params, n_trials = 5)
        ens.fit(X_dev, y_dev, X_test, y_test)
        ens.plot()
        preds = ens.predict(X_test)
        mse_score = LOSSES[ens.loss](y_test, preds)
        mee_score = EVALUATION_METRICS[ens.evaluation_metric](y_test, preds)
        scores = {ens.loss : mse_score, ens.evaluation_metric : mee_score}
        print(scores)
        
        save_obj(preds, "pkls/internal_preds.pkl")
        with open("jsons/ensemble_assessment_scores.json", 'w') as f:
            json.dump(scores, fp = f, indent = 4)

    elif (experiment_name == "cup-ensemble_cv"):
        params = read_csv_results("csvs/fine_gs.csv")['params'][:10]
        X_dev, y_dev = load_dev_set_cup()
        ens = Ensemble(params, n_trials = 5)
        cv_results = ens.validate(X_dev, y_dev, k=5)
        results_path = 'csvs/cv_ens_results.csv'
        df = pd.DataFrame([cv_results])
        df.to_csv(results_path)

    elif (experiment_name == "cup-coarse_gs"):
        X_dev_cup, y_dev_cup = load_dev_set_cup()
        with open("jsons/grid_searches.json") as json_gs:
            coarse_grid = ParameterGrid(json.load(json_gs).get("coarse"))
            grid_search_cv(grid=coarse_grid, X=X_dev_cup, y=y_dev_cup, k=5, results_path="csvs/coarse_gs.csv")
 
    elif (experiment_name == "cup-fine_gs"):
        X_dev_cup, y_dev_cup = load_dev_set_cup()
        with open("jsons/grid_searches.json") as json_gs:
            coarse_grid = ParameterGrid(json.load(json_gs).get("fine"))
            grid_search_cv(grid=coarse_grid, X=X_dev_cup, y=y_dev_cup, k=5, results_path="csvs/fine_gs.csv")
    
    else:
        raise ValueError(f"Incorrect input. Experiment name must be one of {AVAILABLE_EXPERIMENTS}")

if __name__ == "__main__":
    AVAILABLE_EXPERIMENTS = [
        "monks-1", "monks-2", "monks-3", "monks-3reg", "cup-best_single", "cup-ensemble_pred",
        "cup-custom", "cup-coarse_gs", "cup-fine_gs", "cup-ensemble_assessment", "cup-ensemble_blind_pred",
        "cup-ensemble_cv", "cup-best_models_assessment"
        ]
    parser = argparse.ArgumentParser(description='Input an experiment name for ML project 22/23')
    parser.add_argument(
        'experiment_name', type=str,
        help=f'The name of the experiment. \ Must be one of: {AVAILABLE_EXPERIMENTS}') 
    args = parser.parse_args()

    main(args.experiment_name)

    