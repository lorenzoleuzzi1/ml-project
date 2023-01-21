import argparse
from monks import run_monks
from utils import read_csv_results, save_obj, LOSSES, EVALUATION_METRICS
from cup import run_cup, best_models_assessment
from sklearn.model_selection import ParameterGrid
from cup import load_dev_set_cup, load_blind_test_cup, load_internal_test_cup, load_train_cup
from validation import grid_search_cv
import json
import pickle
import pandas as pd
from ensemble import Ensemble

def main(script_name):
    if (script_name in AVAILABLE_SCRIPTS[:4]):
        # run monks
        with open("jsons/monks_params.json") as json_monks:
            monks_config = json.load(json_monks).get(script_name)
            monks_config['name'] = script_name
            run_monks(monks_config)
    
    elif (script_name == "cup-best_single"):
        # run the best single congifuration for the cup
        best_gs_config = read_csv_results("csvs/fine_gs.csv")['params'][0]
        run_cup(best_gs_config)
    
    elif (script_name == "cup-custom"):
        # run a custom configuration for the cup
        custom_config = json.load("jsons/cup_custom_config.json")
        run_cup(custom_config)
    
    elif (script_name == "cup-best_models_assessment"):
        # produce scores and plots for the best 10 single configurations on the cup
        configs = read_csv_results("csvs/fine_gs.csv")['params'][:10]
        scores = best_models_assessment(configs)
        with open("jsons/best_models_assessment_scores.json", 'w') as f:
            json.dump(scores, fp = f, indent = 4)
    
    elif (script_name == "cup-ensemble_blind_pred"):
        # make the cup blind predictions using the final ensemble model
        X_blind = load_blind_test_cup()
        with open('pkls/ens.pkl', 'rb') as pkl_file:
            ens = pickle.load(pkl_file) # read the previously saved ensemble model
        preds = ens.predict(X_blind)
        df = pd.DataFrame(preds)
        df.index += 1
        f = open('TheEnsemblers_ML-CUP22-TS.csv', 'a')
        f.write('# Giulia Ghisolfi	Lorenzo Leuzzi	Irene Testa\n')
        f.write('# TheEnsemblers\n')
        f.write('# ML-CUP22\n')
        f.write('# 23/01/2023\n')
        df.to_csv(f, header=False)
        f.close()

    elif (script_name == "cup-ensemble_assessment"):
        # produce scores and plots for the ensemble model assessment on the cup
        params = read_csv_results("csvs/fine_gs.csv")['params'][:10]
        X_dev, y_dev = load_dev_set_cup()
        X_test, y_test = load_internal_test_cup()

        ens = Ensemble(params, n_trials = 5)
        ens.fit(X_dev, y_dev, X_test, y_test)
        ens.plot()
        preds = ens.predict(X_test)
        test_mse_score = LOSSES[ens.loss](y_test, preds)
        test_mee_score = EVALUATION_METRICS[ens.evaluation_metric](y_test, preds)
        scores = {
            f"training_{ens.loss}" : ens.final_train_loss, f"training_{ens.evaluation_metric}" : ens.final_train_score,
            f"test_{ens.loss}" : test_mse_score, f"test_{ens.evaluation_metric}" : test_mee_score
            }
        print(scores)
        
        # save_obj(preds, "pkls/internal_preds.pkl")
        with open("jsons/ensemble_assessment_scores.json", 'w') as f:
            json.dump(scores, fp = f, indent = 4)

    elif (script_name == "cup-ensemble_cv"):
        # perform cross validation on the cup ensemble model (10 best configurations)
        params = read_csv_results("csvs/fine_gs.csv")['params'][:10]
        X_dev, y_dev = load_dev_set_cup()
        ens = Ensemble(params, n_trials = 5)
        cv_results = ens.validate(X_dev, y_dev, k=5)
        with open("jsons/ensemble_cv_results.json", 'w') as f:
            json.dump(cv_results, fp = f, indent = 4)
        results_path = 'jsons/ensemble_cv_results.csv'
        df = pd.DataFrame([cv_results])
        df.to_csv(results_path)

    elif (script_name == "cup-ensemble_final_fit"):
        # fit the ensemble with the whole cup training dataset
        params = read_csv_results("csvs/fine_gs.csv")['params'][:10]
        X_tr, y_tr = load_train_cup()
        ens = Ensemble(params, 5)    
        ens.fit(X_tr, y_tr)
        save_obj(ens, "pkls/ens.pkl") # save the ensemble model

    elif (script_name == "cup-coarse_gs"):
        # perform the coarse grid search for the cup
        X_dev_cup, y_dev_cup = load_dev_set_cup()
        with open("jsons/grid_searches.json") as json_gs:
            coarse_grid = ParameterGrid(json.load(json_gs).get("coarse"))
            grid_search_cv(grid=coarse_grid, X=X_dev_cup, y=y_dev_cup, k=5, results_path="csvs/coarse_gs.csv")
 
    elif (script_name == "cup-fine_gs"):
        # perform the fine grid search for the cup
        X_dev_cup, y_dev_cup = load_dev_set_cup()
        with open("jsons/grid_searches.json") as json_gs:
            coarse_grid = ParameterGrid(json.load(json_gs).get("fine"))
            grid_search_cv(grid=coarse_grid, X=X_dev_cup, y=y_dev_cup, k=5, results_path="csvs/fine_gs.csv")
    
    else:
        raise ValueError(f"Incorrect input. Script name must be one of {AVAILABLE_SCRIPTS}")

if __name__ == "__main__":
    AVAILABLE_SCRIPTS = [
        "monks-1", "monks-2", "monks-3", "monks-3reg", "cup-best_single", "cup-ensemble_pred",
        "cup-custom", "cup-coarse_gs", "cup-fine_gs", "cup-ensemble_assessment", "cup-ensemble_blind_pred",
        "cup-ensemble_cv", "cup-best_models_assessment", "cup-ensemble_final_fit"
        ]
    parser = argparse.ArgumentParser(description='Input an experiment name for ML project 22/23')
    parser.add_argument(
        'script_name', type=str,
        help=f'The name of the experiment. \ Must be one of: {AVAILABLE_SCRIPTS}') 
    args = parser.parse_args()

    main(args.script_name)

    