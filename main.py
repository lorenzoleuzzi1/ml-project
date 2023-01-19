import argparse
from monks import run_monks
from utils import read_csv_results
from cup import run_cup
import json
from sklearn.model_selection import ParameterGrid
from cup import load_dev_set_cup
from validation import grid_search_cv

def main(experiment_name):
    if (experiment_name in AVAILABLE_EXPERIMENTS[:4]):
        with open("monks.json") as json_monks:
            monks_config = json.load(json_monks).get(experiment_name)
            monks_config['name'] = experiment_name
            run_monks(monks_config)
    
    elif (experiment_name == "cup-best_gs"):
        best_gs_config = read_csv_results("fine_csv.csv")['params'][0]
        run_cup(best_gs_config)
    
    elif (experiment_name == "cup-custom"):
        custom_config = json.load("cup_custom_config.json")
        run_cup(custom_config)
    
    elif (experiment_name == "cup-ensemble_pred"):
        pass
    
    elif (experiment_name == "cup-coarse_gs"):
        X_dev_cup, y_dev_cup = load_dev_set_cup()
        with open("grid_searches.json") as json_gs:
            coarse_grid = ParameterGrid(json.load(json_gs).get("coarse"))
            grid_search_cv(grid=coarse_grid, X=X_dev_cup, y=y_dev_cup, k=5, results_path="coarse_gs.csv", evaluation_metric='mse')
 
    elif (experiment_name == "cup-fine_gs"):
        X_dev_cup, y_dev_cup = load_dev_set_cup()
        with open("grid_searches.json") as json_gs:
            coarse_grid = ParameterGrid(json.load(json_gs).get("fine"))
            grid_search_cv(grid=coarse_grid, X=X_dev_cup, y=y_dev_cup, k=5, results_path="fine_gs.csv", evaluation_metric='mse')
    
    else:
        raise ValueError(f"Incorrect input. Experiment name must be one of {AVAILABLE_EXPERIMENTS}")

if __name__ == "__main__":
    AVAILABLE_EXPERIMENTS = [
        "monks-1", "monks-2", "monks-3", "monks-3reg", "cup-best_gs", "cup-ensemble_pred",
        "cup-custom", "cup-coarse_gs", "cup-fine_gs"
        ]
    parser = argparse.ArgumentParser(description='Input an experiment name for ML project 22/23')
    parser.add_argument(
        'experiment_name', type=str,
        help=f'The name of the experiment. \ Must be one of: {AVAILABLE_EXPERIMENTS}') 
    args = parser.parse_args()

    main(args.experiment_name)

    