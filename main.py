import argparse
from monks import run_monks
from validation import read_csv_results
from cup import run_cup
import json

def main(experiment_name):
    if (experiment_name in AVAILABLE_EXPERIMENTS[:3]):
        run_monks(experiment_name)
    
    elif (experiment_name == "cup-best_gs"):
        best_gs_config = read_csv_results("fine_csv.csv")['params'][0]
        run_cup(best_gs_config)
    
    elif (experiment_name == "cup-custom"):
        custom_config = json.load("cup_custom_config.json")
        run_cup(custom_config)
    
    elif (experiment_name == "cup-ensemble_pred"):
        pass
    
    elif (experiment_name == "cup-coarse_gs"):
        # leggi griglia o da un .py o .json
        # X_dev, y_dev = load_dev_set_cup()
        # grid_search_cv(grid=grid, X=X_dev, y=y_dev, k=5, results_path=results_path, evaluation_metric='mse')
        pass
    
    elif (experiment_name == "cup-fine_gs"):
        # leggi griglia o da un .py o .json
        # X_dev, y_dev = load_dev_set_cup()
        # grid_search_cv(grid=grid, X=X_dev, y=y_dev, k=5, results_path=results_path, evaluation_metric='mse')
        pass 
    
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

    