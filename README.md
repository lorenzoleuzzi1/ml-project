# Machine Learning (ML) project
Final project for the Machine Learning course at University of Pisa, a.y. 2022/23. The project consists in implementing Neural Networks.

### Installation and Usage ###
`pip install -r requirements.txt`

The application allows to run different trials using a basic command-line interface. To do so you need to type the following command `python main.py <experiment_name>` where `experiment_name` must be one of the following:
- `monks-1`: to fit and test a network with the monks-1 dataset
- `monks-2`: to fit and test a network with the monks-2 dataset
- `monks-3`: to fit and test a network with the monks-3 dataset
- `monks-3reg`: to fit and test a network with the monks-3 dataset using L2 regularization
- `cup-best_gs`: to fit and test a network with the cup dataset using the best configuration found after the fine grid search
- `cup-custom`: to fit and test a network with the cup dataset using a custom configuration properly defined in the cup_custom_config.json file
- `cup-ensemble_pred`: to load the final ensembled model and make the predictions on the cup blind test set
- `cup-coarse_gs`: to perform the coarse grid search with the cup dataset
- `cup-fine_gs`: to perform the fine grid search with the cup dataset

### Directory Structure
```
ml-project
