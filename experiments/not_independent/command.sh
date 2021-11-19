#!/usr/bin/env bash
export PYTHONPATH=/data/Project/AutoML/ultraopt
#python run_ultraopt_levy.py levy 3
#python run_ultraopt_levy.py levy 4
#python run_ultraopt_levy.py levy 5
#python run_ultraopt_levy.py levy 6
#python run_ultraopt_levy.py levy 8

#python run_ultraopt_levy.py rosenbrock 3 true false
#python run_ultraopt_levy.py rosenbrock 4
#python run_ultraopt_levy.py rosenbrock 5
#python run_ultraopt_levy.py rosenbrock 6
#python run_ultraopt_levy.py rosenbrock 8
#python run_hyperopt_levy.py rosenbrock

#python run_hyperopt_levy.py levy

# multivariate=False, optimize_each_varGroups=true # 目的: 看能否和hyperopt打平
#python run_ultraopt_levy.py levy 0 false true
#python run_ultraopt_levy.py levy 0 false false
#python run_optuna_levy.py levy true
python run_optuna_levy.py rosenbrock true
#python run_ultraopt_levy.py rosenbrock 0 true false
#python run_random_levy.py rosenbrock
#python run_bohb_kde.py

