export data_dir=/media/tqc/doc/Project/fcnet_tabular_benchmarks
export PYTHONPATH=/data/Project/AutoML/ultraopt
export n_iters=100
export min_points_in_model=20
export DEBUG=true
export pretrain=true
export scale=true
python run_ultraopt.py --data_dir=$data_dir  --run_id=0 --n_iters=$n_iters --benchmark protein_structure #--max_groups=4
#python run_ultraopt.py --data_dir=$data_dir  --run_id=0 --n_iters=$n_iters --benchmark slice_localization
#python run_ultraopt.py --data_dir=$data_dir  --run_id=0 --n_iters=$n_iters --benchmark naval_propulsion
#python run_ultraopt.py --data_dir=$data_dir  --run_id=0 --n_iters=$n_iters --benchmark parkinsons_telemonitoring

