export data_dir=/home/tqc/fcnet_tabular_benchmarks
export PYTHONPATH=/data/Project/AutoML/ultraopt
export n_iters=2000
export min_points_in_model=1500
export DEBUG=true
export scale=true
export ORD_EMB_REG=cos
python run_ultraopt_produce_emb.py --data_dir=$data_dir  --run_id=0 --n_iters=$n_iters --benchmark protein_structure
#python run_ultraopt.py --data_dir=$data_dir  --run_id=0 --n_iters=$n_iters --benchmark slice_localization
#python run_ultraopt.py --data_dir=$data_dir  --run_id=0 --n_iters=$n_iters --benchmark naval_propulsion
#python run_ultraopt.py --data_dir=$data_dir  --run_id=0 --n_iters=$n_iters --benchmark parkinsons_telemonitoring

