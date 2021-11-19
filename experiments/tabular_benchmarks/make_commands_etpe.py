#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-08
# @Contact    : qichun.tang@bupt.edu.cn
from pathlib import Path

import click

@click.command()
@click.option("--data_dir", "-d", default="/media/tqc/doc/Project/fcnet_tabular_benchmarks")
@click.option("--n_iters", "-n", default=500)
def main(data_dir, n_iters):
    cmds = []
    pairs = [
        ["ultraopt", "ETPE", "default"],
        # ["ultraopt", "ETPE", "univar"],
        # ["ultraopt", "ETPE", "univar_cat"],
        
        # ["ultraopt", "Random"],
        # ["tpe", ""],
    ]
    for benchmark in ["protein_structure", "slice_localization", "naval_propulsion", "parkinsons_telemonitoring"]:
        for i in range(20):
            for algo, opt, mode in pairs:
                for g in [3,4,5,6,8]:
                    cmd = f"python run_{algo}.py --data_dir={data_dir}  --run_id={i} --n_iters={n_iters} --benchmark {benchmark} --mode {mode} "
                    if opt:
                        cmd += f"--optimizer {opt}"
                    cmd+= f" --max_groups {g} "
                    cmds.append(cmd)
    # Path(f"commands.sh").write_text("\n".join(cmds))
    Path(f"commands_etpe.sh").write_text("\n".join(cmds))


if __name__ == '__main__':
    main()
