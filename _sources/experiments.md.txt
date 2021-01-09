# Synthetic Benchmarks

All experimental code are public available in [here](https://github.com/auto-flow/ultraopt/tree/dev/experiments/synthetic) .


We use 9 Synthetic Benchmarks in HPOlib[<sup>\[5\]</sup>](#refer-5), you can find more synthetic function in [here](http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page364.htm).

We compare three optimizers in this section:

1. HyperOpt's TPE[<sup>\[1\]</sup>](#refer-1) optimizer
2. UltraOpt's ETPE optimizer
3. Random optimizer as baseline

<center id="figure-1">Figure 1: Performance over Iterations in Synthetic Benchmarks</center>

![synthetic_benchmarks_log.png](https://gitee.com/TQCAI/ultraopt_img/raw/master/synthetic_benchmarks_log.png)

We can see UltraOpt's ETPE optimizer better than HyperOpt's TPE[<sup>\[1\]</sup>](#refer-1) optimizer in 5/9 times, tied it in 4 cases.

# Tabular Benchmarks

All experimental code are public available in [here](https://github.com/auto-flow/ultraopt/tree/dev/experiments/tabular_benchmarks) .

Tabular Benchmarks[<sup>\[3\]</sup>](#refer-3)  performed an exhaustive search for a large neural architecture search problem and
compiled all architecture and performance pairs into a neural architecture search benchmark. 

Tabular Benchmarks[<sup>\[3\]</sup>](#refer-3) use 4 popular UCI[<sup>\[4\]</sup>](#refer-4) datasets(see [Table 1](#table-1) for an overvie) for regression, and used a two layer feed forward neural network followed by a linear output
layer on top. The configuration space (denoted in [Table 2](#table-2)) only includes a modest number of 4
architectural choice (number of units and activation functions for both layers) and 5 hyperparameters
(dropout rates per layer, batch size, initial learning rate and learning rate schedule) in order to
allow for an exhaustive evaluation of all the 62 208 configurations resulting from discretizing the
hyperparameters as in [Table 2](#table-2). Tabular Benchmarks encode numerical hyperparameters as ordinals and all other
hyperparameters as categoricals.



<center id="table-1">Table 1: Dataset splits</center>

|Dataset | # training datapoints | # validation datapoints | # test datapoints | # features |
|---|---|---|---|---|
|HPO-Bench-Protein | 27 438 | 9 146 |9 146 |9|
|HPO-Bench-Slice | 32 100| 10 700 |10 700| 385|
|HPO-Bench-Naval | 7 160| 2 388 |2 388 |15|
|HPO-Bench-Parkinson | 3 525 |1 175| 1 175 |20|


<center id="table-2">Table 2: Configuration space of the fully connected neural network</center>

|Hyperparameters|Choices|
|---------------|-------|
|Initial LR| {.0005, .001, .005, .01, .05, .1} |
|Batch Size| {8, 16, 32, 64} |
|LR Schedule| {cosine, fix} |
|Activation/Layer 1| {relu, tanh} |
|Activation/Layer 2| {relu, tanh} |
|Layer 1 Size| {16, 32, 64, 128, 256, 512} |
|Layer 2 Size| {16, 32, 64, 128, 256, 512} |
|Dropout/Layer 1| {0.0, 0.3, 0.6} |
|Dropout/Layer 2| {0.0, 0.3, 0.6} |

Based on the gathered data, we compare ours optimization package to other optimizers such as HyperOpt[<sup>\[1\]</sup>](#refer-1) and HpBandSter[<sup>\[2\]</sup>](#refer-2) in two scenario: (1) [Full-Budget Evaluation Strategy](#Full-Budget%20Evaluation%20Strategy) (2) [HyperBand Evaluation Strategy](#HyperBand%20Evaluation%20Strategy).

Tabular Benchmarks[<sup>\[3\]</sup>](#refer-3) is publicly available at [here](https://github.com/automl/nas_benchmarks).

## Full-Budget Evaluation Strategy 

In this section we use the generated benchmarks to evaluate different HPO methods. To mimic the
randomness that comes with evaluating a configuration, in each function evaluation Tabular Benchmarks randomly
sample one of the four performance values.  Tabular Benchmarks do not take the additional overhead of the optimizer into account since it is negligible
compared to the training time of the neural network. 

After each function evaluation we estimate the
incumbent as the configuration with the lowest observed error and compute the regret between the
incumbent and the globally best configuration in terms of test error. Each method that operates on the full budget of 100 epochs was allowed
to perform 200 function evaluations (200 iterations). We performed 20 independent
runs of each method and report the median and the 25th and 90th quantile.

We compare three optimizers in this section:

1. HyperOpt's TPE optimizer
2. UltraOpt's ETPE optimizer
3. Random optimizer as baseline

`HyperOpt's TPE` optimizer's experimental code uses [this](https://github.com/automl/nas_benchmarks/blob/master/experiment_scripts/run_tpe.py).


<center id="figure-2">Figure 2: Performance over Iterations</center>

![tabular_benchmarks.png](https://gitee.com/TQCAI/ultraopt_img/raw/master/tabular_benchmarks.png)


We can see UltraOpt's ETPE optimizer better than HyperOpt's TPE[<sup>\[1\]</sup>](#refer-1) optimizer in 3/4 times, tied it in one case.

## HyperBand Evaluation Strategy 

After evaluate optimizers in full-budget and compare performance over iterations, we want to know how much improvement **HyperBand Evaluation Strategy**[<sup>\[6\]</sup>](#refer-6) can bring and compares optimizers' performance over time.

To obtain a realistic estimate of the wall-clock time
required for each optimizer, we accumulated the stored runtime of each configuration the optimizer evaluated. 

For BOHB[<sup>\[2\]</sup>](#refer-2) and HyperBand[<sup>\[6\]</sup>](#refer-6) we set the minimum budget to 3 epochs, the maximum budget to 100,  $\eta$ to 3 and the number of successive halving iterations to 250.

You can view iterations table by entering following code in IPython:

```python
In [1]: from ultraopt.multi_fidelity import HyperBandIterGenerator
In [2]: HyperBandIterGenerator(min_budget=3, max_budget=100, eta=3)
Out[2]: 
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">iter 0</th>
      <th colspan="3" halign="left">iter 1</th>
      <th colspan="2" halign="left">iter 2</th>
      <th>iter 3</th>
    </tr>
    <tr>
      <th></th>
      <th>stage 0</th>
      <th>stage 1</th>
      <th>stage 2</th>
      <th>stage 3</th>
      <th>stage 0</th>
      <th>stage 1</th>
      <th>stage 2</th>
      <th>stage 0</th>
      <th>stage 1</th>
      <th>stage 0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>num_config</th>
      <td>27</td>
      <td>9</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>budget</th>
      <td>3.70</td>
      <td>11.11</td>
      <td>33.33</td>
      <td>100.00</td>
      <td>11.11</td>
      <td>33.33</td>
      <td>100.00</td>
      <td>33.33</td>
      <td>100.00</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>

In addition to the three optimizers described above, three optimizers are added in this section:

1. HpBandster's BOHB optimizer[<sup>\[2\]</sup>](#refer-2)
2. UltraOpt's BOHB optimier
3. HyperBand as baseline


`HpBandster's BOHB` optimizer's experimental code uses [this](https://github.com/automl/nas_benchmarks/blob/master/experiment_scripts/run_bohb.py): 



`UltraOpt's BOHB` optimizer is implemented in following code:

```python
iter_generator = HyperBandIterGenerator(min_budget=3, max_budget=100, eta=3)
fmin_result = fmin(objective_function, cs, optimizer="ETPE",
                    multi_fidelity_iter_generator=iter_generator)
```

`HyperBand` optimizer is implemented in following code:

```python
iter_generator = HyperBandIterGenerator(min_budget=3, max_budget=100, eta=3)
fmin_result = fmin(objective_function, cs, optimizer="Random",
                    multi_fidelity_iter_generator=iter_generator)
```




<center id="figure-3">Figure 3: Performance over Time (Protein Structure)</center>

![protein_structure_HB.png](https://gitee.com/TQCAI/ultraopt_img/raw/master/protein_structure_HB.png)


First, let's draw some conclusions from `Protein Structure` dataset's benchmarks:

- HyperBand achieved a reasonable performance relatively quickly but only slightly improves over simple Random Search eventually.
- BOHB is in the beginning as good as HyperBand but starts outperforming it as soon as it obtains a meaningful model. 
- UltraOpt's BOHB is better than HpBandSter's BOHB .

<center id="figure-4">Figure 4: Performance over Time (Slice Localization)</center>

![slice_localization_HB.png](https://gitee.com/TQCAI/ultraopt_img/raw/master/slice_localization_HB.png)

<center id="figure-5">Figure 5: Performance over Time (Naval Propulsion)</center>

![naval_propulsion_HB.png](https://gitee.com/TQCAI/ultraopt_img/raw/master/naval_propulsion_HB.png)

<center id="figure-6">Figure 6: Performance over Time (Parkinsons Telemonitoring)</center>

![parkinsons_telemonitoring_HB.png](https://gitee.com/TQCAI/ultraopt_img/raw/master/parkinsons_telemonitoring_HB.png)



-------

**Reference**

<div id="refer-1"></div>

[1] [James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. 2011. Algorithms for hyper-parameter optimization. In Proceedings of the 24th International Conference on Neural Information Processing Systems (NIPS'11). Curran Associates Inc., Red Hook, NY, USA, 2546–2554.](https://dl.acm.org/doi/10.5555/2986459.2986743)

<div id="refer-2"></div>

[2] [Falkner, Stefan et al. “BOHB: Robust and Efficient Hyperparameter Optimization at Scale.” ICML (2018).](https://arxiv.org/abs/1807.01774)

<div id="refer-3"></div>

[3] [Klein, A. and F. Hutter. “Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization.” ArXiv abs/1905.04970 (2019): n. pag.](https://arxiv.org/abs/1905.04970)

<div id="refer-4"></div>

[4] [Lichman, M. (2013). UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets.php)

<div id="refer-5"></div>

[5] https://github.com/automl/HPOlib1.5/tree/development

<div id="refer-6"></div>

[6] [Li, L. et al. “Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.” J. Mach. Learn. Res. 18 (2017): 185:1-185:52.](https://arxiv.org/abs/1603.06560)