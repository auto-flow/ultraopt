
这个月，笔者复盘了2020年做的一些AutoML项目，并在多个优秀的开源项目的基础上，博采众长，写了一个超参优化库：`UltraOpt`。这个库包含一个笔者自研的贝叶斯优化算法：`ETPE`，其在基准测试中比`HyperOpt`的`TPE`算法表现更为出色。`UltraOpt`对分布式计算有更强的适应性，支持**MapReduce**和**异步通信**两种并行策略，并且可以扩展到各种计算环境中。

除此之外，`UltraOpt`对与新手也特别友好，笔者特地花了3周的时间写中文文档，就是为了让小白也能0基础看懂AutoML（自动机器学习）是在做什么。`UltraOpt`使用起来也是相当的轻量简洁，并且有大量的可视化工具函数帮助您更快地分析问题。

受篇幅所限，本文只能简明扼要地将`UltraOpt`的关键特性介绍出来，文本的所有代码与参考文献都能在[代码仓库的README](https://github.com/auto-flow/ultraopt)和[中文文档](https://auto-flow.github.io/ultraopt/zh/)中找到。如果您觉得这个项目对您有帮助，也欢迎给[UltraOpt](https://github.com/auto-flow/ultraopt)点一个小小的★star 哟～

---

安装方法：

```bash
pip install ultraopt
```

代码仓库：
[https://github.com/auto-flow/ultraopt](https://github.com/auto-flow/ultraopt)

中文文档：
[https://auto-flow.github.io/ultraopt/zh/](https://auto-flow.github.io/ultraopt/zh/)



**目录**

- [快速上手](#快速上手)
	+ [用于超参优化](#用于超参优化)
	+ [考虑算法选择与较少代价的评估策略](#考虑算法选择与较少代价的评估策略)
- [为什么比HyperOpt更强](#为什么比HyperOpt更强)
	+ [从性能上](#从性能上)
	+ [从功能上](#从功能上)

# 快速上手

## 用于超参优化

首先我们通过一个对随机森林进行超参优化的过程来了解`UltraOpt`～

如果要启动一个优化过程，首先我们要提供两个东西：1) 参数空间 2) 目标函数。

首先，定义一个超参空间，在`UltraOpt`中超参空间可以用`HDL`（Hyperparameters Description Language，超参描述语言）来定义：

![HDL](https://img-blog.csdnimg.cn/20210111172610619.png)

然后我们定义一个用于评判超参好坏的目标函数，接受一个字典类型的`config`，返回`loss`，即 `1` 减去交叉验证中验证集的平均正确率。我们的优化目标是 `loss` 越小越好。

![eval](https://img-blog.csdnimg.cn/20210111172639405.png)

在定义了超参空间和目标函数后，进行优化其实就是一行代码的事情，只需要调用`UltraOpt`的`fmin`函数就行了：

```python
from ultraopt import fmin
result = fmin(eval_func=evaluate, config_space=HDL, 
			optimizer="ETPE", n_iterations=30)
print(result)
```

优化过程中会打印进度条，完成后打印`result`对象会对优化结果进行一个汇总。

![result](https://img-blog.csdnimg.cn/20210111173309236.png)

调用`result`对象的`plot_convergence`方法会绘制拟合曲线。

![plot_convergence](https://img-blog.csdnimg.cn/20210111173554453.png)

在安装了Facebook的可视化工具`hiplot`后，你可以查看这次超参优化结果的高维交互图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210111174056527.png)

## 考虑算法选择与较少代价的评估策略

可能有小伙伴要问，如果我想先从一组优化器中选一个优化器（**算法选择**），再对这个优化器进行**超参优化**，`UltraOpt`能解决这样的算法选择与超参优化问题(`CASH Problem`)吗？答案是可以的。首先我们定义一个解决`CASH Problem`的`HDL`：

![HDL2](https://img-blog.csdnimg.cn/20210111174859829.png)

我们看到，首先我们要从`【随机森林，KNN】`这两个优化器中选一个，再对这个优化器进行超参优化。

这时又有小伙伴提问：上个案例中AutoML的评估函数需要进行一次**完整**的训练与预测流程，耗时比较长，换句话说就是代价比较大。有什么方法可以解决这个问题呢？有的大佬就提出了一种简单粗暴的方法：连续减半（Successive Halving，SH），即用少量样本来评价大量超参配置，将表现好的超参配置保留到下一轮迭代中，如图所示：

![](https://img-blog.csdnimg.cn/20201228104418342.png)

后面又有大佬在SH的基础上提出了HyperBand（HB）， 但为了读者朋友理解，今天咱们就以SH为例，让`UltraOpt`采用**较少代价的评估策略** 。

我们在考虑了`评估代价`（或者称为预算，budget）这一影响因素后，需要重新设计我们的目标函数，即添加`budget`参数并做相应的改变：

![eval2](https://img-blog.csdnimg.cn/20210111180455248.png)

这段代码可能有点难理解，简单说做了以下修改：

1. 考虑了评估代价(量化为`budget`，取值范围0到1，表示训练样本采样率)
2. 考虑了算法选择，`AS`表示算法选择结果，`HP`表示超参优化结果。

再实例化用于支持SH的迭代生成器：

![iter](https://img-blog.csdnimg.cn/20210111180815559.png)

现在菜都配齐了，往`ultraopt.fmin`的锅里一扔，炖了：

![result2](https://img-blog.csdnimg.cn/20210111181047745.png)

`UltraOpt`提供了大量的可视化工具函数，您可以查看优化过程与优化结果：


![result2](https://img-blog.csdnimg.cn/20210111182225528.png)

# 为什么比HyperOpt更强

## 从性能上

我们在`Synthetic Benchmark`和`Tabular Benchmark`上对优化算法进行了对比实验，结果如下：

- Synthetic Benchmark

![Synthetic Benchmark](https://img-blog.csdnimg.cn/20210111182852515.png)

- Tabular Benchmark

![Tabular Benchmark](https://img-blog.csdnimg.cn/20210111182852255.png)

- Tabular Benchmark（考虑HyperBand评价策略）

![Tabular Benchmark2](https://img-blog.csdnimg.cn/20210111182852298.png)




## 从功能上

`UltraOpt`的设计哲学是以优化器为中心，优化器与评估器分离，并且是目前所有黑盒优化库中唯一能做到同时支持`TPE`算法和`SMAC`算法的，未来还会支持`MCMC-GP`和`贝叶斯回归`，`贝叶斯网络`等更多的优化器。

并且`UltraOpt`把`multi-fidelity`（也就是HyperBand、连续减半等评估策略）也解耦和了出来，相比于BOHB的原生实现HpBandSter库，你可以在`UltraOpt`中更自由地将多保真度优化与贝叶斯优化进行混搭，创造属于你的BOHB算法。

下图表格从各种角度列出了`UltraOpt`相比于其他优化库的优势：

![compare](https://img-blog.csdnimg.cn/20210111191905240.png)


如果您在使用中遇到了问题，欢迎issue反馈。如果您有idea，也欢迎Pull Request～
