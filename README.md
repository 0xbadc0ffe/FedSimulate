# FedSimulate
Simulation framework for Federated Learning in PyTorch. The distributed devices runs their updates in a sequential fashon so that there's no need of high computational power, although the training might be pretty long in some cases.


<p align="center">
  <img src="/imgs/pool_20_mu_0.005.png"  title="FedProx: pool= 20, mu= 0.005" />
</p>

## Imbalanced Data Distribution

The clients datasets are made imbalanced by the following distribution over the classes (in this case is CIFAR10 dataset):

<p align="center">
  <img src="/imgs/data_distribution.gif" width=50% height=50% center >
</p>


<p float="left">
    <img src="/imgs/data_distr_alpha0,09_beta0.png" width="500" title="alpha = 0.09 | beta = 0" />
  <img src="/imgs/data_distr_alpha0,02_beta0.png" width="500" title="alpha = 0.02 | beta = 0" />
</p>


<p float="left">
  <img src="/imgs/data_distr_alpha0,09_beta0,2.png" width="500" title="alpha = 0.09 | beta = 0.2" />
  <img src="/imgs/data_distr_alpha0,09_beta-0,2.png" width="500" title="alpha = 0.09 | beta = -0.2" />
</p>

Obviously, the classes are shuffled before applying this distributions. An example is presented in the following plot:  

<p align="center">
  <img src="/imgs/data_distr_alpha0,02_beta0_shuffled.png"  width="500" title="alpha = 0.02 | beta = 0" />
</p>


## Results
 
<p align="center">
  <img src="/imgs/accuracy_vs_alpha.png" title="Experiment1: Accuracy vs alpha" />
</p>

<p align="center">
  <img src="/imgs/accuracy_vs_mu.png" title="Experiment2: Accuracy vs mu" />
</p>
