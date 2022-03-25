# FedSimulate
Simulation framework for Federated Learning in PyTorch based on this [paper](https://arxiv.org/abs/1812.06127). The distributed devices runs their updates in a sequential fashon so that there's no need of high computational power, although the training might be pretty long in some cases.


<p align="center">
  <img src="/imgs/pool_20_mu_0.005.png"  title="FedProx: pool= 20, mu= 0.005" />
</p>


**Server**: Centralize and unify the clients weights following a _weights generation policy_ (e.g. averaging, weighted averaging, top-k, ...) and then update the client devices with the result. This is performed for _rounds_ turns and at each round _pool_ devices are sampled from the whole set to update the server model.

**Clients**: Each client is defined by a _Model-Free Trainer_ which handle its hyperparamentrs and training procedure. At each round, only the sampled devices have thier models instantiated to optimize the memory consumption. 

**Loner**: This is a stand-alone model trained in parallel for _rounds_ epochs and it is used as a baseline to evaluate the FL algorithm perfomances. It has the complete dataset to train on.

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
