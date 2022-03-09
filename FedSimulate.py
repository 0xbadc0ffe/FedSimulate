from __future__ import print_function, division
from cProfile import label

from typing import Mapping, Union, Optional, Callable, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm, trange
from torchsummary import summary
import utils_alg

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import platform
import models
import copy

from timeit import default_timer as timer
from datetime import timedelta

import matplotlib.pyplot as plt

def clean():
    plat = platform.system()
    if plat == "Windows":
        os.system("cls")
    else:
        os.system("clear")



class FedDevice():

    def __init__(self, trainer:Union[models.Trainer, models.MFTrainer], state_dict, tag:str, pk:float, mask_weights:torch.tensor=None):
        self.trainer = trainer
        self.state_dict=copy.deepcopy(state_dict)
        self.tag = tag
        self.pk = pk
        self.mask_weights = mask_weights
        self.major_class = self.major_class()

    def __str__(self):
        return f"Device {self.tag} | Rounds Completed: {self.trainer.rounds_completed}"

    def round_fit(self, model):
        acc, loss = self.trainer.round_fit_from_checkpoint(model, checkpoint=self.state_dict)
        self.state_dict = copy.deepcopy(model.state_dict())
        return acc, loss

    def load_state_dict(self, state_dict):
        self.state_dict = copy.deepcopy(state_dict)

    def set_mu(self, mu):
        self.trainer.mu=mu

    def free(self):
        self.state_dict=None

    def major_class(self):
        if self.mask_weights is None or not (self.mask_weights-1).any():
            return None # all weights are 1
        else:
            return torch.argmax(self.mask_weights)


class FedServer():

    def __init__(self, model, trainer, tag:str="server", weights_generator:Union[Callable,str]=None):
        self.model = model
        self.state_dict=model.state_dict()
        self.tag = tag
        self.trainer = trainer
        self.updates_cnt = 0
        if weights_generator is None or weights_generator == "average":
            self.gen_method = self.dicts_avg
            self.weights_generator = "average"
        elif weights_generator == "first":
            self.gen_method = self.dicts_first
            self.weights_generator = "first"
        elif weights_generator == "top-k_avg":
            self.gen_method = self.dicts_top_k_avg
            self.weights_generator = "top-k_avg"
        else:
            self.weights_generator = "custom"
            self.gen_method = weights_generator


    def __str__(self):
        return f"Device {self.tag} | Rounds Completed: {self.updates}"

    def round_fit(self, model):
        acc, loss = self.trainer.round_fit_from_checkpoint(model, checkpoint=self.state_dict)
        self.state_dict = copy.deepcopy(model.state_dict())
        return acc, loss

    def update(self, *args):
        result = self.gen_method(*args)
        self.updates_cnt += 1 
        if self.weights_generator == "custom":
            self.load_state_dict(result)
        
    # Takes the average of the dicts as the new server state dict.
    def dicts_avg(self, wk_list):
        if wk_list is None or len(wk_list) == 0:
            self.model.load_state_dict(self.state_dict)
            return None
        if len(wk_list) == 1:
            self.state_dict = copy.deepcopy(wk_list[0])
            self.model.load_state_dict(self.state_dict)
            return self.state_dict
        # cloning first element in state_dict
        self.state_dict = copy.deepcopy(wk_list[0])
        for key in wk_list[0]:
            tot = wk_list[0][key]
            for client_wk in wk_list[1:]:
                tot = tot + client_wk[key]
            self.state_dict[key] = tot/len(wk_list)
        # cloning result in model_dict
        self.model.load_state_dict(self.state_dict)
        return self.state_dict

    # Pick the first of the list as the new server state dict.
    # If the list is given already ordered by accuracy/loss or wethever this 
    # will be like picking the most fitting trained instance.
    # ALERT: is not advisible to use this when heterogenious clients data is 
    # involved or when the single clients trainings differ.
    # dicts_avg can be also used in this way by giving a singleton list with 
    # the maximal state_dict
    def dicts_first(self, wk_list):
        if wk_list is None or len(wk_list) == 0:
            self.model.load_state_dict(self.state_dict)
            return None
        else:
            self.state_dict = copy.deepcopy(wk_list[0])
            self.model.load_state_dict(self.state_dict)
            return self.state_dict

    def dicts_top_k_avg(self, wk_dict, perform, K):
        if wk_dict is None or len(wk_dict) == 0:
            self.model.load_state_dict(self.state_dict)
            return None
        elif len(wk_dict) == 1:
            self.state_dict = copy.deepcopy(list(wk_dict.values())[0])
            self.model.load_state_dict(self.state_dict)
            return self.state_dict
        else:
            K = max(1,K)
            top_devs = {k: v for k, v in sorted(perform.items(), key=lambda item: item[1])[::-1]}
            top_devs = list(top_devs.keys())[:K]
            top_k_weights = [ wk_dict[tag] for tag in top_devs]
            self.state_dict = copy.deepcopy(top_k_weights[0])
            for key in top_k_weights[0]:
                tot = top_k_weights[0][key]
                for client_wk in top_k_weights[1:]:
                    tot = tot + client_wk[key]
                self.state_dict[key] = tot/len(top_k_weights)  # len(..) can differ by K if top_devs is shorter
            self.model.load_state_dict(self.state_dict)
            return self.state_dict


    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.state_dict = copy.deepcopy(state_dict)

    def test(self):
        return self.trainer.test(self.model)

    def set_mu(self, mu):
        self.trainer.mu=mu


def update_weights(devices_list, server_weights):
    for dev in devices_list:
        dev.load_state_dict(server_weights)

def update_mu(devices_list, mu):
    for dev in devices_list:
        dev.set_mu(mu)

def free_all(devices_list):
    for dev in devices_list:
        dev.free()


models.set_reproducibility()
clean()



##### Dataset

n_channels = 3
input_size_w = 32
input_size_h = 32
input_size = input_size_w*input_size_h



##### Model Hyper params

# Multi Layer Perceptron
# n_hidden = 9
#model = models.MLP(input_size, n_channels, n_hidden, models.CIFAR10_output_size)

# Convolutional Nerual Network
n_features = 12
model = models.CNN(input_size, n_channels, n_features, models.CIFAR10_output_size)



##### Training Hyper params

device = torch.device("cpu") #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dict = {
    "device": device,
    "output_dim": models.CIFAR10_output_size, # 10
    "epochs": 1,
    "batch_size" : 128,
    "batch_size_val" : 1000,
    "data_transform" : "RGB",
    "opt_name" : "Adam",
    "lr": 0.003,
    "momentum": 0.1,
    "scheduler_bool": True,
    "gamma": 0.9,
    #"perm": models.permute_pixels,
    "mu":  0.005         # if mu=0 => FedAvg
}

model.to(device)
test_trainer = False
if test_trainer:
    #trainer = models.Trainer(model=model, train_dict=train_dict) # Model-based version
    trainer = models.MFTrainer(train_dict=train_dict)             # Model-free version
    print(trainer.fit(model))


#### FedAVG/Prox Hyper params

emulated_devices = 200
rounds = 15
train_loner = True
pool = 20    # pool = emulated_devices => FedAvg
p_uniform = pool/emulated_devices   # uniform probability to be choosed
adaptive_mu = False
adaptive_phase = 5

# Synthetic Data Heterogeneity (alpha = beta = 0 homogeneous case)
# Imbalance follows this power law : clip(exp(vals*-alpha*numb_of_classes)-beta, min=0)
alpha = 0.03    # power factor
beta = 0 #0.2   # constant factor

devices_list = []
w_generators = ["average", "first", "top-k_avg"]
weights_generator = w_generators[0]
pick_top_k = pool   # for top-k_avg

fn_list = ["uniform", "normal"]
sample_prob_fn = fn_list[0]

if sample_prob_fn == "uniform":
    sample_prob = lambda : np.random.uniform(0,1)

# Note: this is not so usefull in an unifrom device probability context
elif sample_prob_fn == "normal":
    from scipy.stats import norm
    norm_mean = 0.5
    sigma = 0.3/emulated_devices
    sample_prob = lambda : norm.cdf(np.random.uniform(-4,4)) #np.random.normal(norm_mean, sigma)


# Test Sampling
#utils_alg.test_sampling(pool, emulated_devices, sample_prob)





# Using a single data loaders pair for the homegenous case may improve the 
# performances but it could also not be ideal for some specfic models:
# https://stackoverflow.com/questions/60311307/how-does-one-reset-the-dataloader-in-pytorch 
train_loader, test_loader, train_dataset, test_dataset =  models.get_CIFARloaders(train_dict["batch_size"],train_dict["batch_size_val"],train_dict["data_transform"], ret_datasets=True)
data_loaders = (train_loader, test_loader)


# Server Device Initialization
trainer = models.MFTrainer(data_loaders=data_loaders, train_dict=train_dict) # this is needed only for the testing phase
server = FedServer(model, trainer, tag="server", weights_generator=weights_generator)


if train_loner:
    train_dict_loner = copy.deepcopy(train_dict)
    train_dict_loner["mu"] = 0 
    # Loner Device used for comparison (weights do not update with server)
    trainer = models.MFTrainer(data_loaders=data_loaders, train_dict=train_dict_loner)
    loner =  FedDevice(trainer=trainer, state_dict=server.state_dict, tag="loner", pk=1)


# Initializating devices to emulate
for i in range(emulated_devices):
    # resetting state_dict is not necessary since they are gonna train after a global model update by the server
    # models.reset_model_params(model)         
    # initial_state_dict = model.state_dict()

    # Note: hard_mask=True slows the Decives Initialization but is more realistic (expecially when simulating few devices)
    train_loader, mask_weights = utils_alg.SIP(train_dataset, torch.arange(models.CIFAR10_output_size), train_dict["batch_size"], alpha=alpha, beta=beta, hard_mask=True)
    data_loaders = (train_loader, data_loaders[1])

    trainer = models.MFTrainer(data_loaders=data_loaders, train_dict=train_dict)
    dev =  FedDevice(trainer=trainer, state_dict=None, tag=str(i), pk=p_uniform, mask_weights=mask_weights)
    devices_list.append(dev)
    print(f"Building Federation Clients (devices):  {i}/{emulated_devices}", end="\r")


# Test initial accuracy
test_out, test_string = devices_list[0].trainer.test(model)
init_loss = test_out["loss_averager"](None).detach().numpy()
print("\n\n"+test_string)


# Testing FedAvg
seq_runs = 0            # counts the number of sequential model training (counting the loner device also) 
start_time = timer()    # timer to get the total elapsed time
sampled = []            # store at each round the number of sampled devices (mean should be the pool value)
server_acc = []         # store at each round the server accuracy
mean_client_acc = []    # store at each round the mean of clients' accuracy
server_loss = []        # store at each round the server loss
best_dev = []           # store at each round the client device with best accuracy
tot_masks = torch.zeros(mask_weights.shape) # store the sum of the weights of the different masks

# Initializing accuracy of the untrained model 
server_acc.append(test_out["accuracy"])
mean_client_acc.append(test_out["accuracy"])        

if train_loner:
    loner_loss = []         # store at each round the loner loss
    loner_acc = []          # store at each round the loner device accuracy
    # Initializing accuracy of the untrained model 
    loner_acc.append(test_out["accuracy"])

for round in range(1,rounds+1):
    
    round_weights = {}
    round_sampled_devices = []

    # Sampling phase
    for dev in devices_list:
        if sample_prob() <= dev.pk:
            round_sampled_devices.append(dev)
            tot_masks += dev.mask_weights
    sampled_len = len(round_sampled_devices)
    sampled.append(sampled_len)
    update_weights(round_sampled_devices, server.state_dict)  # more efficient, we update only this round working devices 

    
    print("\n##########################################\n")
    sampled_len = len(round_sampled_devices)
    print(f"\n\n## Round {round}/{rounds} | Selected: {sampled_len}\n")

    # Training
    sum_acc = 0
    max_acc = 0
    bdev = None     # best device tag
    client_perform = {}
    for i, dev in enumerate(round_sampled_devices):
        print(f"Training Client {i+1}/{sampled_len}:\n")
        acc, _ = dev.round_fit(server.model)
        client_perform[dev.tag] = acc
        if acc > max_acc:
            max_acc = acc
            bdev = int(dev.tag)
        sum_acc += acc
        print(str(dev) + f"/{round} | Accuracy: {acc} %    | Major class: {dev.major_class} |  Device hash: {models.state_hash(dev.state_dict)}\n")
        # print(f"\nDevice hash: {models.state_hash(dev.state_dict)}\n")
        print("-----------------------------\n")
        round_weights[dev.tag] = dev.state_dict
        seq_runs += 1

    if sampled_len != 0:
        mean_acc = sum_acc/sampled_len
        best_dev.append(bdev)
    else:
        if len(mean_client_acc)!=0:
            mean_acc = mean_client_acc[-1]
        else:
            mean_acc = sum_acc
    mean_client_acc.append(mean_acc)


    if train_loner:
        # Training the loner
        print(f"Training Loner device:\n")
        acc, lon_loss = loner.round_fit(server.model)
        loner_acc.append(acc)
        loner_loss.append(lon_loss.numpy())
        print(str(loner) + f"/{round} | Accuracy: {acc} %    | Device hash: {models.state_hash(loner.state_dict)}\n")
        print("-----------------------------\n")
        seq_runs+=1

    # Updating server weights    
    if weights_generator == "average":
        server.update(list(round_weights.values()))
    elif weights_generator == "first":
        if bdev is not None:
            server.update([round_weights[str(bdev)]])
    elif weights_generator == "top-k_avg":
        server.update(round_weights, client_perform, pick_top_k)

    # Testing server
    test_out, test_string = server.test()
    server_acc.append(test_out["accuracy"])
    round_server_loss = test_out["loss_averager"](None).detach().numpy()
    server_loss.append(round_server_loss)
    print(f"\n\n** Round {round}/{rounds} completed **\n")
    print("Sever  " + test_string+"\n")
    print(f"Server hash: {models.state_hash(server.state_dict)}")           # must be equal
    print(f"Model hash:  {models.state_hash(server.model.state_dict())}\n") # must be equal

    # Adaptive mu
    if adaptive_mu and round % adaptive_phase == 0:
        if init_loss - round_server_loss > 0:
            update_mu(devices_list, max(0,server.trainer.mu-0.1))
            server.set_mu(max(0,server.trainer.mu-0.1))
        else:
            update_mu(devices_list, max(0,server.trainer.mu+0.1))
            server.set_mu(max(0,server.trainer.mu-0.1))

    # Free (None overwrite) the selected devices state_dict to keep memory occupancy low
    # Since every device has its own copy of state_dict we would end with high memory allocated 
    free_all(round_sampled_devices)


end_time = timer()
print(f"\n\n###########################################\n")
print(f"\nDevices: {emulated_devices}       [ Alpha:   {alpha}  |  Beta: {beta} ]")
print(f"\nAvg pool per round: {sum(sampled)/rounds}    [sample prob fn: {sample_prob_fn} | Expected pool: {pool}]")
print(f"\nFinal Mu:   {server.trainer.mu}    |  Weights generator: {server.weights_generator}")

if weights_generator == w_generators[2]:
    print(f"Avg of top {pick_top_k} clients")
print(f"\nRunned trainings: {seq_runs}   [Rounds: {rounds}]\n")
print(f"Sever        | Rounds completed: {rounds} | Accuracy: {test_out['accuracy']} %      | Device hash: {models.state_hash(server.state_dict)}")
print(f"Clients avg  | Rounds completed: {rounds} | Accuracy: {np.round(mean_client_acc[-1],2)} %      | Device hash:  --- ")
if train_loner:
    print(str(loner) + f" | Accuracy: {acc} %      | Device hash: {models.state_hash(loner.state_dict)}\n")

print(f"Elapsed time: {timedelta(seconds=end_time-start_time)}")

print(f"\n\nTraining Dictionary: {train_dict}")


# Sampled devices per round 
plt.figure(1)
plt.plot(range(1,rounds+1), sampled, label="sampled clients")
plt.plot(range(1,rounds+1), [pool]*len(sampled), label="expected avg")
plt.title(f"Sampled clients [tot: {np.sum(sampled)}]")
plt.xlabel("round")
plt.ylabel(f"#")
plt.legend()

# Accuracy
plt.figure(2)
plt.plot(server_acc, color="red", label="server")
plt.plot(mean_client_acc, color="blue",linestyle="--", label="clients-avg")
if train_loner:
    plt.plot(loner_acc, color="green", label="loner")
plt.title(f"Accuracy")
plt.xlabel("round")
plt.ylabel(f"%")
plt.legend()

# Test Loss (Server vs loner, clients Loss is not comparable)
plt.figure(3)
plt.plot(server_loss, color="red", label="server")
if train_loner:
    plt.plot(loner_loss, color="green", label="loner")
plt.title(f"Test Loss")
plt.xlabel("round")
plt.ylabel(f"Loss")
plt.legend()


# Devices usage histogram
plt.figure(4)
hist_data = []
for dev in devices_list:
    hist_data = hist_data + [int(dev.tag)]*dev.trainer.rounds_completed
plt.hist(hist_data, emulated_devices)
plt.title("Devices usage")
plt.xlabel("Device Tag")
plt.ylabel("Usage")
#plt.legend()

# Best Devices
plt.figure(5)
plt.hist(best_dev, emulated_devices)
plt.title(f"Best Devices per epoch")
plt.xlabel("Device Tag")
plt.ylabel(f"Round Winner counter")
#plt.legend()

# Distribution of Data
plt.figure(6)
plt.plot(tot_masks.numpy()/np.sum(sampled))
plt.title(f"Cumulative clients training data usage per class")
plt.xlabel("Class")
plt.ylabel(f"%")
#plt.legend()


plt.show()

