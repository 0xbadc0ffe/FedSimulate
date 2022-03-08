from __future__ import print_function, division

from typing import OrderedDict, Union, Optional, Callable, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm, trange
from torchsummary import summary
import random
import copy
import hashlib

import ssl
ssl._create_default_https_context = ssl._create_unverified_context




def set_reproducibility(seed=42):
    # reproducibility stuff
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True  # Note that this Deterministic mode can have a performance impact
    torch.backends.cudnn.benchmark = False


# Count the number of parameters
def count_parameters(model: torch.nn.Module) -> int:
  """ Counts the number of trainable parameters of a module
  
  :param model: model that contains the parameters to count
  :returns: the number of parameters in the model
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Credits: https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


# Check also this: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/12
def reset_model_params(model):
    for layer in get_children(model):
        layer.reset_parameters()


def print_model(model: torch.nn.Module, dim) -> None:
    # print([i for i in model.children()][-3:]) # residual layer, pooling and our linear layer

    # print(model.layer1[0].conv1) 
    '''
    for name, layer in model.named_modules():
        #if isinstance(layer, nn.ReLU):  # to filter layers
        #    print(name, layer)
        print("\n", name, layer, count_parameters(layer))
    '''
    #summary(residual_net, image_datasets['train'][0][0].shape, batch_size=-1)
    summary(model, dim, batch_size=-1)


def get_CIFARloaders(batch_size=128, batch_size_val=1000, data_transform: Union[str, bool]="RGB", ret_datasets=False) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    if data_transform:
        image_transforms_gray = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.47,), std=(0.251,)),
            ]
        )

        image_transforms_RGB = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.47,), std=(0.251,)),
            ]
        )
        if data_transform == "RGB":
            image_transforms = image_transforms_RGB
        elif data_transform == "GRAY":
            image_transforms = image_transforms_gray
    else:
        image_transforms = None

    train_dataset = datasets.CIFAR10(
            "../../data",
            train=True,
            download=True,
            transform=image_transforms
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


    test_dataset = datasets.CIFAR10(
        "../../data",
        train=False,
        transform=image_transforms
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_val,
        shuffle=True,
    )
    
    if ret_datasets:
        return train_loader, test_loader, train_dataset, test_dataset
    
    else:
        return train_loader, test_loader        


def get_img_dim(train_loader):
    # Retrieve the image size and the number of color channels
    x, _ = next(iter(train_loader))
    n_channels = x.shape[1]
    input_size_w = x.shape[2]
    input_size_h = x.shape[3]
    #input_size = input_size_w * input_size_h
    return n_channels, input_size_w, input_size_h



# Specify the number of classes in CIFAR10
CIFAR10_output_size = 10  # there are 10 classes
CIFAR10_output_classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck')


# Credits: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
# takes in a module and applies the specified weight initialization
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
    values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)




class MLP(nn.Module):
    def __init__(
        self, input_size: int, input_channels: int, n_hidden: int, output_size: int
    ) -> None:
        """
        Simple MLP model

        :param input_size: number of pixels in the image
        :param input_channels: number of color channels in the image
        :param n_hidden: size of the hidden dimension to use
        :param output_size: expected size of the output
        """
        super().__init__()
        self.name= "MLP"
        self.network = nn.Sequential(
            nn.Linear(input_size * input_channels, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]

        :returns: predictions with size [batch, output_size]
        """
        x = x.view(x.shape[0], -1)
        o = self.network(x)
        return o


class CNN(nn.Module):
    def __init__(
        self, input_size: int, input_channels: int, n_feature: int, output_size: int
    ) -> None:
        """
        Simple model that uses convolutions

        :param input_size: number of pixels in the image
        :param input_channels: number of color channels in the image
        :param n_feature: size of the hidden dimensions to use
        :param output_size: expected size of the output
        """
        super().__init__()
        self.name="CNN"
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=n_feature, kernel_size=3
        )
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=3)
        self.conv3 = nn.Conv2d(n_feature, n_feature, kernel_size=3)
        self.conv4 = nn.Conv2d(n_feature, n_feature, kernel_size=2)

        self.fc1 = nn.Linear(n_feature * 5 * 5, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, 
                x: torch.Tensor, 
                return_conv1: bool = False, 
                return_conv2: bool = False, 
                return_conv3: bool = False,
                return_conv4: bool = False
        ) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]
        :param return_conv1: if True return the feature maps of the first convolution
        :param return_conv2: if True return the feature maps of the second convolution
        :param return_conv3: if True return the feature maps of the third convolution

        :returns: predictions with size [batch, output_size]
        """
        x = self.conv1(x)
        if return_conv1:
            return x
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        if return_conv2:
            return x
        x = F.relu(x)

        # Not so easy to keep track of shapes... right?
        # An useful trick while debugging is to feed the model a fixed sample batch
        # and print the shape at each step, just to be sure that they match your expectations.

        # print(x.shape)

        x = self.conv3(x)
        if return_conv3:
            return x
        x = F.relu(x)
        #x = F.max_pool2d(x, kernel_size=2) # comment if add conv4

        
        x = self.conv4(x)
        if return_conv4:
            return x
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def permute_pixels(images: torch.Tensor, perm: Optional[torch.Tensor]) -> torch.Tensor:
    """ Permutes the pixel in each image in the batch

    :param images: a batch of images with shape [batch, channels, w, h]
    :param perm: a permutation with shape [w * h]

    :returns: the batch of images permuted according to perm
    """
    if perm is None:
        return images

    batch_size = images.shape[0]
    n_channels = images.shape[1]
    w = images.shape[2]
    h = images.shape[3]
    images = images.view(batch_size, n_channels, -1)
    images = images[..., perm]
    images = images.view(batch_size, n_channels, w, h)
    return images

def make_averager() -> Callable[[Optional[float]], float]:
    """ Returns a function that maintains a running average

    :returns: running average function
    """
    count = 0
    total = 0

    def averager(new_value: Optional[float]) -> float:
        """ Running averager

        :param new_value: number to add to the running average,
                          if None returns the current average
        :returns: the current average
        """
        nonlocal count, total
        if new_value is None:
            return total / count if count else float("nan")
        count += 1
        total += new_value
        return total / count

    return averager

def test_model(
    test_dl: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    perm: Optional[torch.Tensor] = None,
    device: str = "cuda",
    mu:float = 0,
    init_dict:OrderedDict = None
) -> Dict[str, Union[float, Callable[[Optional[float]], float]]]:
    """Compute model accuracy on the test set

    :param test_dl: the test dataloader
    :param model: the model to train
    :param perm: if not None, permute the pixel in each image according to perm

    :returns: computed accuracy
    """
    model.eval()
    test_loss_averager = make_averager()  # mantain a running average of the loss
    correct = 0
    for data, target in test_dl:
        # send to device
        data, target = data.to(device), target.to(device)

        if perm is not None:
            data = permute_pixels(data, perm)

        output = model(data)
        if init_dict is not None:
            test_loss_averager(F.cross_entropy(output, target)+ mu*sum(torch.norm(x - y) for x, y in zip(model.state_dict().values(), init_dict.values())))
        else:
            test_loss_averager(F.cross_entropy(output, target))

        # get the index of the max probability
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).cpu().sum().item()

    return {
        "accuracy": 100.0 * correct / len(test_dl.dataset),
        "loss_averager": test_loss_averager,
        "correct": correct,
    }


def fit(
    epochs: int,
    train_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    tag: str,
    scheduler: torch.optim.lr_scheduler = None,
    perm: Optional[torch.Tensor] = None,
    device: str = "cuda",
    models_accuracy: dict={},
    mu:float = 0,
) -> float:
    """Train the model and computes metrics on the test_loader at each epoch

    :param epochs: number of epochs
    :param train_dl: the train dataloader
    :param test_dl: the test dataloader
    :param model: the model to train
    :param opt: the optimizer to use to train the model
    :param tag: description of the current model
    :param perm: if not None, permute the pixel in each image according to perm

    :returns: accucary on the test set in the last epoch
    """

    init_dict = copy.deepcopy(model.state_dict())

    for epoch in trange(epochs, desc="train epoch"):
        model.train()
        train_loss_averager = make_averager()  # mantain a running average of the loss

        # TRAIN
        tqdm_iterator = tqdm(
            enumerate(train_dl),
            total=len(train_dl),
            desc=f"batch [loss: None]",
            leave=False,
        )
        for batch_idx, (data, target) in tqdm_iterator:
            # send to device
            data, target = data.to(device), target.to(device)

            if perm is not None:
                data = permute_pixels(data, perm)
                #data = permute_pixels(data, torch.randperm(input_size)) # My code, see execise on independet permutations

            output = model(data)
            # NB: model.parametrs() contains also the grad informations, state_dict doesn't
            loss = mu/2*sum(torch.pow(torch.norm(x - y),2) for x, y in zip(model.parameters(), init_dict.values()))
            loss = loss + F.cross_entropy(output, target) 
            loss.backward()
            opt.step()
            opt.zero_grad()

            train_loss_averager(loss.item())

            tqdm_iterator.set_description(
                f"train batch [avg loss: {train_loss_averager(None):.3f}]"
            )
            tqdm_iterator.refresh()
        

        # TEST
        test_out = test_model(test_dl, model, perm, device, mu=mu, init_dict=init_dict)

        print(
            f"Epoch: {epoch}\n"
            f"Train set: Average loss: {train_loss_averager(None):.4f}\n"
            f"Test set: Average loss: {test_out['loss_averager'](None):.4f}, "
            f"Accuracy: {test_out['correct']}/{len(test_dl.dataset)} "
            f"({test_out['accuracy']:.0f}%)\n"
        )

        if scheduler is not None:
            scheduler.step()
            #print(scheduler.get_last_lr())

    models_accuracy[tag] = test_out['accuracy']
    return test_out['accuracy'], test_out["loss_averager"](None).detach()


def get_model_optimizer(model: torch.nn.Module, opt_name:str = "Adam", lr:float = 0.001, momentum:float = 0.1) -> torch.optim.Optimizer:
    """
    Encapsulate the creation of the model's optimizer, to ensure that we use the
    same optimizer everywhere

    :param model: the model that contains the parameter to optimize

    :returns: the model's optimizer
    """
    if opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # lr initially was 0.001
    elif opt_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-5) #lr initially was 0.01
    else:
        raise Exception("Undefined Optimizer name")




class Trainer():

    # TODO: add more hyperparameters

    def __init__(self, model, data_loaders:Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]=None, train_dict={"epochs":10, "output_dim":10, "opt_name":"Adam", "lr":0.003}):
        self.model = model

        try: 
            if train_dict["device"] is None:
                # Define the device to use: use the gpu runtime if possible!
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:   
                self.device = train_dict["device"] 
        except KeyError:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)

        if data_loaders is None:                              # Data loaders
            try:
                self.batch_size=train_dict["batch_size"]
            except KeyError:
                raise Exception("train_dict must includ batch_size when Trainer object is not instaciated with a tuple of data_loaders")
            try:
                self.batch_size_val=train_dict["batch_size_val"]
            except KeyError:
                raise Exception("train_dict must includ batch_size_val when Trainer object is not instaciated with a tuple of data_loaders")
            try:
                self.data_transform=train_dict["data_transform"]
            except KeyError:
                self.data_transform: Union[str, bool]="RGB"
                
            data_loaders = get_CIFARloaders(self.batch_size, self.batch_size_val, self.data_transform)
        
        self.train_dl = data_loaders[0]
        self.test_dl = data_loaders[1]
        
        try:
            self.epochs = train_dict["epochs"]                # Define the number of the epochs
        except KeyError:
            raise Exception("train_dict must includ epochs")


        ''' Needed only if we want to build the model inside this class
        try:
            self.n_hidden = train_dict["n_hidden"]            # Number of hidden units for the MLP
        except KeyError:
            if self.model.name == "MLP":
                raise Exception("train_dict must includ n_hidden for MLP models")
        try:
            self.n_feature = train_dict["n_feature"]        # Number of the feature maps in the CNN
        except KeyError:
            if self.model.name == "CNN":
                raise Exception("train_dict must includ n_features for CNN models")       
        '''
        if self.model.name == "MLP":
            self.n_hidden = self.model.n_hidden
        elif self.model.name == "CNN":
            self.n_feature = self.model.n_feature


        try:
            self.opt_name = train_dict["opt_name"]            # Oprimizer
        except KeyError:
            self.opt_name = "Adam"
        try:
            self.lr = train_dict["lr"]                        # Learning rate
        except KeyError:
            self.lr = 0.003
        try:
            self.momentum = train_dict["momentum"]            # Optimizer momentum
        except KeyError:
            self.momentum = 0.1    
        try:                                                  # Scheduler boolean
            self.scheduler_bool = train_dict["scheduler_bool"] 
        except KeyError:
            self.scheduler_bool = False   
        try:                                                  # Scheduler decay gamma
            self.gamma = train_dict["gamma"]
        except KeyError:
            self.gamma = 0.9
        try:
            self.output_dim = train_dict["output_dim"]        # Output Dimension
        except KeyError:
            raise Exception("train_dict must includ output_dim") 
    
        self.models_accuracy = {}
        self.opt = get_model_optimizer(self.model, opt_name=self.opt_name, lr=self.lr, momentum=self.momentum)
        if self.scheduler_bool:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=self.gamma)
        else:
            self.scheduler = None

        try:
            self.perm = train_dict["perm"]                    # Permutation in dataset
        except KeyError:
            self.perm = None

        try:
            self.mu = train_dict["mu"]                        # Mu value in FedProx
        except KeyError:
            self.mu = 0

        self.rounds_completed = 0                              # rounds completed

    # TODO: add all other features to this map
    def __str__(self) -> str:
        st =  f"\nModel: {self.model.name}\nDevice: {self.device}\nEpochs: {self.epochs}\n"
        if self.model.name == "MLP":
            st = st + f"Hidden Dim: {self.n_hidden}\n"
        elif self.model.name == "CNN":
            st = st + f"Feature Maps: {self.n_feature}\n"
        return st + f"Optimizer: {self.opt_name}\nLearning Rate: {self.lr}\nScheduler: {self.scheduler_bool}\nBatch size (train | val) = {self.batch_size} | {self.batch_size_val}\nNumber of parameters: {count_parameters(self.model)}\n"


    def fit(self) -> float:
        return fit(
            epochs= self.epochs,
            train_dl=self.train_dl,
            test_dl=self.test_dl,
            model=self.model,
            opt=self.opt,
            tag=self.model.name,
            scheduler=self.scheduler, 
            perm=None,    # TODO: add
            device=self.device,
            models_accuracy=self.models_accuracy,
            mu=self.mu
        )

    def test(self, model):
        return test_model(self.test_dl, model, self.perm, self.device)

    # round: number of epochs in one round of FedAvg/Prox
    # checkpoint: state_dict from server
    def round_fit_from_checkpoint(self, checkpoint, round:int=1) -> float:
        self.round=round
        self.model.load_state_dict(checkpoint)
        acc = fit(
            epochs= self.round,
            train_dl=self.train_dl,
            test_dl=self.test_dl,
            model=self.model,
            opt=self.opt,
            tag=self.model.name,
            scheduler=self.scheduler, 
            perm=None,    # TODO: add
            device=self.device,
            models_accuracy=self.models_accuracy,
            mu=self.mu
        )
        self.rounds_completed += 1
        return acc

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


# Model Free Trainer
# when we emulate multiple devices there is no need to instantiate all the models
# each device is defined by its MFTrainer instance, state_dict and 
# other FedProx parameters (like pick probability pk)
# We then use only one model with N different state_dicts 
# (AVG operations and similar can be made directly on those).
class MFTrainer():

    # TODO: add more hyperparameters

    def __init__(self, data_loaders:Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]=None, train_dict={"epochs":10, "output_dim":10, "opt_name":"Adam", "lr":0.003}):

        try: 
            if train_dict["device"] is None:
                # Define the device to use: use the gpu runtime if possible!
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:   
                self.device = train_dict["device"] 
        except KeyError:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

        if data_loaders is None:                              # Data loaders
            try:
                self.batch_size=train_dict["batch_size"]
            except KeyError:
                raise Exception("train_dict must includ batch_size when Trainer object is not instaciated with a tuple of data_loaders")
            try:
                self.batch_size_val=train_dict["batch_size_val"]
            except KeyError:
                raise Exception("train_dict must includ batch_size_val when Trainer object is not instaciated with a tuple of data_loaders")
            try:
                self.data_transform=train_dict["data_transform"]
            except KeyError:
                self.data_transform: Union[str, bool]="RGB"
            data_loaders = get_CIFARloaders(self.batch_size, self.batch_size_val, self.data_transform)
        
        self.train_dl = data_loaders[0]
        self.test_dl = data_loaders[1]
        
        try:
            self.epochs = train_dict["epochs"]                # Define the number of the epochs
        except KeyError:
            raise Exception("train_dict must includ epochs")

        try:
            self.opt_name = train_dict["opt_name"]            # Oprimizer
        except KeyError:
            self.opt_name = "Adam"
        try:
            self.lr = train_dict["lr"]                        # Learning rate
        except KeyError:
            self.lr = 0.003
        try:
            self.momentum = train_dict["momentum"]            # Optimizer momentum
        except KeyError:
            self.momentum = 0.1    
        try:                                                  # Scheduler boolean
            self.scheduler_bool = train_dict["scheduler_bool"] 
        except KeyError:
            self.scheduler_bool = False   
        try:                                                  # Scheduler decay gamma
            self.gamma = train_dict["gamma"]
        except KeyError:
            self.gamma = 0.9
        try:
            self.output_dim = train_dict["output_dim"]        # Output Dimension
        except KeyError:
            raise Exception("train_dict must includ output_dim") 

        try:
            self.perm = train_dict["perm"]                    # Permutation in dataset
        except KeyError:
            self.perm = None

        try:
            self.mu = train_dict["mu"]                        # Mu value in FedProx
        except KeyError:
            self.mu = 0

        self.opt = None         
        self.models_accuracy = {}
        self.scheduler = None
        self.rounds_completed = 0

    # TODO: add all other features to this map
    def __str__(self) -> str:
        st =  f"\nDevice: {self.device}\nEpochs: {self.epochs}\n"
        return st + f"Optimizer: {self.opt_name}\nLearning Rate: {self.lr}\nScheduler: {self.scheduler_bool}\nBatch size (train | val) = {self.batch_size} | {self.batch_size_val}\nNumber of parameters: {count_parameters(self.model)}\n"

    def set_optimizer(self, model):
        self.opt = get_model_optimizer(model, opt_name=self.opt_name, lr=self.lr, momentum=self.momentum)
        if self.scheduler_bool:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=self.gamma)
        else:
            self.scheduler = None

    def fit(self, model) -> float:
        self.set_optimizer(model)
        try:
            tag = model.name
        except:
            tag = model.__class__.__name__
        return fit(
            epochs= self.epochs,
            train_dl=self.train_dl,
            test_dl=self.test_dl,
            model=model,
            opt=self.opt,
            tag=tag,
            scheduler=self.scheduler, 
            perm=self.perm,
            device=self.device,
            models_accuracy=self.models_accuracy,
            mu=self.mu
        )

    def test(self, model):
        test_out =  test_model(self.test_dl, model, self.perm, self.device)
        st = f"Test set: Average loss: {test_out['loss_averager'](None):.4f}, "
        st += f"Accuracy: {test_out['correct']}/{len(self.test_dl.dataset)} "
        st += f"({test_out['accuracy']:.0f}%)"
        return test_out, st

    # round: number of epochs in one round of FedAvg/Prox
    # checkpoint: state_dict from server (if None use its checkpoint)
    def round_fit_from_checkpoint(self, model, checkpoint=None, round:int=1, reset_optimizer=False) -> float:

        if reset_optimizer:
            self.set_optimizer(model)
        else:
            if self.opt is None:
                self.set_optimizer(model)

        self.round=round
        if checkpoint is not None:
            model.load_state_dict(checkpoint)
        try:
            tag = model.name
        except:
            tag = model.__class__.__name__
        acc = fit(
            epochs= self.round,
            train_dl=self.train_dl,
            test_dl=self.test_dl,
            model=model,
            opt=self.opt,
            tag=tag,
            scheduler=self.scheduler, 
            perm=self.perm,    
            device=self.device,
            models_accuracy=self.models_accuracy,
            mu=self.mu
        )
        self.rounds_completed += 1
        return acc




def make_server_dict(wk_list):
    if wk_list is None or len(wk_list) == 0:
        return None
    if len(wk_list) == 1:
        return copy.deepcopy(wk_list[0])

    new = copy.deepcopy(wk_list[0])
    for key in wk_list[0]:
        tot = wk_list[0][key]
        for client_wk in wk_list[1:]:
            tot = tot + client_wk[key]
        new[key] = tot/len(wk_list)
    return new
        

def model_hash(model):
    return hashlib.md5(str(model.state_dict()).encode('utf-8')).hexdigest()

def state_hash(state_dict):
    return hashlib.md5(str(state_dict).encode('utf-8')).hexdigest()



