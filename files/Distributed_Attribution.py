#!/usr/bin/env python
# coding: utf-8

# # Distributed Computation of Attributions using Captum

# In this tutorial, we provide some examples of using Captum with the torch.distributed package and DataParallel, allowing computing attributions in a distributed manner across processors, machines or GPUs.
# 
# In the first part of this tutorial, we demonstrate dividing a single batch of inputs and computing attributions for each part of the batch in a separate process or GPU if available using torch.distributed and DataParallel. In the second part of this tutorial, we demonstrate computing attributions over the Titanic dataset in a distributed manner, dividing the dataset among processes and computing the global average attribution.

# ## Part 1: Distributing computation of Integrated Gradients for an input batch

# In this part, our goal is to distribute a small batch of input examples across processes, compute the attributions independently on each process, and collect the resulting attributions. This approach can be very helpful for algorithms such as IntegratedGradients, which internally expand the input, since they can be performed with a larger number of steps when inputs are distributed across devices.

# We will first demonstrate this with torch.distributed and then demonstrate the same computation with DataParallel, which is particularly for distribution across GPUs.

# Initial imports:

# In[1]:


import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.multiprocessing import Process

from captum.attr import IntegratedGradients

# We now define a small toy model for this example.

# In[2]:


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear1.weight = nn.Parameter(torch.ones(4, 3))
        self.linear1.bias = nn.Parameter(torch.tensor([-10.0, 1.0, 1.0, 1.0]))
        self.relu = nn.ReLU()

    def forward(self, x: Tensor):
        return self.relu(self.linear1(x))

# ### torch.distributed Example

# In the following cell, we set parameters USE_CUDA and WORLD_SIZE. WORLD_SIZE corresponds to the number of processes initialized and should be set to either 1, 2, or 4 for this example. USE_CUDA should be set to true if GPUs are available and there must be at least WORLD_SIZE GPUs available.

# In[3]:


USE_CUDA = True
WORLD_SIZE = 4

# We now define the function that runs on each process, which takes the rank (identifier for current process), size (total number of processes), and inp_batch, which corresponds to the input portion for the current process. Integrated Gradients is computed on the given input and concatenated with other processes on the process with rank 0. The model can also be wrapped in Distributed Data Parallel, which synchronizes parameter updates across processes, by uncommenting the corresponding line, but it is not necessary for this example, since no parameters updates / training is conducted.

# In[4]:


#Uncomment the following import and corresponding line in run to test with DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

def run(rank, size, inp_batch):
    # Initialize model
    model = ToyModel()
    
    # Move model and input to device with ID rank if USE_CUDA is True
    if USE_CUDA:
        inp_batch = inp_batch.cuda(rank)
        model = model.cuda(rank)
        # Uncomment line below to wrap with DistributedDataParallel
        model = DDP(model, device_ids=[rank])

    # Create IG object and compute attributions.
    ig = IntegratedGradients(model)
    attr = ig.attribute(inp_batch, target=0)
    
    # Combine attributions from each device using distributed.gather
    # Rank 0 process gathers all attributions, each other process
    # sends its corresponding attribution.
    if rank == 0:
        output_list = [torch.zeros_like(attr) for _ in range(size)]
        torch.distributed.gather(attr, gather_list=output_list)
        combined_attr = torch.cat(output_list)
        # Rank 0 prints the combined attribution tensor after gathering
        print(combined_attr)
    else:
        torch.distributed.gather(attr)

# This function performs required setup and cleanup steps on each process and executes the chosen function (run).

# In[5]:


def init_process(rank, size, fn, inp_batch, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, inp_batch)
    dist.destroy_process_group()

# We are now ready to run the initialize and run the processes. The gathered output attributions are printed by the rank 0 process upon completion.

# In[6]:


size = WORLD_SIZE
processes = []
batch = 1.0 * torch.arange(12).reshape(4,3)
batch_chunks = batch.chunk(size)
for rank in range(size):
    p = Process(target=init_process, args=(rank, size, run, batch_chunks[rank]))
    p.start()
    processes.append(p)

for p in processes:
    p.join()


# To confirm the correctness of the attributions, we can compute the same attributions from a single process and confirm the results match.

# In[7]:


model = ToyModel()
ig = IntegratedGradients(model)

batch = 1.0 * torch.arange(12).reshape(4,3)
print(ig.attribute(batch, target=0))

# ### DataParallel Example

# If GPUs are available, we can also distribute computation using torch.nn.DataParallel instead. DataParallel is a wrapper around a module which internally splits each input batch across available CUDA device, parallelizing computation. Note that DistributedDataParallel is expected to be faster than DataParallel, but DataParallel can be simpler to setup, with only a wrapper around the module. More information regarding comparing the 2 approaches can be found [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

# The same attributions can be computed using DataParallel as follows. Note that this can only be run when CUDA is available.

# In[8]:


dp_model = nn.DataParallel(model.cuda())
ig = IntegratedGradients(dp_model)

print(ig.attribute(batch.cuda(), target=0))

# ## Part 2: Distributing computation of Titanic Dataset Attribution

# In this part, our goal is to distribute a small batch of input examples across processes, compute the attributions independently on each process, and collect the resulting attributions. For this part, make sure that pandas is installed and available.
# 
# NOTE: Please restart your kernel before executing this portion, due to issues with mutliprocessing from Jupyter notebooks.
# 
# Initial Imports:

# In[1]:


import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Process

from captum.attr import IntegratedGradients

# Download the Titanic dataset from: http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv. 
# Update path to the dataset here.

# In[2]:


dataset_path = "titanic3.csv"

# We define a simple neural network architecture, which is trained in the Titanic tutorial.

# In[3]:


class TitanicSimpleNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12, 12)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(12, 8)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(8, 2)

    def forward(self, x):
        lin1_out = self.linear1(x)
        sigmoid_out1 = self.sigmoid1(lin1_out)
        sigmoid_out2 = self.sigmoid2(self.linear2(sigmoid_out1))
        return self.linear3(sigmoid_out2)

# We now define a helper method to read the CSV and generate a TensorDataset object corresponding to the test set of the Titianic dataset. For more details on the pre-processing, refer to the Titanic_Basic_Interpret tutorial.

# In[4]:


# Read dataset from csv file.
def load_dataset():
    titanic_data = pd.read_csv(dataset_path)
    titanic_data = pd.concat([titanic_data,
                              pd.get_dummies(titanic_data['sex']),
                              pd.get_dummies(titanic_data['embarked'],prefix="embark"),
                              pd.get_dummies(titanic_data['pclass'],prefix="class")], axis=1)
    titanic_data["age"] = titanic_data["age"].fillna(titanic_data["age"].mean())
    titanic_data["fare"] = titanic_data["fare"].fillna(titanic_data["fare"].mean())
    titanic_data = titanic_data.drop(['name','ticket','cabin','boat','body','home.dest','sex','embarked','pclass'], axis=1)
    # Set random seed for reproducibility.
    np.random.seed(131254)

    # Convert features and labels to numpy arrays.
    labels = titanic_data["survived"].to_numpy()
    titanic_data = titanic_data.drop(['survived'], axis=1)
    feature_names = list(titanic_data.columns)
    data = titanic_data.to_numpy()

    # Separate training and test sets using 
    train_indices = np.random.choice(len(labels), int(0.7*len(labels)), replace=False)
    test_indices = list(set(range(len(labels))) - set(train_indices))

    test_features = data[test_indices]
    test_features_tensor = torch.from_numpy(test_features).type(torch.FloatTensor)
    dataset = TensorDataset(test_features_tensor)
    return dataset

# In the following cell, we set parameters USE_CUDA and WORLD_SIZE. WORLD_SIZE corresponds to the number of processes initialized. USE_CUDA should be set to true if GPUs are available and there must be at least WORLD_SIZE GPUs available.

# In[5]:


USE_CUDA = True
WORLD_SIZE = 4

# We now define the function that runs on each process, which takes the rank (identifier for current process) and size (total number of processes). The model and appropriate part of the dataset are loaded, and attributions are computed for this part of the dataset. The attributions are then averaged across processes. Note that DistributedSampler repeats examples to ensure that each partition has the same number of examples.
# 
# Note that this method loads a pretrained Titanic model, which can be downloaded from here: https://github.com/pytorch/captum/blob/master/tutorials/models/titanic_model.pt . Alternatively, the model can be trained from scratch from the Titanic_Basic_Interpret tutorial.

# In[6]:


def run(rank, size):
    # Load Dataset
    dataset = load_dataset()
    
    # Create TitanicSimpleNNModel and load saved weights.
    net = TitanicSimpleNNModel()
    net.load_state_dict(torch.load('models/titanic_model.pt'))
    
    # Create sampler which divides dataset among processes.
    sampler = DistributedSampler(dataset,num_replicas=size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=50, sampler=sampler)
    
    # If USE_CUDA, move model to CUDA device with id rank.
    if USE_CUDA:
        net = net.cuda(rank)
        
    # Initialize IG object
    ig = IntegratedGradients(net)
    
    # Compute total attribution
    total_attr = 0
    for batch in loader:
        inp = batch[0]
        if USE_CUDA:
            inp = inp.cuda(rank)
        attr = ig.attribute(inp, target=1)
        total_attr += attr.sum(dim=0)
        
    # Divide by number of examples in partition
    total_attr /= len(sampler)
    
    # Sum average attributions from each process on rank 0.
    torch.distributed.reduce(total_attr, dst=0)
    if rank == 0:
        # Average across processes, since each partition has same number of examples.
        total_attr = total_attr / size
        print("Average Attributions:", total_attr)

# This function performs required setup and cleanup steps on each process and executes the chosen function (run).

# In[7]:


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()

# We are now ready to run the initialize and run the processes. The average attributions over the dataset are printed by the rank 0 process upon completion.

# In[8]:


size = WORLD_SIZE
processes = []

for rank in range(size):
    p = Process(target=init_process, args=(rank, size, run))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

