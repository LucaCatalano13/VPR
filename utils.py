import faiss
import logging
import random
import numpy as np
from typing import Tuple
import torch
from torch.utils.data import Dataset, Sampler
from pytorch_metric_learning import losses
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import math

import visualizations


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class CosPlace(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gem = GeM()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.gem(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        # Mixer Inner Structure: Norm , Linear , ReLu, Linear
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )
        # Initialization of each Linear layer with normal distributed weights N(mean = 0 , std = 0.02) and bias = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Forward uses a skip connection and the Mixer layer defined above
        return x + self.mix(x)

class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=512,
                 in_h=7,
                 in_w=7,
                 out_channels=512,
                 mix_depth=4,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h  # height of input feature maps
        self.in_w = in_w  # width of input feature maps
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio  # ratio of the mid projection layer in the mixer block

        hw = in_h * in_w
        # Build the L MixerLayer in a cascade architecture
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        # x is [batch_ize, 512, 7, 7] and flattened to [batch_size, 512, 49] from now on we refer as h*w = 49 dimension as "row"
        x = x.flatten(2)
        # mix layer preserves dimension, so it is still [batch_size, 512 , 49]
        x = self.mix(x)
        # Change the order of last two dimension of x from [batch_size, 512 , 49] to [batch_size, 49, 512]
        x = x.permute(0, 2, 1)
        # Reduce dimensionality of channels via Linear Layer from [batch_size , 49, 512] to [batch_size, 49, out_channels]
        x = self.channel_proj(x)
        # Come back to original order of dimension [batch_size, out_channels, 49]
        x = x.permute(0, 2, 1)
        # Reduce dimensionality of h*w called "row" via Linear Layer from [batch_size, out_channels, 49] to [batch_size, out_channels, out_rows]
        x = self.row_proj(x)
        # Produces an output of shape [batch_size, out_channels*out_channels]
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x

class ProxyHead(nn.Module):
    def __init__(self, in_dim, out_dim = 512):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, p=2)
        return x

class ProxyBank:
    """This class stores the places' proxies together with their identifier
       and performs exhaustive search on the index to retrieve the mini-batch sampling pool."""

    def __init__(self, k = 512):
        self.__bank = {}
        self.k = k
        self.dim = 512
        self.n_samples = self.dim
        self.index = faiss.index_factory(self.dim, 'IDMap,Flat')

    def update_bank(self, proxies, labels):
        #riempo la banca
        for d, l in zip(proxies, labels):
            # From Tensor to int to be usable in dictionary as key
            label = int(l)
            # Create or Update the content of the bank dictionary
            if label not in self.__bank.keys():
                self.__bank[label] = ProxyBank.Proxy( tensor = d , n = 1 )
            else:
                self.__bank[label] = ProxyBank.Proxy( tensor = d , n = 1 ) + self.__bank[label]

    def update_index():
        for label, proxy_acc in self.__bank.items:
            # Use get_avg() to go from accumulator to average and compute the global proxy for each place
            self.index.add_with_ids( proxy_acc.get_avg().reshape(1,-1).detach().cpu() , label )
 
    def batch_sampling(self , batch_dim):
        batches = []
        #TODO: check if bank is updated 
        # While places are enough to generate the KNN
        while len(self.__bank) >= batch_dim:
            # Extract from bank a random label-proxy related to a place
            rand_index = random.randint(0 , len(self.__bank) - 1)
            rand_bank_item = list(self.__bank.items())[rand_index]
            # Inside bank i have ProxyAccumulator --> get_avg gives me the Proxy
            starting_proxy = rand_bank_item[1].get_avg()
            # Compute the batch_size_Nearest_Neighbours with faiss_index w.r.t. the extracted proxy
            distances, batch_of_labels = self.__index.search(starting_proxy.reshape(1,-1).detach().cpu(), batch_dim)
            # Faiss return a row per query in a multidim np.array, extract the one row
            batch_of_labels = batch_of_labels.flatten()
            # Add the new generated batch the one alredy created. KNN contains the starting proxy itself. Labels is the new Batch
            batches.append(batch_of_labels)
            # Remove all the already picked places from the index and the bank (no buono)
            for key_to_del in batch_of_labels:
                del self.__bank[key_to_del]
            self.__index.remove_ids(batch_of_labels)
        # Output the batches
        return batches 
    
    class Proxy:
        def __init__(self, tensor = None, n = 1, dim = 128):
            if tensor is None:
                self.__arch = torch.zeros(dim)
            else:
                self.__arch = tensor
            self.__n = n

        def get_avg(self):
            return self.__arch / self.__n

        def __add__(self, other):
            return ProxyBank.Proxy(tensor=self.__arch + other.__arch, n=self.__n + other.__n)

class ProxyBankBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, bank):
        # Epoch counter
        self.is_first_epoch = True
        # Save dataset
        self.dataset = dataset
        # Set dim of batch
        self.batch_size = batch_size
        # Compute the floor of the length of the iterable
        self.iterable_size = len(dataset) // batch_size
        # This is our ProxyBank, hopefully updated at the end of each epoch
        self.bank = bank
        self.batch_iterable = []
        
    # Return an iterable over a list of groups of indeces (list of batches)
    def __iter__(self): 
        # Epoch 0 case
        if self.is_first_epoch:
            # Change flag, first epoch is done
            self.is_first_epoch = False
            # Generate a random order of the indeces of the dataset, inside the parentesis there is the len of the dataset
            random_indeces_perm = torch.randperm(len(self.dataset))
            # Generate a fixed size partitioning of indeces
            batches =  torch.split(random_indeces_perm , self.batch_size)
            self.batch_iterable = iter(batches)
        # Epochs where Bank is informative, after epoch 0
        else:
            # Generate batches from ProxyBank
            batches = self.bank.batch_sampling( self.batch_size )
            self.batch_iterable = iter(batches)
        self.counter += 1
        return  self.batch_iterable
    
    # Return the length of the generated iterable, the one over the batches
    def __len__(self):
        return self.iterable_size

def compute_recalls(eval_ds: Dataset, queries_descriptors : np.ndarray, database_descriptors : np.ndarray,
                    output_folder : str = None, num_preds_to_save : int = 0,
                    save_only_wrong_preds : bool = True) -> Tuple[np.ndarray, str]:
    """Compute the recalls given the queries and database descriptors. The dataset is needed to know the ground truth
    positives for each query."""

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(queries_descriptors.shape[1])
    faiss_index.add(database_descriptors)
    del database_descriptors

    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])

    # Save visualizations of predictions
    if num_preds_to_save != 0:
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, output_folder, save_only_wrong_preds)
    
    return recalls, recalls_str
