import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Dict

class ANN(nn.Module):
    """
    Class for creating simple fully-connected networks
    
    TO-DO:
        -Implement unittests for the individual functions
    """
    def __init__(self, num_inputs: int, num_outputs: int, hidden_layer_dims: list[int], activation_fun: str="ReLU", activation_function_kw: Dict={}, dropout: float=0., use_batch_norm: bool=False, weight_init: str="kaiming_uniform", weight_init_kw: Dict={"a": np.sqrt(5)}, dtype=torch.float32, seed: int=42) -> None:
        super(ANN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layer_dims = hidden_layer_dims

        self.weight_init = weight_init
        self.weight_init_kw = weight_init_kw
        
        self.activation_fun = activation_fun
        self.activation_fun_kw = activation_function_kw

        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.seed = seed

        torch.manual_seed(seed)
        self._define_layers()

    
    def forward(self, x): 
        # Hidden Layers (FC->Activation->Dropout->BatchNorm)
        for k in range(len(self.hidden_layer_dims)):
            x = getattr(self, "fc_" + str(k+1))(x)
            x = getattr(self, "activation_" + str(k+1))(x)
            x = getattr(self, "dropout_" + str(k+1))(x)
            if self.use_batch_norm:
                x = getattr(self, "bn_" + str(k+1))(x)
        
        # Output Layer
        x = self.out(x)

        return x


    def _define_layers(self) -> None:
        # Hidden Layers (FC->Activation->Dropout->BatchNorm)
        prev = self.num_inputs
        for k, current in enumerate(self.hidden_layer_dims):
            setattr(self, "fc_" + str(k+1), nn.Linear(prev, current, device=self.device, dtype=self.dtype))
            self._init_weights(getattr(self, "fc_" + str(k+1)))
            
            setattr(self, "activation_" + str(k+1), eval("nn." + self.activation_fun + "(**self.activation_fun_kw)"))
            
            setattr(self, "dropout_" + str(k+1), nn.Dropout(self.dropout))
            
            if self.use_batch_norm:
                setattr(self, "bn_" + str(k+1), nn.BatchNorm1d(current, device=self.device, dtype=self.dtype))
            
            prev = current

        # Output Layer
        self.out = nn.Linear(current, self.num_outputs, device=self.device, dtype=self.dtype)
    
    def _init_weights(self, layer) -> None:
        eval("nn.init." + self.weight_init + "_(layer.weight, **self.weight_init_kw)")

if __name__ == "__main__":
    # Example
    torch.manual_seed(42)

    batch = 4
    num_inputs = 7
    num_outputs = 2
    hidden_layer_dims = [4, 4]

    dropout = 0.1
    use_batch_norm = True
    weight_init = "xavier_normal"
    weight_init_kw = {"gain": 1.0}

    net = ANN(num_inputs, num_outputs, hidden_layer_dims, dropout=dropout, use_batch_norm=use_batch_norm, weight_init=weight_init, weight_init_kw=weight_init_kw)
    x = torch.randn((batch, num_inputs), device="cpu", dtype=torch.float32) # batch x dim
    y = net(x)
    assert y.size(0) == batch and y.size(1) == num_outputs
    print("Success!")

