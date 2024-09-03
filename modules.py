'''Define basic blocks
'''

import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F

'''Adapted from the SIREN repository https://github.com/vsitzmann/siren
'''

class BatchLinear(nn.Linear,MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.
    '''
    __doc__ = nn.Linear.__doc__ 
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.weight_normalization = False
    def init_weight_normalization(self):
        self.weight_normalization = True
        self.c = nn.Parameter(torch.max(torch.sum(torch.abs(self.weight), dim=1)))
        self.softplus= torch.nn.Softplus()
        self.register_parameter(name='c', param=self.c)
    def weight_normalization_(self,W, softplus_c):
        """
        Lipschitz weight normalization based on the L-infinity norm (see Eq.9 in [Liu et al 2022])
        There is no regularization on c(Adding the regularization to loss function didn't result in better performance), 
        so this shouldn't have any effect on the system, but somehow it does. 
        * It leads to more consistent training and smoother transition between generated images.
        * Performance in the two benchmarking test
        """
        absrowsum = torch.sum(torch.abs(W), axis=1)
        scale = torch.minimum(torch.ones_like(absrowsum), softplus_c/absrowsum)
        return W * scale[:,None]
    def forward(self, input, params=None):
        
        
        if params is None:
            bias = self.bias
            weight = self.weight
            if self.weight_normalization:
                weight = self.weight_normalization_(weight,self.softplus(self.c))
            output = F.linear(input, weight, bias)
        else:

            bias = params.get('bias', None)
            weight = params['weight']
            weight = weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2)
            bias = bias.unsqueeze(-2)
            if self.weight_normalization:
                weight = self.weight_normalization_(weight,self.softplus(self.c))            
            output = input.matmul(weight)
            output += bias
            # Shape example = torch.Size([16, 2000, 33]) torch.Size([16, 33, 256]) torch.Size([16, 1, 256])
        
        return output
        
class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, weight_init=None, weight_normalization = True):
        super().__init__()

        self.first_layer_init = None

        nl = nn.ReLU(inplace=True)

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features=in_features, out_features=hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(in_features=hidden_features, out_features=hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(in_features=hidden_features, out_features=out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(in_features=hidden_features, out_features=out_features), nl
            ))

        self.net = MetaSequential(*self.net)

        self.net.apply(init_weights_normal)
        
        batchlinear_layers = [module for module in self.net.modules() if isinstance(module, BatchLinear)]
        for layer in batchlinear_layers:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=2 / layer.weight.shape[1])
            if weight_normalization:
                layer.init_weight_normalization()


    def forward(self, coords, params=None, **kwargs):
        if params is not None:
            params = self.get_subdict(params, 'net')

        output = self.net(coords, params = params)
        return output

""" Following two functions are adapted from BARF.
"""

def positional_encoding(input,L): # [B,...,N]
    shape = input.shape
    freq = 2**torch.arange(L,dtype=torch.float32).cuda()*np.pi # [L]
    spectrum = input[...,None]*freq # [B,...,N,L]
    sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
    input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
    input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
    return input_enc
def positional_encoding_masked(input,L,epoch): # [B,...,N]
    input_enc = positional_encoding(input,L) # [B,...,2NL]
    # coarse-to-fine: smoothly mask positional encoding for BARF
    # set weights for different frequency bands
    start,end = [0.0,0.5]
    alpha = (epoch/1000.0-start)/(end-start)*L
    k = torch.arange(L,dtype=torch.float32).cuda()
    weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
    # apply weights
    shape = input_enc.shape
    input_enc = (input_enc.view(-1,L)*weight).view(*shape)
    return input_enc

class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, in_features=2,
                 hidden_features=256, num_hidden_layers=3,L = 5, pos_encoding=False, weight_normalization = True, **kwargs):
        super().__init__()
        self.L = L
        self.pos_encoding=pos_encoding
        if pos_encoding:
            self.net = FCBlock(in_features=in_features + in_features*2*L, out_features=out_features, 
                               num_hidden_layers=num_hidden_layers, hidden_features=hidden_features,
                               outermost_linear=True, weight_normalization = weight_normalization)
        else:
            self.net = FCBlock(in_features=in_features, out_features=out_features, 
                               num_hidden_layers=num_hidden_layers,hidden_features=hidden_features, 
                               outermost_linear=True, weight_normalization = weight_normalization)
        print(self)

    def forward(self, model_input, epoch, params=None,requires_grad=True):

        # Enables us to compute gradients w.r.t. coordinates
        if requires_grad:
            coords_org = model_input['coords'].requires_grad_(True)
        else:
            coords_org = model_input['coords']
        if self.pos_encoding:
            coords = torch.cat([coords_org, positional_encoding_masked(coords_org,self.L,epoch)],dim=-1)
        else:
            coords = coords_org

        # various input processing methods for different applications
        output = self.net(coords, self.get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}
    
class SingleBVPNetWithEmbedding(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, in_features=2,
                 hidden_features=256, num_hidden_layers=3, L = 5, pos_encoding=False, **kwargs):
        super().__init__()
        self.L = L
        self.pos_encoding=pos_encoding
        if pos_encoding:
            self.net = FCBlock(in_features=in_features + in_features*2*L+hidden_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                               hidden_features=hidden_features, outermost_linear=True)
        else:
            self.net = FCBlock(in_features=in_features+hidden_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                               hidden_features=hidden_features, outermost_linear=True)
        print(self)

    def forward(self, model_input, epoch, params=None):

        # Enables us to compute gradients w.r.t. coordinates
        embed = model_input['z']
        coords_org = model_input['coords'].requires_grad_(True)
        if self.pos_encoding:
            coords = torch.cat([coords_org, positional_encoding_masked(coords_org,self.L,epoch)],dim=-1)
        else:
            coords = coords_org
        net_input = torch.cat([coords,
                               embed[:,None,:].repeat(1,coords.shape[1],1)
                               ],dim=-1)
        # various input processing methods for different applications
        output = self.net(net_input)
        return {'model_in': coords_org, 'model_out': output}
    
def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')