import torch
from torch import nn
from modules import *
from meta_modules import HyperNetwork
from losses import task_loss, task_loss_with_deform, task_loss_rgb,task_loss_with_deform2

## Task 1 - Task 2
class NFSMP_Image(nn.Module):
    def __init__(self, traj_count, latent_dim=128, 
                 hyper_hidden_layers=2, hyper_hidden_features=64,
                 hidden_layers = 3, hidden_num=256,
                 mp_in_size=1, mp_out_size=6, 
                 sp_in_size=2, sp_out_size=3, 
                 L=8, L2=2):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(traj_count, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0)
        self.mp_out_size =mp_out_size
        
        self.mp_net = SingleBVPNet(in_features=mp_in_size,
                                   hidden_features=hidden_num, num_hidden_layers=hidden_layers,
                                   out_features=mp_out_size,L =L,pos_encoding=True)
        self.rgb_net = SingleBVPNet(in_features=sp_in_size,
                                   hidden_features=hidden_num, num_hidden_layers=hidden_layers,
                                    out_features=sp_out_size,L =L,pos_encoding=True) 
        self.deform_net = SingleBVPNet(in_features=sp_in_size,
                                   hidden_features=hidden_num, num_hidden_layers=hidden_layers,
                                    out_features=sp_in_size,L =L2,pos_encoding=True)         
        self.deform_epoch_multiplier = L/L2
        # Hyper-Net
        self.hyper_net_mp = HyperNetwork(hyper_in_features=self.latent_dim,
                                         hyper_hidden_layers=hyper_hidden_layers,
                                         hyper_hidden_features=hyper_hidden_features,
                                         hypo_module=self.mp_net)           
        self.hyper_net_deform= HyperNetwork(hyper_in_features=self.latent_dim,
                                         hyper_hidden_layers=hyper_hidden_layers, 
                                         hyper_hidden_features=hyper_hidden_features,
                                         hypo_module=self.deform_net)      
        last_layer = [layer for layer in self.deform_net.modules() if isinstance(layer, BatchLinear)][-1]
        torch.nn.init.zeros_(last_layer.weight)
        torch.nn.init.zeros_(last_layer.bias)   
    def get_embedding(self,instance_idx):
        embedding = self.latent_codes(instance_idx)
        return embedding        
    def latent_reset(self):
        nn.init.normal_(self.latent_codes.weight, mean=0, std=1e-6)
    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params_mp = self.hyper_net_mp(embedding)
        hypo_params_deform = self.hyper_net_deform(embedding)
        return hypo_params_mp, hypo_params_deform, embedding
    
    def inference(self,coords_rgb,embedding,epoch = 1000):

        with torch.no_grad():
            model_out = dict()
            
            hypo_params_deform = self.hyper_net_deform(embedding)
            model_in = {'coords': coords_rgb}
            deform =self.deform_net(model_in,self.deform_epoch_multiplier*epoch, params=hypo_params_deform)['model_out']
            
            model_in = {'coords': coords_rgb+deform}
            model_out['rgb'] =self.rgb_net(model_in,epoch)['model_out']            
            return model_out
        
    def inference_mp(self,coords_mp,embedding,epoch = 1000):

        with torch.no_grad():
            model_out = dict()
            hypo_params_mp = self.hyper_net_mp(embedding)
            model_in = {'coords': coords_mp}
            model_out['mp'] =self.mp_net(model_in,epoch, params=hypo_params_mp)['model_out']
            return model_out        

    def forward(self, model_input,gt,epoch):
        coords  = model_input['coords'] 
        t = model_input['t']
        model_in1 = {'coords': t}
        model_in2 = {'coords': coords}
        hypo_params_mp, hypo_params_deform, embedding= self.get_hypo_net_weights(model_input)            
        mp = self.mp_net(model_in1,epoch, params=hypo_params_mp)['model_out']      
        deform =self.deform_net(model_in2,self.deform_epoch_multiplier*epoch, params=hypo_params_deform)['model_out']
        model_in3 = {'coords': coords+deform}
        rgb = self.rgb_net(model_in3,epoch)['model_out']     
        
        model_out = {'t': t,
                     'coords':coords,
                     'mp':mp,
                     'rgb':rgb,
                     'deform':deform,                     
                     'latent_vec':embedding,}

        losses = task_loss_with_deform(model_out, gt)
        return losses 
    
## Task 1 - Task 2 - Inference

class NFSMP_Image_Inference(nn.Module):
    def __init__(self,traj_count,model, **kwargs):
        super().__init__()
        self.ref_latent_codes = model.latent_codes.weight.detach()
        self.latent_codes = nn.Embedding(traj_count, self.ref_latent_codes.shape[0])
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0)
        self.mp_net = model.mp_net
        self.rgb_net = model.rgb_net
        self.deform_net = model.deform_net         
        self.deform_epoch_multiplier = model.deform_epoch_multiplier
        self.traj_count=traj_count
        # Hyper-Net
        self.hyper_net_mp = model.hyper_net_mp
        self.hyper_net_deform= model.hyper_net_deform
    
    def get_embedding(self,instance_idx):
        embedding = torch.softmax(self.latent_codes(instance_idx),dim=-1)@self.ref_latent_codes
        return embedding
        
    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.get_embedding(instance_idx)#self.latent_codes(instance_idx)
        hypo_params_mp = self.hyper_net_mp(embedding)
        hypo_params_deform = self.hyper_net_deform(embedding)
        return hypo_params_mp, hypo_params_deform, embedding
    
    def inference(self,coords_rgb,embedding,epoch = 1000):

        with torch.no_grad():
            model_out = dict()
            
            hypo_params_deform = self.hyper_net_deform(embedding)
            model_in = {'coords': coords_rgb}
            deform =self.deform_net(model_in,epoch*self.deform_epoch_multiplier, params=hypo_params_deform)['model_out']
            
            model_in = {'coords': coords_rgb+deform}
            model_out['rgb'] =self.rgb_net(model_in,epoch)['model_out']            
            return model_out
    def inference_mp(self,coords_mp,embedding,epoch = 1000):

        with torch.no_grad():
            model_out = dict()
            hypo_params_mp = self.hyper_net_mp(embedding)
            model_in = {'coords': coords_mp}
            model_out['mp'] =self.mp_net(model_in,epoch, params=hypo_params_mp)['model_out']
            return model_out        
        
    def forward(self, model_input,gt,epoch,**kwargs):
        coords  = model_input['coords'] 
        t = model_input['t']
        model_in1 = {'coords': t}
        model_in2 = {'coords': coords}
        hypo_params_mp, hypo_params_deform, embedding= self.get_hypo_net_weights(model_input)            
        mp = self.mp_net(model_in1,epoch, params=hypo_params_mp)['model_out']      
        deform =self.deform_net(model_in2,epoch*self.deform_epoch_multiplier, params=hypo_params_deform)['model_out']
        model_in3 = {'coords': coords+deform}
        rgb = self.rgb_net(model_in3,epoch)['model_out']     
        model_out = {'t': t,
                     'coords':coords,
                     'mp':mp,
                     'rgb':rgb,
                     'latent_vec':embedding,}
        losses = task_loss_rgb(model_out, gt)
        return losses

    
class NFSMP_Implicit(nn.Module):
    def __init__(self, traj_count, latent_dim=128, 
                 hyper_hidden_layers=2, hyper_hidden_features=64, 
                 hidden_layers = 3, hidden_num=256,
                 mp_in_size=4, mp_out_size=1, 
                 sp_in_size=2, sp_out_size=3, 
                 L=8, L2=2):
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(traj_count, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0)

        self.sp_out_size = sp_out_size
        self.sp_in_size  = sp_in_size
        
        self.mp_out_size = mp_out_size
        self.mp_in_size  = mp_in_size
        
        self.mp_net = SingleBVPNet(in_features=mp_in_size,hidden_features = hidden_num, num_hidden_layers=hidden_layers,
                                   out_features=mp_out_size,L =L,pos_encoding=True)
        self.sp_net = SingleBVPNet(in_features=sp_in_size,hidden_features = hidden_num, num_hidden_layers=hidden_layers,
                                    out_features=sp_out_size,L =L,pos_encoding=True) 
        self.deform_net_sp = SingleBVPNet(in_features=sp_in_size,hidden_features = hidden_num, num_hidden_layers=hidden_layers,
                                    out_features=sp_in_size,L =L2,pos_encoding=True)     
        self.deform_net_mp = SingleBVPNet(in_features=mp_in_size,hidden_features = hidden_num, num_hidden_layers=hidden_layers,
                                    out_features=mp_in_size,L =L2,pos_encoding=True)    
        
        self.deform_epoch_multiplier = 1.0*L/L2
        
        # Hyper-Net
        self.hyper_net_mp_deform = HyperNetwork(hyper_in_features=self.latent_dim,
                                         hyper_hidden_layers=hyper_hidden_layers,
                                         hyper_hidden_features=hyper_hidden_features,
                                         hypo_module=self.deform_net_mp)           
        self.hyper_net_sp_deform= HyperNetwork(hyper_in_features=self.latent_dim,
                                         hyper_hidden_layers=hyper_hidden_layers, 
                                         hyper_hidden_features=hyper_hidden_features,
                                         hypo_module=self.deform_net_sp)    
        
        last_layer = [layer for layer in self.deform_net_mp.modules() if isinstance(layer, BatchLinear)][-1]
        torch.nn.init.zeros_(last_layer.weight)
        torch.nn.init.zeros_(last_layer.bias)    
        
        last_layer = [layer for layer in self.deform_net_sp.modules() if isinstance(layer, BatchLinear)][-1]
        torch.nn.init.zeros_(last_layer.weight)
        torch.nn.init.zeros_(last_layer.bias)  
        
    def latent_reset(self):
        nn.init.normal_(self.latent_codes.weight, mean=0, std=1e-6)
    def get_embedding(self,instance_idx):
        embedding = self.latent_codes(instance_idx)
        return embedding
    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params_mp = self.hyper_net_mp_deform(embedding)
        hypo_params_sp = self.hyper_net_sp_deform(embedding)
        return hypo_params_mp, hypo_params_sp, embedding
    
    def inference(self, coords_sp, embedding, epoch = 1000):

        with torch.no_grad():
            model_out = dict()
            
            hypo_params_sp = self.hyper_net_sp_deform(embedding)
            model_in = {'coords': coords_sp}
            deform_sp =self.deform_net_sp(model_in,self.deform_epoch_multiplier*epoch, 
                                          params=hypo_params_sp)['model_out']
            
            model_in = {'coords': coords_sp+deform_sp}
            model_out['rgb'] =self.sp_net(model_in,epoch)['model_out']            
            return model_out
        
    def inference_mp(self,coords_mp,embedding,epoch = 1000):

        with torch.no_grad():
            model_out = dict()
            hypo_params_mp = self.hyper_net_mp_deform(embedding)
            
            model_in = {'coords': coords_mp}
            deform_mp =self.deform_net_mp(model_in,self.deform_epoch_multiplier*epoch, params=hypo_params_mp)['model_out']  
            
            model_in = {'coords': coords_mp+deform_mp}
            model_out['mp'] =self.mp_net(model_in,epoch)['model_out']
            return model_out           

    def forward(self, model_input, gt, epoch):
        coords_sp  = model_input['coords'] 
        coords_mp = model_input['coords_mp']
        model_in_sp_deform = {'coords': coords_sp}
        model_in_mp_deform = {'coords': coords_mp}
        hypo_params_mp, hypo_params_sp, embedding= self.get_hypo_net_weights(model_input) 
        
        deform_mp =self.deform_net_mp(model_in_mp_deform,self.deform_epoch_multiplier*epoch, 
                                      params=hypo_params_mp)['model_out']
        deform_sp =self.deform_net_sp(model_in_sp_deform,self.deform_epoch_multiplier*epoch, 
                                      params=hypo_params_sp)['model_out']
        
        model_in_mp = {'coords': coords_mp+deform_mp}
        model_in_sp = {'coords': coords_sp+deform_sp}
        
        sp = self.sp_net(model_in_sp,epoch)['model_out']     
        mp  = self.mp_net(model_in_mp,epoch)['model_out']      

        
        model_out = {'coords_sp': coords_sp,
                     'coords_mp':coords_mp,
                     'mp': mp,
                     'rgb': sp,
                     'deform': deform_sp,                     
                     'deform_mp': deform_mp,                     
                     'latent_vec': embedding,}

        losses = task_loss_with_deform2(model_out, gt)
        return losses
    
    
class NFSMP_Implicit_Inference(nn.Module):
    def __init__(self,traj_count,model, **kwargs):
        super().__init__()
        self.ref_latent_codes = model.latent_codes.weight.detach()
        self.latent_codes = nn.Embedding(traj_count, self.ref_latent_codes.shape[0])
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0)
        self.mp_net = model.mp_net
        self.sp_net = model.sp_net
        self.deform_net_sp = model.deform_net_sp   
        self.deform_net_mp = model.deform_net_mp  
        
        self.deform_epoch_multiplier = model.deform_epoch_multiplier
        
        # Hyper-Net
        self.hyper_net_mp_deform = model.hyper_net_mp_deform
        self.hyper_net_sp_deform= model.hyper_net_sp_deform
        
    def get_embedding(self,instance_idx):
        embedding = torch.softmax(self.latent_codes(instance_idx),dim=-1)@self.ref_latent_codes
        return embedding
        
    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.get_embedding(instance_idx)#self.latent_codes(instance_idx)
        hypo_params_mp = self.hyper_net_mp_deform(embedding)
        hypo_params_sp = self.hyper_net_sp_deform(embedding)
        return hypo_params_mp, hypo_params_sp, embedding
    
    def inference(self, coords_sp, embedding, epoch = 1000):

        with torch.no_grad():
            model_out = dict()
            
            hypo_params_sp = self.hyper_net_sp_deform(embedding)
            model_in = {'coords': coords_sp}
            deform_sp =self.deform_net_sp(model_in,self.deform_epoch_multiplier*epoch, 
                                          params=hypo_params_sp)['model_out']
            
            model_in = {'coords': coords_sp+deform_sp}
            model_out['rgb'] =self.sp_net(model_in,epoch)['model_out']            
            return model_out
        
    def inference_mp(self,coords_mp,embedding,epoch = 1000):

        with torch.no_grad():
            model_out = dict()
            hypo_params_mp = self.hyper_net_mp_deform(embedding)
            
            model_in = {'coords': coords_mp}
            deform_mp =self.deform_net_mp(model_in,self.deform_epoch_multiplier*epoch, params=hypo_params_mp)['model_out']  
            
            model_in = {'coords': coords_mp+deform_mp}
            model_out['mp'] =self.mp_net(model_in,epoch)['model_out']
            return model_out           

    def forward(self, model_input, gt, epoch):
        coords_sp  = model_input['coords'] 
        coords_mp = model_input['coords_mp']
        model_in_sp_deform = {'coords': coords_sp}
        model_in_mp_deform = {'coords': coords_mp}
        hypo_params_mp, hypo_params_sp, embedding= self.get_hypo_net_weights(model_input) 
        
        deform_mp =self.deform_net_mp(model_in_mp_deform,self.deform_epoch_multiplier*epoch, 
                                      params=hypo_params_mp)['model_out']
        deform_sp =self.deform_net_sp(model_in_sp_deform,self.deform_epoch_multiplier*epoch, 
                                      params=hypo_params_sp)['model_out']
       
        model_in_mp = {'coords': coords_mp+deform_mp}
        model_in_sp = {'coords': coords_sp+deform_sp}
        
        sp = self.sp_net(model_in_sp,epoch)['model_out']     
        mp  = self.mp_net(model_in_mp,epoch)['model_out']      

        model_out = {'coords_sp': coords_sp,
                     'coords_mp':coords_mp,
                     'mp': mp,
                     'rgb': sp,
                     'deform': deform_sp,                     
                     'deform_mp': deform_mp,                     
                     'latent_vec': embedding,}

        losses = task_loss_rgb(model_out, gt)
        return losses
    
    
# BASELINE Ref: https://github.com/myunusseker/CNMP

class CNNEncoder(torch.nn.Module):
    def __init__(self, layer_info =[3,8,16,32,64,128]): # Given 128*128*3 image, this will give 1*1*256
        super(CNNEncoder, self).__init__()
        layers = []
        in_channel = layer_info[0]
        first = True
        for out_channel in layer_info[1:]:
            layers.append(torch.nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1,bias=False))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            in_channel = out_channel
        layers.append(torch.nn.Conv2d(in_channel, in_channel, kernel_size=4, stride=1,bias=False))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y
    
class CNN_CNMP(nn.Module):
    
    def __init__(self,dx=1,dy=6):
        super(CNN_CNMP, self).__init__()
        self.cnn = CNNEncoder()
        # Encoder takes observations which are (X,Y) tuples and produces latent representations for each of them
        self.encoder = nn.Sequential(
            nn.Linear(128+dx+dy,128),nn.ReLU(),
            nn.Linear(128,128),nn.ReLU(),
            nn.Linear(128,128)
        )
        
        #Decoder takes the (r_mean, target_t) tuple and produces mean and std values for each dimension of the output
        self.decoder = nn.Sequential(
            nn.Linear(128+dx,128),nn.ReLU(),
            nn.Linear(128,128),nn.ReLU(),
            nn.Linear(128,2*dy)
        )
        
    def forward(self,image,observations,target_t):
        image_features = torch.tile(self.cnn(image).reshape(1,1,-1),(1,observations.shape[1],1))
        x = torch.cat((image_features,observations),dim=-1)
        r = self.encoder(x) # Generating observations
        r_mean = torch.mean(r,dim=1,keepdim=True) # .mean(dim=1)
        r_mean = r_mean.repeat(1,target_t.shape[1],1) # Duplicating general representation for every target_t
        concat = torch.cat((r_mean,target_t),dim=-1) # Concatenating each target_t with general representation
        output = self.decoder(concat) # Producing mean and std values for each target_t
        return output

class CNMP(nn.Module):
    
    def __init__(self,dx=1,m=0,dy=6):
        super(CNMP, self).__init__()
        
        # Encoder takes observations which are (X,Y) tuples and produces latent representations for each of them
        self.encoder = nn.Sequential(
        nn.Linear(m+dx+dy,128),nn.ReLU(),
        nn.Linear(128,128),nn.ReLU(),
        nn.Linear(128,128)
        )
        
        #Decoder takes the (r_mean, target_t) tuple and produces mean and std values for each dimension of the output
        self.decoder = nn.Sequential(
        nn.Linear(128+dx,128),nn.ReLU(),
        nn.Linear(128,128),nn.ReLU(),
        nn.Linear(128,2*dy)
        )
        
    def forward(self,observations,target_t):
        r = self.encoder(observations) # Generating observations
        r_mean = torch.mean(r,dim=1,keepdim=True) # .mean(dim=1)
        r_mean = r_mean.repeat(1,target_t.shape[1],1) # Duplicating general representation for every target_t
        concat = torch.cat((r_mean,target_t),dim=-1) # Concatenating each target_t with general representation
        output = self.decoder(concat) # Producing mean and std values for each target_t
        return output