import torch
import torch.distributions as D
import torch.nn.functional as F

l1_loss = torch.nn.L1Loss()

def task_loss(model_output, gt):
    embeddings = model_output['latent_vec']    
    embeddings_constraint = torch.mean(embeddings ** 2)
    
    return {'rgb' :((model_output['rgb']-gt['rgb'])**2).mean() * 3e4, 
            'mp' : ((model_output['mp']-gt['mp'])**2).mean() * 1e5, 
            'embeddings_constraint': embeddings_constraint.mean() * 1e6,
           }


def task_loss_with_deform(model_output, gt):
    embeddings = model_output['latent_vec']    
    embeddings_constraint = torch.mean(embeddings ** 2)
    deform = model_output['deform']
    return {'rgb' :((model_output['rgb']-gt['rgb'])**2).mean() * 3e4, 
            'mp' : ((model_output['mp']-gt['mp'])**2).mean() * 1e5, 
            'deform': deform.norm(dim=-1).mean()*1e-1,       
            'embeddings_constraint': embeddings_constraint.mean() * 1e6,
           }

def task_loss_with_deform2(model_output, gt):
    embeddings = model_output['latent_vec']    
    embeddings_constraint = torch.mean(embeddings ** 2)
    deform = model_output['deform']
    return {'rgb' :((model_output['rgb']-gt['rgb'])**2).mean() * 3e4, 
            'mp' : ((model_output['mp']-gt['mp'])**2).mean() * 1e5, 
            'deform': deform.norm(dim=-1).mean()*1e-4,       
            'embeddings_constraint': embeddings_constraint.mean() * 1e5,
           }

def task_loss_rgb(model_output, gt):    
    return {'rgb' :((model_output['rgb']-gt['rgb'])**2).mean() * 3e4}


def log_prob_loss(output, target):
    mean, sigma = output.chunk(2, dim = -1)
    sigma = F.softplus(sigma)
    dist = D.Independent(D.Normal(loc=mean, scale=sigma), 1)
    return -torch.mean(dist.log_prob(target))