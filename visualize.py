import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches

def img_predict(model,grid_size = 64,epoch=2000,number_of_traj=9):
    xlist = np.linspace(-0.5, 0.5, grid_size)
    ylist = np.linspace(-0.5, 0.5, grid_size)
    X, Y = np.meshgrid(xlist, ylist)
    Z_rgb = list() #y.reshape(64,64)
    for index in range(number_of_traj):
        with torch.no_grad():
            samples = torch.from_numpy(np.vstack([Y.reshape(-1),X.reshape(-1)])).float().permute(1,0).cuda()            
            samples.requires_grad=False

            subject_idx = torch.Tensor([index]).squeeze().long().cuda()[None,...]
            embedding = model.get_embedding(subject_idx)    
            out = model.inference(samples,embedding,epoch=epoch)['rgb'].squeeze().detach().cpu().numpy() + 0.5
            out[out>1]=1
            out[out<0]=0
            Z_rgb.append(out.reshape(grid_size,grid_size,3))          
            
    fig,ax=plt.subplots(3,3,figsize=(16,16))
    for index in range(9):
        ax[index//3,index%3].imshow(Z_rgb[index])
    plt.show()
def img_predict_inference(model,grid_size = 64,epoch=2000,number_of_traj=9):
    xlist = np.linspace(-0.5, 0.5, grid_size)
    ylist = np.linspace(-0.5, 0.5, grid_size)
    X, Y = np.meshgrid(xlist, ylist)
    Z_rgb = list() #y.reshape(64,64)
    for index in range(number_of_traj):
        with torch.no_grad():
            samples = torch.from_numpy(np.vstack([Y.reshape(-1),X.reshape(-1)])).float().permute(1,0).cuda()            
            samples.requires_grad=False

            subject_idx = torch.Tensor([index]).squeeze().long().cuda()[None,...]
            embedding = model.get_embedding(subject_idx)    
            out = model.inference(samples,embedding,epoch=epoch)['rgb'].squeeze().detach().cpu().numpy() + 0.5
            out[out>1]=1
            out[out<0]=0
            Z_rgb.append(out.reshape(grid_size,grid_size,3))          
            
    fig,ax=plt.subplots(3,3,figsize=(16,16))
    for index in range(9):
        ax[index//3,index%3].imshow(Z_rgb[index])
    plt.show()    
def img_predict_interp(model,grid_size = 64,epoch=2000,index_1=0,index_2=8):
    xlist = np.linspace(-0.5, 0.5, grid_size)
    ylist = np.linspace(-0.5, 0.5, grid_size)
    X, Y = np.meshgrid(xlist, ylist)
    Z_rgb = list() #y.reshape(64,64)
    for index in range(5):
        with torch.no_grad():
            samples = torch.from_numpy(np.vstack([Y.reshape(-1),X.reshape(-1)])).float().permute(1,0).cuda()            
            samples.requires_grad=False

            subject_idx = torch.Tensor([index_1]).squeeze().long().cuda()[None,...]
            embedding1 = model.get_embedding(subject_idx)    
            subject_idx = torch.Tensor([index_2]).squeeze().long().cuda()[None,...]
            embedding2 = model.get_embedding(subject_idx)    
            embedding = embedding1*(1-index/4.0)+embedding2*(index/4.0)
            out = model.inference(samples,embedding,epoch=epoch)['rgb'].squeeze().detach().cpu().numpy()+0.5
            out[out>1]=1
            out[out<0]=0
            Z_rgb.append(out.reshape(grid_size,grid_size,3))         
            
    fig,ax=plt.subplots(1,5,figsize=(30,6))
    for index in range(5):
        ax[index].imshow(Z_rgb[index])
        ax[index].get_xaxis().set_visible(False)
        ax[index].get_yaxis().set_visible(False)        
    plt.show()    
def img_predict_interp_plot(model,grid_size = 64,epoch=2000,index1=0,index2=2):
    xlist = np.linspace(-0.5, 0.5, 1280)
    ylist = np.linspace(-0.5, 0.5, 720)
    X, Y = np.meshgrid(xlist, ylist)
    Z_rgb = list() #y.reshape(64,64)
    for index in range(5):
        with torch.no_grad():
            samples = torch.from_numpy(np.vstack([Y.reshape(-1),X.reshape(-1)])).float().permute(1,0).cuda()            
            samples.requires_grad=False

            subject_idx = torch.Tensor([index1]).squeeze().long().cuda()[None,...]
            embedding1 = model.get_embedding(subject_idx)    
            subject_idx = torch.Tensor([index2]).squeeze().long().cuda()[None,...]
            embedding2 = model.get_embedding(subject_idx)    
            embedding = embedding1*(index/4.0)+embedding2*(1-index/4.0)
            out = model.inference(samples,embedding,epoch=epoch)['rgb'].squeeze().detach().cpu().numpy()+0.5
            out[out>1]=1
            out[out<0]=0
            Z_rgb.append(out.reshape(720,1280,3))         
            
    fig,ax=plt.subplots(1,5,figsize=(20,5))
    for index in range(5):
        ax[index].imshow(Z_rgb[index])
        ax[index].get_xaxis().set_visible(False)
        ax[index].get_yaxis().set_visible(False)        
    plt.show()    
def contour_plot(model,dataset, epoch=2000): # Real_world_exp_1    
    xlist = np.linspace(-0.1, 0.4, 32)
    ylist = np.linspace(-0.5, 0.35, 32)
    zlist = np.linspace(0, 0.4, 32)
    tlist = np.linspace(-0.5, 0.5, 32)    
    X,T, Y, Z = np.meshgrid(xlist,tlist,ylist,zlist)
    Z_mp = list()
    for index in range(3):
        with torch.no_grad():
            subject_idx = torch.Tensor([index]).squeeze().long().cuda()[None,...]
            embedding = model.get_embedding(subject_idx)    
            samples_mp = torch.from_numpy(np.vstack([T.reshape(-1),X.reshape(-1),Y.reshape(-1),
                                                     Z.reshape(-1)])).float().permute(1,0).cuda()
            samples_mp.requires_grad=False
            out = model.inference_mp(samples_mp,embedding,epoch=epoch)  
            z_mp = out['mp'].squeeze().detach().cpu().numpy() 
            z_mp[z_mp>1]=1
            z_mp[z_mp<-1]=-1
            z_mp = z_mp.reshape(32,32,32,32)
            Z_mp.append(z_mp)
    xlist = np.linspace(-0.1, 0.4, 32)
    ylist = np.linspace(-0.5, 0.35, 32)
    fig,ax=plt.subplots(1,3,figsize=(20,6))
    Y, X= np.meshgrid(ylist, xlist)
    for index in range(3):
        cp = ax[index].contourf(Y, X, Z_mp[index].min(axis=(0,3)))
        ax[index].plot(dataset.demo_x[index,:100,1],dataset.demo_x[index,:100,0],color='red')
        ax[index].plot(dataset.demo_x[index,100:,1],dataset.demo_x[index,100:,0],color='red')
        ax[index].set_xlabel(str(index))
        fig.colorbar(cp,ax=ax[index]) # Add a colorbar to a plot
    plt.show() 
def contour_plot_interp(model,epoch=2000,index1=0,index2=2):    
    xlist = np.linspace(-0.3, 0.55, 32)
    ylist = np.linspace(-0.5, 0.35, 32)
    zlist = np.linspace(0, 0.4, 32)
    tlist = np.linspace(-0.5, 0.5, 32)    
    X,T,   Y, Z = np.meshgrid(xlist,tlist,ylist,zlist)
    Z_mp = list() 
    for index in range(5):
        with torch.no_grad():
            subject_idx = torch.Tensor([index1]).squeeze().long().cuda()[None,...]
            embedding1 = model.get_embedding(subject_idx)    
            subject_idx = torch.Tensor([index2]).squeeze().long().cuda()[None,...]
            embedding2 = model.get_embedding(subject_idx)    
            embedding = embedding1*(1-index/4.0)+embedding2*(index/4.0) 
            samples_mp = torch.from_numpy(np.vstack([T.reshape(-1),X.reshape(-1),Y.reshape(-1),
                                                     Z.reshape(-1)])).float().permute(1,0).cuda()
            samples_mp.requires_grad=False
            out = model.inference_mp(samples_mp,embedding,epoch=epoch)  
            z_mp = out['mp'].squeeze().detach().cpu().numpy() 
            z_mp[z_mp>1]=1
            z_mp[z_mp<-1]=-1
            z_mp = z_mp.reshape(32,32,32,32)
            Z_mp.append(z_mp)
    xlist = np.linspace(-0.3, 0.55, 32)
    ylist = np.linspace(-0.5, 0.35, 32)
    fig,ax=plt.subplots(1,5,figsize=(30,6))
    Y, X= np.meshgrid(ylist, xlist)
    for index in range(5):
        cp = ax[index].contourf(Y, X, Z_mp[index].min(axis=(0,3)))
        rect = patches.Rectangle((-0.20, 0.09-index*0.0225), 0.20, 0.06+index*0.045,
                                 linewidth=1, edgecolor='black', facecolor='gray')
        ax[index].add_patch(rect)
        ax[index].set_xlabel(str(index))
        fig.colorbar(cp,ax=ax[index]) # Add a colorbar to a plot
    plt.show()     