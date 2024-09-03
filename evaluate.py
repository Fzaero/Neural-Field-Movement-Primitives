import numpy as np
import torch
import time
from tqdm import tqdm

from datasets import Robot_Traj_Dataset_Experiment_Sim,Robot_Traj_Dataset_Experiment_Real_World_1
from torch.utils.data import DataLoader,Dataset

from networks import NFSMP_Image, NFSMP_Image_Inference, NFSMP_Implicit_Inference
from visualize import img_predict_inference

def traj_field(model,grid_size = 100,epoch=1000, number_of_traj = 9):
    tlist = np.linspace(-0.5, 0.5, grid_size)
    Z_mp = list() #y.reshape(64,64)
    for index in range(number_of_traj):
        with torch.no_grad():
            samples = torch.from_numpy(tlist.reshape(-1,1)).float().cuda()            
            samples.requires_grad=False

            subject_idx = torch.Tensor([index]).squeeze().long().cuda()[None,...]
            embedding = model.get_embedding(subject_idx)
            out = model.inference_mp(samples,embedding,epoch=epoch)['mp'].squeeze().detach().cpu().numpy()
            Z_mp.append(out.reshape(grid_size,6))
            
    return Z_mp 


def validate(model, experiment_name="ur10_peg_in_hole_3x3",num_of_traj=9,task_id=1):
    
    val_dataset = Robot_Traj_Dataset_Experiment_Sim(experiment_name,num_of_traj,task_id)
    
    val_dataloader = DataLoader(val_dataset, shuffle=False,batch_size=40, num_workers=0, drop_last = False)  
    val_model = NFSMP_Image_Inference(num_of_traj,model)
    val_model.to(device=torch.device('cuda:0'))
    optim = torch.optim.Adam([
                    {'params': val_model.latent_codes.parameters()},
                ],
        lr=0.1)
    total_steps=0
    epochs=200
    val_losses = []
    for epoch in range(epochs):
        val_model.train()
        for step, (model_input, gt) in enumerate(val_dataloader):

            start_time = time.time()
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}
            losses = val_model(model_input,gt,epoch*5)
            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                train_loss += single_loss            

            optim.zero_grad()
            train_loss.backward()
            optim.step()
            total_steps += 1
    
    tlist = np.linspace(-0.5, 0.5, 100)
    Z_mp = list() #y.reshape(64,64)
    for index in range(num_of_traj):
        with torch.no_grad():
            samples = torch.from_numpy(tlist.reshape(-1,1)).float().cuda()            
            samples.requires_grad=False

            subject_idx = torch.Tensor([index]).squeeze().long().cuda()[None,...]
            embedding = val_model.get_embedding(subject_idx)
            out = model.inference_mp(samples,embedding,epoch=1000)['mp'].squeeze().detach().cpu().numpy()
            Z_mp.append(out.reshape(100,6))
    errors = list()
    for i in range(num_of_traj):
        errors.append(np.linalg.norm(Z_mp[i]-val_dataset.demo_q[i],axis=-1).mean()/np.pi*180) 
    return np.mean(errors)

def validate_real_1(model,num_of_traj=18):
    val_dataset = Robot_Traj_Dataset_Experiment_Real_World_1(num_of_traj)
    
    val_dataloader = DataLoader(val_dataset, shuffle=False,batch_size=40, num_workers=0, drop_last = False)  
    val_model = NFSMP_Implicit_Inference(num_of_traj//2,model)
    val_model.to(device=torch.device('cuda:0'))
    optim = torch.optim.Adam([
                    {'params': val_model.latent_codes.parameters()},
                ],
        lr=0.1)
    total_steps=0
    epochs=200
    val_losses = []
    for epoch in range(epochs):
        val_model.train()
        for step, (model_input, gt) in enumerate(val_dataloader):

            start_time = time.time()
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}
            losses = val_model(model_input,gt,epoch*5)
            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                train_loss += single_loss            

            optim.zero_grad()
            train_loss.backward()
            optim.step()
            total_steps += 1
    samples_all = torch.from_numpy(val_dataset.xy).float().cuda()  
    df_all = val_dataset.df.reshape(num_of_traj//2,-1,1)
    Z_mp = list() #y.reshape(64,64)
    for index in range(num_of_traj//2):
        with torch.no_grad():
            samples = samples_all[index]         
            samples.requires_grad=False

            subject_idx = torch.Tensor([index]).squeeze().long().cuda()[None,...]
            embedding = val_model.get_embedding(subject_idx)
            out = model.inference_mp(samples,embedding,epoch=1000)['mp'].squeeze().detach().cpu().numpy()
            Z_mp.append(out.reshape(-1,1))
    errors = list()
    for i in range(num_of_traj//2):
        errors.append(np.linalg.norm(Z_mp[i]-df_all[i],axis=-1).mean()) 
    return np.mean(errors)
def test(test_model,test_dataloader, visualize = True):
    test_model.to(device=torch.device('cuda:0'))
    optim = torch.optim.Adam([
                    {'params': test_model.latent_codes.parameters()},
                ],
        lr=0.1)
    total_steps=0
    epochs=200
    with tqdm(total = len(test_dataloader) * epochs) as pbar:
        for epoch in range(epochs):
            if epoch%100==99 and visualize:
                img_predict_inference(test_model,epoch=epoch*5)
            test_model.train()
            for step, (model_input, gt) in enumerate(test_dataloader):

                start_time = time.time()
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
                losses = test_model(model_input,gt,epoch*5)
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    if epoch %100== 0 and step==0:
                        print(loss_name,single_loss)
                    train_loss += single_loss            

                optim.zero_grad()
                train_loss.backward()
                optim.step()
                pbar.update(1)
                pbar.set_postfix(loss=train_loss.item(), time=time.time() - start_time, epoch=epoch)
                total_steps += 1