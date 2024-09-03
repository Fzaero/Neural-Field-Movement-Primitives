import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join
import open3d as o3d
import trimesh
import cv2
import pickle
from scipy.spatial import distance

class Robot_Traj_Dataset_Experiment_Sim(Dataset):
    def __init__(self,task_path,traj_count, task_id = 1):
        noj =6
        self.traj_count=traj_count
        self.demo_t = np.zeros((traj_count,100,1))
        self.demo_q = np.zeros((traj_count,100,noj))
        self.demo_img = np.zeros((traj_count,256,256,3))
        self.demo_envs = list()
        for p in range(traj_count):
            with open("demo_data/"+task_path+"/"+str(p)+"/data.pickle", 'rb') as handle:
                data = pickle.load(handle)
            self.demo_envs.append(data['env_parameters'])
            self.demo_t[p,:,0] = np.array(data['t'])-0.5
            self.demo_q[p] = np.array(data['joint_traj'])#/np.pi/2
            # TODO: Change this, it doesn't look nice
            path_to_img = "demo_data/"+task_path+"/"+str(p)+"/img.png"
            if task_id==1:
                self.demo_img[p] = cv2.cvtColor(cv2.resize(cv2.imread(path_to_img,-1)[50:750,100:1200],(256,256)), cv2.COLOR_BGR2RGB)            
            elif task_id==2:
                self.demo_img[p] = cv2.cvtColor(cv2.resize(cv2.imread(path_to_img,-1)[0:750,100:1200],(256,256)), cv2.COLOR_BGR2RGB)
    def __len__(self):
        return self.traj_count  
    
    def __getitem__(self,index):
        random_points_on_traj = np.random.choice(np.arange(100),100)
        
        t_mp = torch.from_numpy(self.demo_t[index,random_points_on_traj]).float()
        q_mp = torch.from_numpy(self.demo_q[index,random_points_on_traj]).float()
        
        shape_points_grid_ind = np.random.randint(65536, size=8192)
        shape_points = np.stack([shape_points_grid_ind//256, shape_points_grid_ind%256]).T
        image_values = self.demo_img[index, shape_points_grid_ind//256, shape_points_grid_ind%256]/255.0-0.5
        
        x_rgb = torch.from_numpy(shape_points).float()/256.0-0.5
        y_rgb = torch.from_numpy(image_values).float()

        observations =  {'t': t_mp,
                         'coords': x_rgb,
                         'rgb': y_rgb,
                         'mp': q_mp,
                         'instance_idx':torch.Tensor([index]).squeeze().long()}
    
        ground_truth = {'rgb': y_rgb,
                        'mp':  q_mp,}
        
        return observations, ground_truth
class Robot_Traj_Dataset_Experiment_Real_World_1(Dataset):
    def __init__(self,traj_count):
        self.x_limits = np.array([
            [-0.3,-0.5,-0.1],
            [0.5,0.5,0.5]
        ])
        self.demo_t = np.zeros((traj_count//2,200,1))
        self.demo_x = np.zeros((traj_count//2,200,3))
        self.traj_count = traj_count
        self.demo_q = np.zeros((traj_count//2,200,6))
        self.demo_depths = np.zeros((traj_count//2,256,256,1))
        self.demo_img = np.zeros((traj_count//2,256,256,3))
        self.demo_envs = list()
        for p in range(traj_count):
            with open("demo_data/real_world_exp1_exp_ready/"+str(p)+"/data.pickle", 'rb') as handle:
                data = pickle.load(handle)
            self.demo_envs.append(data['env_parameters'])
            self.demo_t[p//2,(p%2)*100:(p%2+1)*100,0] = np.array(data['t'])
            self.demo_x[p//2,(p%2)*100:(p%2+1)*100] = np.array(data['pos_traj'])
            self.demo_q[p//2,(p%2)*100:(p%2+1)*100] = np.array(data['joint_traj'])
            img = cv2.resize(cv2.imread("demo_data/real_world_exp1_exp_ready/"+str(p)+"/img.png",-1)[100:,50:],(256,256))
            self.demo_img[p//2] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.demo_depths[p//2,:,:,0] = cv2.resize(cv2.imread("demo_data/real_world_exp1_exp_ready/"+str(p)+"/depth.png",-1)[100:,50:],(256,256))
        self.x = np.zeros((traj_count//2,4000,1))
        self.y = np.zeros((traj_count//2,4000,3))
        self.df = np.zeros((traj_count//2,4000))
        for i in range(20):
            self.x[:,200*i:200*i+200] = self.demo_t-0.5
            self.y[:,200*i:200*i+200] = self.demo_x

        self.xy = np.concatenate([self.x,self.y],axis=2)

        rand_vector = np.random.randn(traj_count//2,3800,4)*0.1
        self.xy[:,200:] += rand_vector
        for p in range(traj_count//2):
            self.df[p,200:] = distance.cdist(self.xy[p,200:],self.xy[p,:200]).min(axis=1)                
        self.df*=4
        self.df[self.df>1]=1
        
    def __len__(self):
        return self.traj_count//2
    
    def __getitem__(self,index):
        random_points_on_traj = np.random.choice(np.arange(200),200)
        random_points_not_on_traj = np.random.choice(np.arange(3800),1800)       
        
        x_mp = torch.from_numpy(np.vstack([self.xy[index,random_points_on_traj], 
                              self.xy[index,200+random_points_not_on_traj]])).float()
        
        shape_points_grid_ind = np.random.randint(65536, size=2048)
        shape_points = np.stack([shape_points_grid_ind//256, shape_points_grid_ind%256]).T
        image_values = self.demo_img[index, shape_points_grid_ind//256, shape_points_grid_ind%256]/255.0-0.5
        
        x_rgb = torch.from_numpy(shape_points).float()/256.0-0.5

        y = {'mp':torch.from_numpy(np.hstack([self.df[index,random_points_on_traj],
                                               self.df[index,200+random_points_not_on_traj]]).reshape(-1,1)).float(),
             'rgb':torch.from_numpy(image_values).float(),
             }
        observations =  {'coords_mp': x_mp,
                         'coords': x_rgb,
                         'rgb': y['rgb'],
                         'mp': y['mp'],
                         'instance_idx':torch.Tensor([index]).squeeze().long()}
    
        ground_truth = {'rgb':observations['rgb'],
                        'mp':observations['mp']}
        
        return observations, ground_truth
    
class Robot_Traj_Dataset_CNN_CNMP(Dataset):
    def __init__(self,task_path, traj_count,task_id=1):
        noj =6
        self.traj_count=traj_count
        self.demo_t = np.zeros((traj_count,100,1))
        self.demo_q = np.zeros((traj_count,100,noj))
        self.demo_img = np.zeros((traj_count,256,256,3))
        for p in range(traj_count):
            with open("demo_data/"+task_path+"/"+str(p)+"/data.pickle", 'rb') as handle:
                data = pickle.load(handle)
            self.demo_t[p,:,0] = np.array(data['t'])-0.5
            self.demo_q[p] = np.array(data['joint_traj'])
            path_to_img = "demo_data/"+task_path+"/"+str(p)+"/img.png"
            if task_id==1:
                self.demo_img[p]=cv2.cvtColor(cv2.resize(cv2.imread(path_to_img,-1)[50:750,100:1200],(256,256)), cv2.COLOR_BGR2RGB)            
            elif task_id==2:
                self.demo_img[p]=cv2.cvtColor(cv2.resize(cv2.imread(path_to_img,-1)[0:750,100:1200],(256,256)), cv2.COLOR_BGR2RGB)
    def __len__(self):
        return self.traj_count  
    
    def __getitem__(self,index):
        random_points_on_traj = np.random.choice(np.arange(100),np.random.randint(1,6))
        
        t_mp = torch.from_numpy(self.demo_t[index,random_points_on_traj]).float()
        q_mp = torch.from_numpy(self.demo_q[index,random_points_on_traj]).float()
        
        image = torch.from_numpy(self.demo_img[index]).float().permute(2,0,1)/255.0
        observations = torch.cat((t_mp,q_mp),dim=-1)
        
        return image,observations,t_mp,q_mp
    def get_test(self,index):
        t_mp = torch.from_numpy(self.demo_t[index]).float()
        q_mp = torch.from_numpy(self.demo_q[index]).float()
        t_mp_obs = torch.from_numpy(self.demo_t[index,:1]).float()        
        q_mp_obs = torch.from_numpy(self.demo_q[index,:1]).float()

        image = torch.from_numpy(self.demo_img[index]).float().permute(2,0,1)/255.0
        observations = torch.cat((t_mp_obs,q_mp_obs),dim=-1)
        return image.reshape(1,*image.shape),observations.reshape(1,*observations.shape), \
                t_mp.reshape(1,*t_mp.shape),q_mp.reshape(1,*q_mp.shape)
                                                
class Robot_Traj_Dataset_CNMP(Dataset):
    def __init__(self,task_path,traj_count, task_id = 1):
        noj =6
        self.traj_count=traj_count
        self.demo_t = np.zeros((traj_count,100,1))
        self.demo_q = np.zeros((traj_count,100,noj))
        self.demo_img = np.zeros((traj_count,256,256,3))
        self.demo_envs = list()
        for p in range(traj_count):
            with open("demo_data/"+task_path+"/"+str(p)+"/data.pickle", 'rb') as handle:
                data = pickle.load(handle)
            if task_id == 1:
                task_params = data['env_parameters']
            elif task_id == 2:
                task_params = list()
                task_params.append(data['env_parameters'][0][1][2])
                task_params.append(data['env_parameters'][1][1][2])
                task_params.append(data['env_parameters'][2][1][0])
            self.demo_envs.append(task_params)

            self.demo_t[p,:,0] = np.array(data['t'])-0.5
            self.demo_q[p] = np.array(data['joint_traj'])
            path_to_img = "demo_data/"+task_path+"/"+str(p)+"/img.png"
            if task_id==1:
                self.demo_img[p]=cv2.cvtColor(cv2.resize(cv2.imread(path_to_img,-1)[50:750,100:1200],(256,256)), cv2.COLOR_BGR2RGB)            
            elif task_id==2:
                self.demo_img[p]=cv2.cvtColor(cv2.resize(cv2.imread(path_to_img,-1)[0:750,100:1200],(256,256)), cv2.COLOR_BGR2RGB)            
        self.demo_envs=np.array(self.demo_envs)
        # scaling max and min to 1 and -1 for each parameter
        max_dif_min = (self.demo_envs.max(axis=0)-self.demo_envs.min(axis=0))
        self.demo_envs = (self.demo_envs - self.demo_envs.min(axis=0))/max_dif_min * 2-1 
        self.demo_envs = np.round(self.demo_envs,2) # To make it readable                                              
    def __len__(self):
        return self.traj_count  
    
    def __getitem__(self,index):
        random_points_on_traj = np.random.choice(np.arange(100),np.random.randint(1,6))
        
        t_mp = torch.from_numpy(self.demo_t[index,random_points_on_traj]).float()
        q_mp = torch.from_numpy(self.demo_q[index,random_points_on_traj]).float()
        m = np.array(self.demo_envs[index]).reshape(1,2)
        m = torch.tile(torch.from_numpy(m).float(),(len(random_points_on_traj),1))
        observations = torch.cat((m,t_mp,q_mp),dim=-1)
        
        return observations,t_mp,q_mp
    def get_test(self,index):
        t_mp = torch.from_numpy(self.demo_t[index]).float()
        q_mp = torch.from_numpy(self.demo_q[index]).float()
        t_mp_obs = torch.from_numpy(self.demo_t[index,:1]).float()        
        q_mp_obs = torch.from_numpy(self.demo_q[index,:1]).float()

        m = np.array(self.demo_envs[index]).reshape(1,2)
        m = torch.from_numpy(m).float()
        
        observations = torch.cat((m, t_mp_obs,q_mp_obs),dim=-1)
        return observations.reshape(1,*observations.shape), \
                t_mp.reshape(1,*t_mp.shape),q_mp.reshape(1,*q_mp.shape)      