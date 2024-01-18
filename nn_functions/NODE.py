import numpy as np
import yaml
import os
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from bisect import bisect
from time import time

from data.data_functions.get_data import divide_in_batches, cart_data
from data.data_functions.utils import array_to_tensor,first_index
from plot_functions.trajectory_plot import plot_batch


def system(whole_data,hidden_size1,hidden_size2,sim,method):
    class System(nn.Module):
        def __init__(self, hidden_size1, hidden_size2):
            super().__init__()
            self.odeint = odeint
            self.L_track,self.kappa_track,self.n_track,self.l_r,self.l_f,self.V_max,self.delta_max = self.track_and_car_data()
            self.s_max,self.e_max,self.dth_max = self.normalization_params()
            self.nominal = False
            self.fc1 = torch.nn.Linear(5, hidden_size1)            
            self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = torch.nn.Linear(hidden_size2, 3)
            self.tanh = torch.nn.Tanh()
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
        def MLP(self,input):
            x = self.fc1(input)
            x = self.tanh(x)
            x = self.fc2(x)
            x = self.tanh(x)
            x = self.fc3(x)
            x = self.tanh(x)
            return(x)
        def track_and_car_data(self):
            base_dir = os.path.dirname(os.path.realpath(__file__))
            la_track = os.path.join(base_dir, 'data', 'data_files', 'la_track.yaml')
            # la_track = r'C:\Users\Oliver Ljungberg\Desktop\EPFL\MA3\Semester_Project\Code\data_functions\la_track.yaml'
            with open(la_track) as file:
                docs = yaml.safe_load_all(file)
                for doc in docs:
                    L_track = doc['track']['trackLength']
                    kappa_track = doc['track']['curvature']
            n_track = int(len(kappa_track)/2) # values repeat once inside of la_track
            l_r = 0.051
            l_f = 0.047
            V_max = 8.6695
            delta_max = 0.19199
            return(L_track,kappa_track,n_track,l_r,l_f,V_max,delta_max)
        def normalization_params(self):
            s_max = self.L_track
            e_max = 0.5
            dth_max = 20*torch.pi/180
            return(s_max,e_max,dth_max)
        def nominal_ODE(self, states, controls):
            s,e,delta_theta = states
            delta_d,delta_V = controls
            s_cut = s%self.L_track
            delta_theta = (delta_theta+torch.pi)%(2*torch.pi)-torch.pi
            idx = int(s_cut*self.n_track/self.L_track)
            kappa = self.kappa_track[idx]
            V = self.V_max*delta_V
            delta = self.delta_max*delta_d
            beta = torch.atan((self.l_r*torch.tan(delta))/(self.l_r+self.l_f))
            s_dot = V*torch.cos(delta_theta+beta)/(1-kappa*e)
            e_dot = V*torch.sin(delta_theta+beta)
            delta_theta_dot = V/self.l_r*torch.sin(beta)-kappa*s_dot # = V/(self.l_f+self.l_r)*torch.cos(beta)*torch.tan(delta)-kappa*s_dot
            states_dot = torch.tensor([s_dot,e_dot,delta_theta_dot])
            return(states_dot)
        def normalize_states(self, states):
            s_norm = states[0]/self.s_max
            e_norm = states[1]/self.e_max
            dth_norm = states[2]/self.dth_max
            norm_states = torch.stack([s_norm, e_norm, dth_norm], dim=0)
            return(norm_states)
        def denormalize_states(self, states):
            s_denorm = states[0]*self.s_max
            e_denorm = states[1]*self.e_max
            dth_denorm = states[2]*self.dth_max
            denorm_states = torch.stack([s_denorm, e_denorm, dth_denorm], dim=0)
            return(denorm_states)
        def forward(self, t, states, controls):
            states_dot = self.nominal_ODE(states, controls)
            states_dot_corr = 0
            if not self.nominal:
                norm_states = self.normalize_states(states)
                NN = self.MLP(torch.cat([norm_states,controls]).float())
                states_dot_corr = self.denormalize_states(NN)
            new_states_dot = states_dot + states_dot_corr
            return new_states_dot

    class Actuators(nn.Module):
        def __init__(self, ctl_var, sim):
            super().__init__()
            self.delta_d,self.delta_V,self.t = ctl_var
            self.sim = sim
            self.comm_delay_time = 0.04
        def forward(self, t):
            if self.sim:
                i = bisect(np.array(self.t),t)
            else:     
                i = bisect(np.array(self.t),t-self.comm_delay_time)           
            if i:
                idx = i-1
            else:
                idx = i 
            return torch.tensor([self.delta_d[idx],self.delta_V[idx]])
        
    class Controller(nn.Module):
        def __init__(self):
            super().__init__()
            self.TC_d = 0.0434
            self.TC_V = 0.3340
            self.odeint = odeint
        def forward(self, t, controls, actuators):
            delta_d, delta_V = controls
            delta_d_cmd, delta_V_cmd = actuators
            delta_d_dot = (delta_d_cmd-delta_d)/self.TC_d
            delta_V_dot = (delta_V_cmd-delta_V)/self.TC_V
            controls_dot = torch.tensor([delta_d_dot,delta_V_dot])
            return controls_dot

    class ControlledSystem(nn.Module):
        def __init__(self, sys, ctl):
            super().__init__()
            self.sys = sys
            self.ctl = ctl
        def forward(self, t, x):
            u = self.ctl(t)
            x_ = self.sys(t, x, u)
            return x_
        
    ctl_var = whole_data[3:]
    if sim:
        ctl = Actuators(ctl_var,sim)
        sys = System(hidden_size1,hidden_size2,sim)
        clsys = ControlledSystem(sys,ctl)
        ode_ctl = []
    else:
        idx0 = first_index(whole_data[-1])
        ctl_0 = whole_data[3:5,idx0]
        t = whole_data[-1]
        act = Actuators(ctl_var,True)
        ctl = Controller()
        act_dyn = ControlledSystem(ctl,act)
        ode_ctl = odeint(act_dyn,ctl_0,t,method=method)
        ode_ctl = torch.transpose(ode_ctl,dim0=0,dim1=1)
        ode_ctl_var = torch.cat([ode_ctl,t.unsqueeze(0)])
        act_ctl = Actuators(ode_ctl_var,sim)
        sys = System(hidden_size1,hidden_size2)
        clsys = ControlledSystem(sys,act_ctl)
    return(clsys,ode_ctl)


def training(whole_data,lr,epochs,hidden_size1,hidden_size2,method,l_batch=50,n_batches=0,sim=True,noise=True,rand_perm=True,val_ratio=0.2):
    time0 = time()

    class CustomLoss(nn.Module):
        def __init__(self):
            super().__init__()
            la_track = r'C:\Users\Oliver Ljungberg\Desktop\EPFL\MA3\Semester_Project\Code\data_functions\la_track.yaml'
            with open(la_track) as file:
                docs = yaml.safe_load_all(file)
                for doc in docs:
                    L_track = doc['track']['trackLength']
            self.s_max = L_track
            self.e_max = 0.3
            self.dth_max = 2*torch.pi
        def forward(self,predictions,targets):
            err = torch.zeros(targets.size())
            err[:,0] = torch.abs(torch.remainder(targets[:,0]-predictions[:,0]+self.s_max/2,self.s_max)-self.s_max/2)/(self.s_max/2)
            err[:,1] = torch.abs(targets[:,1]-predictions[:,1])/self.e_max
            err[:,2] = torch.abs(torch.remainder(targets[:,2]-predictions[:,2]+self.dth_max/2,self.dth_max)-self.dth_max/2)/(self.dth_max/2)
            sq_err = err ** 2
            return torch.sum(torch.mean(sq_err,0))
    
    clsys,ode_ctl = system(whole_data,hidden_size1=hidden_size1,hidden_size2=hidden_size2,sim=sim,method=method)
    
    # Define loss function and optimizer
    # lossFunc = nn.MSELoss()
    lossFunc = CustomLoss()
    optimizer = torch.optim.Adam(clsys.parameters(), lr=lr)

    # Create batches for training and validation
    cat_batches,train_fren = divide_in_batches(whole_data,l_batch,n_batches,rand_perm)
    if noise:
        s_max = clsys.sys.s_max
        e_max = clsys.sys.e_max
        dth_max = clsys.sys.dth_max
        n = len(train_fren[0])
        mean = torch.zeros(3,n)
        weight = torch.tensor([s_max,e_max,dth_max])
        std = torch.ones(3,n)*weight.view(-1,1)*0.01
        train_fren[:3] += torch.normal(mean,std)
    val_idx = int(len(cat_batches)*(1-val_ratio))
    train_batches = cat_batches[:val_idx]
    val_batches = cat_batches[val_idx:]

    # Define initial conditions    
    cat_train_loss = torch.empty(0)
    cat_val_loss = torch.empty(0)
    for i in range(epochs):
        Loss = 0
        VLoss = 0
        if not i:
            with torch.no_grad():
                for batch in train_batches:
                    x_0 = batch[:3,0]
                    t = batch[-1]
                    x = odeint(clsys, x_0, t, method=method)
                    x_measured = batch[:3].transpose(0,1)
                    Loss += lossFunc(x,x_measured).item()
                for batch in val_batches:
                    val_x_0 = batch[:3,0]
                    t = batch[-1]
                    x = odeint(clsys, val_x_0, t, method=method)
                    x_measured = batch[:3].transpose(0,1)
                    VLoss += lossFunc(x,x_measured).item()
        else:
            for batch in train_batches:
                optimizer.zero_grad()
                x_0 = batch[:3,0]
                t = batch[-1]
                x = odeint(clsys, x_0, t, method=method)
                x_measured = batch[:3].transpose(0,1)
                loss = lossFunc(x,x_measured)
                loss.backward()
                Loss += loss.item()
                optimizer.step()
            clsys.eval()
            with torch.no_grad():
                for batch in val_batches:
                    val_x_0 = batch[:3,0]
                    t = batch[-1]
                    x = odeint(clsys, val_x_0, t, method=method)
                    x_measured = batch[:3].transpose(0,1)
                    val_loss = lossFunc(x,x_measured)
                    VLoss += val_loss.item()
            clsys.train()
        act_time = time()-time0
        tot_time = int(act_time*epochs/(i+1))
        rem_time = int(tot_time-act_time)
        tot_h,tot_m,tot_s = tot_time//3600,tot_time//60,tot_time%60
        rem_h,rem_m,rem_s = rem_time//3600,rem_time//60,rem_time%60
        cat_train_loss = torch.cat((cat_train_loss, torch.tensor([Loss/len(train_batches)])))
        cat_val_loss = torch.cat((cat_val_loss, torch.tensor([VLoss/len(val_batches)])))
        if tot_h:
            if tot_m//10:
                tot_str = f' - Estimated Total Time: {tot_h}h{tot_m}  '
            else:
                tot_str = f' - Estimated Total Time: {tot_h}h0{tot_m}  '
        elif tot_m:
            tot_str = f' - Estimated Total Time: {tot_m} min  '
        else:
            tot_str = f' - Estimated Total Time: {tot_s} sec  '        
        if rem_h:
            if rem_m//10:
                rem_str = f' - Estimated Remaining Time: {rem_h}h{rem_m}'
            else:
                rem_str = f' - Estimated Remaining Time: {rem_h}h0{rem_m}'
        elif rem_m:
            rem_str = f' - Estimated Remaining Time: {rem_m} min'
        else:
            rem_str = f' - Estimated Remaining Time: {rem_s} sec'
        str = f'Epoch: [{i+1}/{epochs}] - Training Loss: {cat_train_loss[-1]:.5f} - Validation Loss: {cat_val_loss[-1]:.5f}'+rem_str+tot_str
        print(str, end="\r", flush=True)
    return(cat_train_loss.tolist(),cat_val_loss.tolist(),clsys,array_to_tensor(train_fren),ode_ctl)


def predict(clsys, train_fren, method, sim=True, l_batch=50):
    node_cat = torch.empty(0)
    nom_cat = torch.empty(0)
    comm_delay_margin = torch.nonzero(train_fren[-1]==0).squeeze()
    if comm_delay_margin == []:
        comm_delay_margin = 0
    n = int((len(train_fren[-1])-comm_delay_margin)/l_batch)
    for i in range(n):
        states = train_fren[:3,i*l_batch+comm_delay_margin]
        t = train_fren[-1,i*l_batch+comm_delay_margin:(i+1)*l_batch+comm_delay_margin]
        clsys.sys.nominal = False
        x_nn = odeint(clsys,states.squeeze(),t,method=method)
        clsys.sys.nominal = True
        x_nom = odeint(clsys,states.squeeze(),t,method=method)
        node_cat = torch.cat((node_cat,x_nn))
        nom_cat = torch.cat((nom_cat,x_nom))
    node_cat = torch.transpose(node_cat,dim0=0,dim1=1)
    nom_cat = torch.transpose(nom_cat,dim0=0,dim1=1)
    node_cart = cart_data(sim,node_cat)
    nom_cart = cart_data(sim,nom_cat)
    train_cart = cart_data(sim,train_fren[:,comm_delay_margin:n*l_batch+comm_delay_margin])
    return(array_to_tensor(node_cart,nom_cart,train_cart))
