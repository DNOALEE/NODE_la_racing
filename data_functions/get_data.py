import pandas as pd
from tqdm import tqdm
from time import time
import torch
import yaml

from data_functions.utils import first_index, array_to_tensor
from data_functions.change_of_coords_frame import cart_to_fren, fren_to_cart, frame_rotaton


def ros_data(index_0,sim):
    if sim:
        car_set_control = r'\\wsl.localhost\Ubuntu-20.04\home\oliver\ros2_ws\ros_bag\simcar2\simcar2_0\sim_car_set_control.csv'
        car_pose2d_raw = r'\\wsl.localhost\Ubuntu-20.04\home\oliver\ros2_ws\ros_bag\simcar2\simcar2_0\sim_car_state.csv'
    else:
        car_set_control = r'C:\Users\Oliver Ljungberg\Desktop\EPFL\MA3\Semester_Project\experiment_car_1_track_manual_var_speed\car_set_control.csv'
        car_pose2d_raw = r'C:\Users\Oliver Ljungberg\Desktop\EPFL\MA3\Semester_Project\experiment_car_1_track_manual_var_speed\car_state.csv'

    data = pd.read_csv(car_pose2d_raw)
    x = data.pos_x
    y = data.pos_y
    theta = data.turn_angle
    t1 = data.log_time_ns

    data = pd.read_csv(car_set_control)
    delta_d = data.steering
    delta_V = data.throttle
    t2 = data.log_time_ns

    if sim:
        steps = min(len(t1),len(t2)) - index_0
        x = x[-steps:]
        y = y[-steps:]
        theta = theta[-steps:]
        delta_d = delta_d[-steps:]
        delta_V = delta_V[-steps:]
        t1_0 = first_index(t1) + index_0
        t2_0 = first_index(t2) + index_0
    else:
        comm_delay_margin = 20
        index_0 = max(comm_delay_margin,index_0)
        steps = min(len(t1),len(t2)) - index_0
        x = x[-steps:]
        y = y[-steps:]
        theta = theta[-steps:]
        delta_d = delta_d[-steps-comm_delay_margin:]
        delta_V = delta_V[-steps-comm_delay_margin:]
        t1_0 = first_index(t1) + index_0 - comm_delay_margin
        t2_0 = first_index(t2) + index_0 - comm_delay_margin
        t1_1 = first_index(t1) + index_0
        t2_1 = first_index(t2) + index_0
        x,y,theta = array_to_tensor(x,y,theta)
        x = torch.cat([torch.zeros(comm_delay_margin),x])
        y = torch.cat([torch.zeros(comm_delay_margin),y])
        theta = torch.cat([torch.zeros(comm_delay_margin),theta])
    if len(t1)-index_0 == steps:
        t = (t1[t1_0:]-t1[t1_1])*1e-9
    else:
        t = (t2[t2_0:]-t2[t2_1])*1e-9    
    return(array_to_tensor(x,y,theta,delta_d,delta_V,t))


def fren_data(index_0=0,l_chunk=0,downsample=1,sim=True):
    t0 = time()
    print("Initialization: collecting the ros_bag and converting to Frenet's coordinates frame")
    s,e,delta_theta,cat_t = [],[],[],[]
    x,y,theta,delta_d,delta_V,t = ros_data(index_0,sim)
    comm_delay_margin = torch.nonzero(t==0).squeeze()
    if comm_delay_margin == []:
        comm_delay_margin = 0
    l_tot = len(x)
    if not l_chunk:
        l_chunk = l_tot
    x_track,y_track,theta_track,L_track = track_data(sim)

    progress_bar = tqdm(range(l_chunk), desc='Initialization', unit='datapoints')
    for i in range(l_chunk+comm_delay_margin):
        if i<comm_delay_margin:
            s_i,e_i,dth_i=0,0,0
        else:
            s_i,e_i,dth_i = cart_to_fren(x[i*downsample],y[i*downsample],theta[i*downsample],x_track,y_track,theta_track,L_track)
        s.append(s_i)
        e.append(e_i)
        delta_theta.append(dth_i)
        cat_t.append(t[i*downsample])
        progress_bar.update()
    steps = len(s)
    delta_d,delta_V = delta_d[:steps*downsample:downsample],delta_V[:steps*downsample:downsample]
    tot_time = time()-t0
    tot_h,tot_m,tot_s = tot_time//3600,tot_time//60,int(tot_time%60)
    if tot_h:
        if tot_m//10:
            time_str = f' {tot_h}h{tot_m}'
        else:
            time_str = f' {tot_h}h0{tot_m}'
    elif tot_m:
        if tot_s//10:
            time_str = f' {tot_m}min{tot_s}'
        else:
            time_str = f' {tot_m}min0{tot_s}'
    else:
        time_str = f' {tot_s}sec'
    str = 'Init done:'+time_str
    print(str)
    return(array_to_tensor(s,e,delta_theta,delta_d,delta_V,cat_t))


def cart_data(sim,fren):
    x,y,theta = [],[],[]
    s,e,delta_theta = fren[0],fren[1],fren[2]
    x_track,y_track,theta_track,L_track = track_data(sim)
    for i in range(len(s)):
        x_i,y_i,theta_i = fren_to_cart(s[i],e[i],delta_theta[i],x_track,y_track,theta_track,L_track)
        x.append(x_i)
        y.append(y_i)
        theta.append(theta_i)
    return(x,y,theta)


def divide_in_batches(whole_data,l_batch,n_batches,rand_perm):
    comm_delay_margin = torch.nonzero(whole_data[-1]==0).squeeze()
    if comm_delay_margin == []:
        comm_delay_margin = 0
    n_data_points = whole_data.size(1)-comm_delay_margin
    if n_data_points < l_batch*n_batches or not n_batches:
        n = n_data_points//l_batch
    whole_data_new = whole_data[:,:n*l_batch+comm_delay_margin]

    cat_batches = torch.empty(0)
    for i in range(n):
        cat_batches = torch.cat((cat_batches,whole_data_new[:,i*l_batch+comm_delay_margin:(i+1)*l_batch+comm_delay_margin].unsqueeze(0)))
    if rand_perm:
        cat_batches = cat_batches[torch.randperm(n), :, :]
    if n_batches and n_batches < n:
        cat_batches = cat_batches[:n_batches]
    return(cat_batches,whole_data_new)


def normalize_fren(whole_data,noise=False,norm=True):
    if norm:
        s_max = 10.710722313718323
        e_max = 0.5
        dth_max = 2*torch.pi
        s_norm = whole_data[0]/s_max
        e_norm = whole_data[1]/e_max
        dth_norm = whole_data[2]/dth_max
    else:
        s_norm = whole_data[0]
        e_norm = whole_data[1]
        dth_norm = whole_data[2]
    norm_data = torch.stack([s_norm,e_norm,dth_norm,whole_data[3],whole_data[4],whole_data[5]])
    n = len(s_norm)
    if noise:
        norm_data[:3] += torch.normal(torch.zeros(3,n),torch.ones(3,n)*0.01)
    return(norm_data)


def track_data(sim,transform=True):
    la_track = r'C:\Users\Oliver Ljungberg\Desktop\EPFL\MA3\Semester_Project\Code\data_functions\la_track.yaml'
    with open(la_track) as file:
        docs = yaml.safe_load_all(file)
        for doc in docs:
            x_track = doc['track']['xCoords']
            y_track = doc['track']['yCoords']
            theta_track = doc['track']['tangentAngle']
            L_track = doc['track']['trackLength']
    x_track0,y_track0 = x_track[0],y_track[0]
    if sim:
        dx,dy,dth = 0.1,-0.5,9*torch.pi/180
    else:
        dx,dy,dth = 1.425,-0.675,25.5*torch.pi/180
    x_track_centered = [x - x_track0 for x in x_track]
    y_track_centered = [y - y_track0 for y in y_track]
    x_track_new = []
    y_track_new = []
    theta_track_new = []
    mid = int(len(x_track)/2) # values repeat once inside of la_track
    if transform:
        for i in range(mid):
            xi,yi = frame_rotaton(0,0,x_track_centered[i],y_track_centered[i],dth)
            x_track_new.append(xi+dx)
            y_track_new.append(yi+dy)
            theta_track_new.append(theta_track[i]+dth)
    else:
        x_track_new,y_track_new,theta_track_new = x_track,y_track,theta_track
    return(x_track_new,y_track_new,theta_track_new,L_track)