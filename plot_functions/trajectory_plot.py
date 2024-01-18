import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
from time import time

import plot_functions.mpctools as mpc
from data_functions.get_data import track_data, cart_data

def states_and_controls(sim):
    if sim:
        cart_states = [r'$\bm{x}$',r'$\bm{y}$',r'$\bm{\theta}$']
        fren_states = [r'$\bm{s}$',r'$\bm{e}$',r'$\bm{\Delta\theta}$']
        controls = [r'$\bm{\delta_d}$',r'$\bm{\delta_V}$']
        n_states = 3
        n_controls = 2
    else:
        cart_states = [r'$\bm{x}$',r'$\bm{y}$',r'$\bm{\theta}$']
        fren_states = [r'$\bm{s}$',r'$\bm{e}$',r'$\bm{\Delta\theta}$']
        controls = [r'$\bm{\delta_{d,cmd}}$',r'$\bm{\delta_{V,cmd}}$',r'$\bm{\delta_d}$',r'$\bm{\delta_V}$']
        n_states = 3
        n_controls = 4        
    return(cart_states,fren_states,controls,n_states,n_controls)


def plot_states_and_controls(cat_states_gt,cat_states_nn,cat_states_nom,cat_controls,t,sim,
                            l_batch_train,l_batch_pred,start=0,end=0,plot_nn=True,plot_track=True,ode_ctl=[]):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

    comm_delay_margin = torch.nonzero(t==0).squeeze()
    if comm_delay_margin == []:
        comm_delay_margin = 0
    if not end or end>len(t)-comm_delay_margin:
        end = len(t)-comm_delay_margin

    states,_,controls,n_states,n_controls = states_and_controls(sim)
    x_track_rot,y_track_rot,_,_ = track_data(sim)

    fig = plt.figure(figsize=(12,6))
    
    grid = max(n_states,n_controls)
    gs = gridspec.GridSpec(grid,4)

    timexax = []
    for i in range(n_states):
        timexax.append(fig.add_subplot(gs[i,0]))
        timexax[-1].set_ylabel(states[i],rotation=0)
    timexax[-1].set_xlabel('Time [s]')
    timexax[0].set_title('Cartesian States')

    timeuax = []
    for i in range(n_controls):
        timeuax.append(fig.add_subplot(gs[i,1]))
        timeuax[-1].set_ylabel(controls[i],rotation=0)
    timeuax[-1].set_xlabel('Time [s]')
    timeuax[0].set_title('Controls')

    fig.suptitle("NODE model converted to Cartesian frame\n sim = {}, l_batch_train = {}, l_batch_pred = {}"
                 .format(sim,l_batch_train,l_batch_pred),weight='bold')

    tplot = t[start:comm_delay_margin+end]
    if sim:
        uplot = cat_controls[:,start:end-1]
    else:
        if not len(ode_ctl):
            ode_ctl = torch.zeros(2,len(t))
        uplot = torch.cat([cat_controls[:,start:comm_delay_margin+end-1],ode_ctl[:,start:comm_delay_margin+end-1]])        
    splot = cat_states_gt[:,start:end]
    nnplot = cat_states_nn[:,start:end]
    nomplot = cat_states_nom[:,start:end]
    trrplot = np.array([x_track_rot,y_track_rot])

    if sim:
        str = "Ros data (sim)"
    else:
        str = "Ros data (real)"

    for i in range(n_controls):
        timeuax[i].step(tplot[:-1],uplot[i],color='C1')
    for i in range(3):
        timexax[i].plot(tplot[comm_delay_margin:],splot[i],color='C0',label=str)
        if plot_nn:
            timexax[i].plot(tplot[comm_delay_margin:],nomplot[i],color='C3',label='Nominal ODE')
            timexax[i].plot(tplot[comm_delay_margin:],nnplot[i],color='C2',label='NODE model')
        timexax[i].legend(loc='upper right')        

    planeax = fig.add_subplot(gs[:,2:])
    planeax.set_xlabel(states[0])
    planeax.set_ylabel(states[1],rotation=0)
    planeax.set_title(f'{states[0]}-{states[1]} Plane')
    if plot_track:
        planeax.plot(trrplot[0],trrplot[1],color='C1',label='Track center')
    planeax.plot(splot[0],splot[1],color='C0',label=str)
    if plot_nn:
        planeax.plot(nomplot[0],nomplot[1],color='C3',label='Nominal ODE')
        planeax.plot(nnplot[0],nnplot[1],color='C2',label='NODE model')

    # Some beautification.
    planeax.legend(loc='upper right')
    planeax.set_aspect('equal')
    for ax in timexax + timeuax:
        mpc.plots.zoomaxis(ax,yscale=1.1)
    fig.tight_layout(pad=.5)

    return


def plot_batch(batch,x,t_data,sim,epoch,loss,train=True):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

    cart_states,fren_states,_,n_states,_ = states_and_controls()

    fig = plt.figure(figsize=(6,3))
    gs = gridspec.GridSpec(3,2)

    timefax = []
    for i in range(n_states):
        timefax.append(fig.add_subplot(gs[i,0]))
        timefax[-1].set_ylabel(fren_states[i],rotation=0)
    timefax[-1].set_xlabel('Time [s]')
    timefax[0].set_title("Frenet's States")

    timecax = []
    for i in range(n_states):
        timecax.append(fig.add_subplot(gs[i,1]))
        timecax[-1].set_ylabel(cart_states[i],rotation=0)
    timecax[-1].set_xlabel('Time [s]')
    timecax[0].set_title("Cartesian States")

    tplot = batch[-1]
    bfplot = batch[:3]
    nfplot = np.transpose(x.detach().numpy())
    bcplot = cart_data(sim,batch[:3])
    ncplot = cart_data(sim,nfplot)

    l_batch = len(tplot)
    idx = np.where(t_data == tplot[0])[0]
    n = int(idx/l_batch)

    fig_type = 'Training' if train else 'Validation'
    fig.suptitle(f'Plots of {fig_type} Batch {n+1}, Epoch {epoch+1}, Loss = {loss}', weight='bold')

    for i in range(n_states):
        timefax[i].plot(tplot,bfplot[i],color='C0')
        timefax[i].plot(tplot,nfplot[i],color='C2')

    for i in range(n_states):
        timecax[i].plot(tplot,bcplot[i],color='C0')
        timecax[i].plot(tplot,ncplot[i],color='C2')

    # Some beautification.
    for ax in timefax + timecax:
        mpc.plots.zoomaxis(ax,yscale=1.1)
    fig.tight_layout(pad=.5)

    return


def simulate(cat_states, cat_controls, step_horizon=5e-3, n_frames=400, save=False):
    def create_triangle(state=[0,0,0], h=0.2, w=0.1, update=False):
        x, y, th = state
        triangle = np.array([
            [h, 0   ],
            [0,  w/2],
            [0, -w/2],
            [h, 0   ]
        ]).T
        rotation_matrix = np.array([
            [math.cos(th), -math.sin(th)],
            [math.sin(th),  math.cos(th)]
        ])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]

    def init():
        return path, current_state_s, current_state_t

    def animate(i):
        display_every = int(n_iter/n_frames)
        # get variables
        x = cat_states[0,display_every*i]
        y = cat_states[1,display_every*i]
        th = cat_states[2,display_every*i]
        st = cat_controls[0,display_every*i]*delta_max

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update current_state_t
        current_state_t.set_xy(create_triangle([x, y, th], update=True))

        # update current_state_s
        current_state_s.set_xy(create_triangle([x, y, th+st], h=0.15, w=0.1, update=True))

        # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)

        return path, current_state_s, current_state_t

    delta_max = 0.19199
    n_iter = len(cat_states[0])
    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    min_x = min(cat_states[0])
    min_y = min(cat_states[1])
    max_x = max(cat_states[0])
    max_y = max(cat_states[1])
    ax.set_xlim(left = min_x-0.5, right = max_x+0.5)
    ax.set_ylim(bottom = min_y-0.5, top = max_y+0.5)
    ax.set_aspect('equal')

    # create lines:
    #   path
    path, = ax.plot([], [], 'k', linewidth=2)

    #   current_state_theta
    current_triangle_t = create_triangle(cat_states[:3,0])
    current_state_t = ax.fill(current_triangle_t[:,0], current_triangle_t[:,1], color='r')
    current_state_t = current_state_t[0]

    #   current_state_steering   
    current_triangle_s = create_triangle([cat_states[0,0],cat_states[1,0],cat_states[2,0]+cat_controls[0,0]], h=0.15, w=0.1)
    current_state_s = ax.fill(current_triangle_s[:,0], current_triangle_s[:,1], color='g')
    current_state_s = current_state_s[0]
    
    #   target_state
    # target_triangle = create_triangle(reference[4:])
    # target_state = ax.fill(target_triangle[:, 0], target_triangle[:, 1], color='b')
    # target_state = target_state[0]

    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=n_frames,
        interval=step_horizon*1000,
        blit=True,
        repeat=True
    )
    plt.show()

    if save:
        sim.save('./animation' + str(time()) +'.gif', writer='ffmpeg', fps=30)

    return