import numpy as np
import math


def cart_to_fren(x,y,theta,x_track,y_track,theta_track,L_track):
    dist = []
    for i in range(len(x_track)):
        dist.append(np.linalg.norm([x-x_track[i],y-y_track[i]]))
    idx = dist.index(min(dist))
    x_t = x_track[idx]
    y_t = y_track[idx]
    theta_t = theta_track[idx]
    orth_t = (theta_t+math.pi/2)%(2*math.pi)
    s = L_track/len(x_track)*idx
    e,_ = frame_rotaton(x_t,y_t,x,y,-orth_t)
    delta_theta = (theta-theta_t)%(2*math.pi)
    return(s,e,delta_theta)


def fren_to_cart(s,e,delta_theta,x_track,y_track,theta_track,L_track):
    s_cut = s%L_track
    idx = int(s_cut*len(x_track)/L_track)
    x_t = x_track[idx]
    y_t = y_track[idx]
    theta_t = theta_track[idx]
    orth_t = (theta_t+math.pi/2)%(2*math.pi)
    dx,dy = frame_rotaton(0,0,e,0,orth_t)
    x,y,theta = x_t+dx,y_t+dy,(delta_theta+theta_t)%(2*math.pi)
    return(x,y,theta)


def frame_rotaton(x1,y1,x2,y2,theta):
    x = (x2-x1)*math.cos(theta)-(y2-y1)*math.sin(theta)
    y = (x2-x1)*math.sin(theta)+(y2-y1)*math.cos(theta)
    return(x,y)