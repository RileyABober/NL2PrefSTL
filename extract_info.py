import torch
import numpy as np

torch.set_default_dtype(torch.float32)

def get_speed(ego_data: list) -> torch.Tensor:
    '''
    Calculate the speed of the ego vehicle from the ego data.
    
    The speed is calculated as the norm of the difference between consecutive positions divided by the time step (0.1 seconds).
    The speed is then smoothed using a convolutional filter.
    The filter is a 1D convolutional filter with a kernel size of 15 and a stride of 1.
    
    input:
    - ego_data: list of tensors, each tensor is of shape (N, 2) where N is the number of time steps and 2 is the x and y coordinates of the ego vehicle.
    output:
    - speed_list: tensor of shape (N, max_length) where N is the number of ego vehicles and max_length is the maximum length of the ego data.
    '''
    filt = 1 / 15 * torch.ones((1, 1, 15))  # filtered instantaneous changes in CARLA speed data

    N = len(ego_data)
    max_length = max([len(ego_data[k]) for k in range(N)])
    speed_list = []
    limit = 1
    
    for k in range(N):
        velocity = (ego_data[k][:-1, :] - ego_data[k][1:, :]) / 0.1
        speed = torch.linalg.norm(velocity, axis=-1, keepdim=True)
        #smooth the speed data, original is bumpy
        #first remove irregularities by repeating data
        '''for i in range(len(speed) - 1):
            if speed[i] - speed[i + 1] > limit:
                speed[i + 1] = speed[i]'''
        #convolve to resmooth data
        smoothingWindow = 15
        sNP = np.convolve(np.ndarray.flatten(speed.numpy()), np.ones(smoothingWindow)/smoothingWindow, mode='valid')
        speed = torch.from_numpy(np.reshape(sNP, newshape=(len(sNP), 1)))
        
        # padding the last value to the end of the sequence to make all sequences the same length
        speed_list.append(speed)
    
    return speed_list

def get_acceleration(ego_data: list) -> torch.Tensor:
    '''
    Calculate the acceleration of the ego vehicle from the ego data.
    The acceleration is calculated as the norm of the difference between consecutive (smoothened) speeds divided by the time step (0.1 seconds).

    input:
    - ego_data: list of tensors, each tensor is of shape (N, 2) where N is the number of time steps and 2 is the x and y coordinates of the ego vehicle.
    
    output:
    - acceleration: tensor of shape (N, max_length) where N is the number of ego vehicles and max_length is the maximum length of the ego data.
    '''
    N = len(ego_data)
    max_length = max([len(ego_data[k]) for k in range(N)])
    acceleration = []
    
    speed = get_speed(ego_data)
    limit = 1

    for k in range(N):
        speed_k = speed[k]
        unSmoothedAcceleration = (speed_k[:-1] - speed_k[1:]) / 0.1
        #convolve to resmooth data
        smoothingWindow = 15
        aNP = np.convolve(np.ndarray.flatten(unSmoothedAcceleration.numpy()), np.ones(smoothingWindow)/smoothingWindow, mode='same')
        acceleration.append(torch.from_numpy(aNP))
    return acceleration

def get_jerk(ego_data:list)->torch.Tensor:
    '''
    Calculate the jerk of the ego vehicle from the ego data.
    The jerk is calculated as the norm of the difference between consecutive accelerations divided by the time step (0.1 seconds).
    
    input:
    - ego_data: list of tensors, each tensor is of shape (N, 2) where N is the number of time steps and 2 is the x and y coordinates of the ego vehicle.
    
    output:
    - jerk: tensor of shape (N, max_length) where N is the number of ego vehicles and max_length is the maximum length of the ego data.
    '''
    N = len(ego_data)
    max_length = max([len(ego_data[k]) for k in range(N)])
    jerk = []
    
    acceleration = get_acceleration(ego_data)
    for k in range(N):
        accel = acceleration[k]
        unSmoothedJerk = (accel[:-1] - accel[1:]) / 0.1
        #convolve to smooth data
        smoothingWindow = 15
        jNP = np.convolve(np.ndarray.flatten(unSmoothedJerk.numpy()), np.ones(smoothingWindow)/smoothingWindow, mode='same')
        jerk.append(torch.from_numpy(jNP))

    return jerk

def get_distance(ego_data, ado_data):
    '''
    Calculate the distance between the ego vehicle and the ado vehicle.
    The distance is calculated as the norm of the difference between the ego vehicle and the ado vehicle positions.
    
    input:
    - ego_data: list of tensors, each tensor is of shape (N, 2) where N is the number of time steps and 2 is the x and y coordinates of the ego vehicle.
    - ado_data: list of tensors, each tensor is of shape (N, 2) where N is the number of time steps and 2 is the x and y coordinates of the ado vehicle.
    
    output:
    - distance: tensor of shape (N, max_length) where N is the number of ego vehicles and max_length is the maximum length of the ego data.
    '''
    N = len(ego_data)
    max_length = max([len(ego_data[k]) for k in range(N)])
    distance = torch.ones(size=(N, max_length))
    
    for k in range(N):
        dist = torch.linalg.norm(ego_data[k] - ado_data[k], axis=-1, keepdim=True)
        # padding the last value to the end of the sequence to make all sequences the same length
        distance[k,:] = torch.cat((dist, dist[-1]*torch.ones(size=(max_length - dist.shape[0],1))),axis=0).squeeze(-1)
    return distance


def get_relative_speed(ego_data, ado_data):
    N = len(ego_data)
    max_length = max([len(ego_data[k]) for k in range(N)])
    relative_speed = torch.ones(size=(N, max_length))
    
    for k in range(N):
        velocity = (ego_data[k][:-1, :] - ego_data[k][1:, :]) / 0.1
        ado_velocity = (ado_data[k][:-1, :] - ado_data[k][1:, :]) / 0.1

        relative_s = torch.linalg.norm(velocity - ado_velocity, axis=-1).unsqueeze(-1)
        relative_speed[k, :] = torch.cat((relative_s, relative_s[-1,:]* torch.ones(size=(max_length - relative_s.shape[0], 1))),axis=0).squeeze(-1)
        #convolving
        smoothingWindow = 15
        jNP = np.convolve(np.ndarray.flatten(relative_speed[k].numpy()), np.ones(smoothingWindow)/smoothingWindow, mode='same')
        relative_speed[k,:] = torch.from_numpy(jNP)
    return relative_speed

def get_longitudinal_speed(ego_data, ado_data):
    N = len(ego_data)
    max_length = max([len(ego_data[k]) for k in range(N)])
    longitudinal = torch.ones(size=(N, max_length))
    
    for k in range(N):
        lead2ego = ado_data[k] - ego_data[k]
        yaws = torch.atan2(
            (ego_data[k][:-1, :] - ego_data[k][1:, :])[:, 1],
            (ego_data[k][:-1, :] - ego_data[k][1:, :])[:, 0],
            )
        e_longitudinal = torch.cat((-torch.sin(yaws).unsqueeze(-1), torch.cos(yaws).unsqueeze(-1)), axis=-1)

        long = torch.abs(torch.einsum("ik,ik->i", lead2ego[:-1, :], e_longitudinal))
        longitudinal[k, :] = torch.cat((long,long[-1]* torch.ones(max_length - long.shape[0])),axis=0)
    
    return longitudinal
    
def get_lateral_speed(ego_data, ado_data):
    N = len(ego_data)
    max_length = max([len(ego_data[k]) for k in range(N)])
    lateral = torch.ones(size=(N, max_length))
    
    for k in range(N):
        lead2ego = ado_data[k] - ego_data[k]
        yaws = torch.atan2(
            (ego_data[k][:-1, :] - ego_data[k][1:, :])[:, 1],
            (ego_data[k][:-1, :] - ego_data[k][1:, :])[:, 0],
            )
        e_lateral = torch.cat((torch.cos(yaws).unsqueeze(-1), torch.sin(yaws).unsqueeze(-1)), axis=-1)
        lat = torch.abs(torch.einsum("ik,ik->i", lead2ego[:-1, :], e_lateral))
        lateral[k, :] = torch.cat((lat, lat[-1] * torch.ones(max_length - lat.shape[0])),axis=0)
        
    return lateral
    
    