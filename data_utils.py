import pandas
import pickle
from prompt_gen import trajectory_data

def process_preferences(file):
    '''return formated preferences and reasonings from file name'''
    with open(file) as f:
        data = pandas.read_csv(f)
    

    preferences = []
    reasonings = []

    for pref in range(len(data["Trajectory 1"])):
        pair = [data["Preference"][pref], None]
        if pair[0] == data["Trajectory 1"][pref]:
            pair[1] = data["Trajectory 2"][pref]
        else:
            pair[1] = data["Trajectory 1"][pref]

        preferences.append(pair)
        reasonings.append(data["Reasoning"][pref])

    return (preferences, reasonings)

def process_trajectory_data(file):
    '''return formated data and reasonings from file name'''
    with open(file, "rb") as f:
        pkl_data = pickle.load(f)

    data = {}
    for trajectory in range(len(pkl_data["ego_trajectory"])):
        data[trajectory] = trajectory_data({"ego_trajectory": [pkl_data["ego_trajectory"][trajectory]], 
                                            "ado_trajectory": [pkl_data["ado_trajectory"][trajectory]]})
        
    return data
