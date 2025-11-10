import re
import ollama
import pickle
import pandas
import time

import STLrobustness as robustness
from prompt_gen import *

class Model:
    '''
    LLM functionality and chat logging
    
    Attributes:
        model: name of model in ollama
        client: ollama client for model
        messages: log of messages
        file: file to write messages and responses to

    Methods:
        set_file: set output file
        chat: recieve LLM response from new input and previous inputs
    '''
    def __init__(self, model, file=None):
        '''initialize LLM model'''
        self.model = model
        self.client = ollama.Client()
        self.messages = []
        self.file = file
    
    def set_file(self, file):
        '''Set file for writing messages'''
        self.file = file
    
    def chat(self, message, description=None):
        '''Send message to LLM and recieve response'''
        formated_message = {'role': "user", 'content': message}
        self.messages.append(formated_message)
        try:
            system_message = self.client.chat(model=self.model, messages=self.messages)
        except Exception:
            assert False, "failure to communicate with ollama client"

        self.messages.append(system_message['message'])

        if self.file is not None:
            self.file.write(f"########## written by: {formated_message['role']} ##########\n")
            if description is not None:
                self.file.write(f"\n\n{description}\n\n")
            self.file.write(formated_message['content'] + "\n\n")

            self.file.write(f"########## written by: {system_message['message']['role']} ##########\n")
            self.file.write(f"{system_message['message']['content']}\n\n")

        return system_message['message']['content']


def NL2PrefSTL(preferences, reasonings, ego_name, ado_name, data, scenario, stride=7, 
               model="deepseek-r1:70b", prompt_limit=3, seed=None, 
               LLM_log=None, STL_log=None) -> tuple[robustness.Signal_WSTL, ...]:
    '''
    Provided preferences, reasonings, and information about the scenario, returns
    STL that is determined by the model of choice to suit the user's preferences

    Args:
        preferences: list or tuple of t1, t2 integer pairs where t1 over t2
        reasonings: list or tuple of strings corresponding to each preference
        ego_name: name of ego vehicle
        ado_name: name of other vehicle / object
        data: data of form trajectory, signal_name, torch.Tensor key, key, value pairs
        scenario: name of scenario
        stride: selects every <stride> datapoint for LLM inputs
        model: model as can be detected by Ollama client
        prompt_limit: maximum number of error calls
        LLM_log: file for recording LLM - Program interaction
        STL_log: file for recording produced STL strings

    Returns:
        (Signal_WSTL,...): tuple of Signal_WSTL objects
    '''

    timer = time.time()

    assert(len(preferences) == len(reasonings)), \
           f"number of preferences ({len(preferences)}) unequal to" \
           f"the number of reasonings ({len(reasonings)})\n"

    horizon = 0
    for trajectory in data:
        key = list(data[trajectory].keys())[0]
        l = len(data[trajectory][key])
        if l > horizon:
            horizon = l

    stl_list = []
    valid_data = []
    for key in data[0]:
        valid_data.append(key)
    
    STL_strs = []
    for preference in range(len(preferences)):
        confirmed = False
        print(f"######################## - {preference} - ########################\n")
        print(f"stl: {STL_strs}\n\n")

        client = Model(model=model, file=LLM_log) #LLM
        
        prompt = data_prompt(preferences[preference], reasonings[preference], 
                             ego_name, ado_name, data, stride, scenario, 
                             previous_STL=STL_strs)
        print(f"data prompt sent: {time.time() - timer}")
        timer = time.time()

        client.chat(prompt, f"\n\nDATA PROMPT #{preference}")

        prompt = pre_order_prompt(3)

        print(f"pre-order prompt sent: {time.time() - timer}")
        timer = time.time()
        strSTL = ""
        stl = None
        for i in range(prompt_limit + 1):
            propSTL = client.chat(prompt, f"GENERATION PROMPT {i}")
            valid, description = find_STL(propSTL)
            strSTL = description

            if valid:
                stl, description = robustness.check_stl(description, valid_data, horizon=horizon)

            if stl is None:
                print(f"stl: {description} \n")
                prompt = error_prompt(description)
                print(f"error prompt {i} sent: {time.time() - timer}")
                timer = time.time()
            else:
                check_prompt = STL_check_prompt(strSTL)
                confirmation = client.chat(check_prompt, "CONFIRMATION PROMPT")
                if len(re.findall("yes", confirmation, flags=re.IGNORECASE)) > 0:
                    confirmed = True
                    print("confirmed\n")
                    break
                else:
                    print("rejected\n")
                    prompt = confirmation
                    
        if not confirmed:
            print(f"Warning: On preference {preferences} stl failed to generate\n")
            if STL_log is not None:
                STL_log.write(f"{preference}: failed\n")
        else:
            if STL_log is not None:
                STL_log.write(f"{preference} + :  + {description}\n")
            STL_strs.append(strSTL)
            stl_list.append(stl)

    return tuple(stl_list)

def NL2STL_experiment(data, train_preferences, train_reasonings, 
                      test_preferences, test_reasonings, scenario, 
                      samples, accuracy_metric, seed, LLM_log=None, STL_log=None):
    '''
    Generates and trains wstl then tests accuracy

    Args:
        data: data of form trajectory, signal_name, torch.Tensor key, key, value pairs
        train_preferences: preference pairs for training
        train_reasonings: associated reasonings for train preferences
        test_preferences: preference pairs for testing
        test_reasonings: associated reasonings for test preferences
        scenario: "overtake" or "pedestrian"
        samples: number of weight evaluations
        accuracy_metric: metric from Metric class
        seed: for deterministic testing
        LLM_log: file for recording LLM - program interaction
        STL_log: file for recording STL strings

    Returns:
        tuple: (accuracy, train accuracy, test accuracy)
    '''
    assert scenario in ["overtake", "pedestrian"], "invalid scenario"
    assert isinstance(accuracy_metric, robustness.Metric)

    ego_name, ado_name = ""

    match scenario:
        case "overtake":
            ego_name = "ego car"
            ado_name = "overtaken car"
        case "pedestrian":
            ego_name = "ego car"
            ado_name = "pedestrian"
    
    wstl_list = NL2PrefSTL(train_preferences, train_reasonings, ego_name, ado_name, 
                      data, scenario, LLM_log=LLM_log, STL_log=STL_log)
    
    if len(wstl_list) == 0:
        return 0 

    wstl = robustness.link_STL(wstl_list)

    wstl.set_weights(data, samples=samples, seed=seed)
    evaluations = wstl.robustness(data)
    metric_evaluation = accuracy_metric(evaluations)

    
    train_accuracy = robustness.metric_accuracy(train_preferences, metric_evaluation)
    test_accuracy = robustness.metric_accuracy(test_preferences, metric_evaluation)
    accuracy = robustness.metric_accuracy(train_preferences + test_preferences,
                                          metric_evaluation)
    return (accuracy, train_accuracy, test_accuracy)

if __name__ == '__main__':
    experiment = 'overtake' # name of the experiment
    data_name = experiment + "_trajectories.pkl"
    #pref_name = experiment + "_pref.csv"
    pref_name = "hilda_overtake.csv"
    preference_name = "preferences.csv"
    #trajectories
    with open(data_name, "rb") as f:
        pklDat = pickle.load(f)

    #preferences & reasonings
    with open(pref_name) as f:
        prefDat = pandas.read_csv(f)
    
    #format trajectory data correctly
    data = {}
    for trajectory in range(len(pklDat["ego_trajectory"])):
        data[trajectory] = trajectory_data({"ego_trajectory": [pklDat["ego_trajectory"][trajectory]], 
                                            "ado_trajectory": [pklDat["ado_trajectory"][trajectory]]})
        
    
    preferences = []
    reasonings = []
    #format preference data correctly
    for pref in range(len(prefDat["Trajectory 1"])):
        pair = [prefDat["Preference"][pref], None]
        if pair[0] == prefDat["Trajectory 1"][pref]:
            pair[1] = prefDat["Trajectory 2"][pref]
        else:
            pair[1] = prefDat["Trajectory 1"][pref]

        preferences.append(pair)
        reasonings.append(prefDat["Reasoning"][pref])

    ego_name = "ego car"
    ado_name = "overtaken car"
    scenario = experiment

    LLM_log = open("ollama_log.txt", 'w')
    STL_log = open("STL_log.txt", 'w')
    
    stl = NL2PrefSTL(preferences=preferences, reasonings=reasonings, ego_name=ego_name, ado_name=ado_name,
                      data=data, scenario=scenario, stride=7, prompt_limit=3, LLM_log=LLM_log, STL_log=STL_log)

    for i in stl:
        print(i.wstl)
