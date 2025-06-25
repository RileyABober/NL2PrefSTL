import numpy as np
import ollama
import pickle
import pandas
import time

from STLrobustness import *
from extract_info import *
from prompt_gen import *

#This is the structure for the LLM
class Model:
    #model is a string representing the model name
    def __init__(self, model):
        self.model = model
        self.client = ollama.Client()
        self.messages = []
    
    #input message for the model, outputs a response with context
    #if out is assigned a file to write to then it will output the message to the file in question with relevant information
    def chat(self, message, out=None):
        formatedMessage = {'role': "user", 'content': message}
        self.messages.append(formatedMessage)
        systemMessage = self.client.chat(model=self.model, messages=self.messages)

        self.messages.append(systemMessage['message'])

        if out != None:
            out.write("########## written by: " + formatedMessage['role'] + " ##########\n")
            out.write(formatedMessage['content'] + "\n\n")

            out.write("########## written by: " + systemMessage['message']['role'] + " ##########\n")
            out.write(systemMessage['message']['content'] + "\n\n")

        return systemMessage['message']['content']

    #writes entire chat to a file with context of which party user system wrote which messages
    def outputChat(self, file):
        for message in self.messages:
            file.write("########## written by: " + message['role'] + " ##########\n")
            file.write(message['content'] + "\n\n")

#main file for running preference to stl & trajectory ordering

#preferences are of the form [accepted trajectory number, rejected trajectory number]
#reasonings are of the form "provided reasoning". Both are arrays of the same length
#ego_name is "ego car" for this study but can be changed to fit context
#ado_name is either "pedestrian" or "overtaken car" in this study but can be changed to fit context
#data is a dictionary of name, np array key value pairs
#stride represents the sampling of the provided data. A stride of 7 will use every 7th data point in the LLM prompt
#scenario is in this study "overtake" or "pedestrian" but can be changed to fit context
#model is the name of the LLM model which will be called through the Ollama client. In this study "deepseek-r1:70b" is used
#promptLimit is the maximum number of re-prompts to create valid STL before moving on to the next preference reasoning pair
#preOrderShots is the number of examples of STL to pre-order conversions provided to the STL (useful for syntactic correctness), 3 max 
def NL2PrefSTL(preferences, reasonings, ego_name, ado_name, data, scenario, stride=7, 
               model="deepseek-r1:70b", promptLimit=5, preOrderShots=3):
    #############This file provides a log of the input - output of the model###############
    logFileName = "ollamaLog.txt"
    logFile = open(logFileName, "w", encoding='utf-8')
    timer = time.time()

    #check if preferences and reasonings are of the same length
    if (len(preferences) != len(reasonings)):
        print("number of preferences (" + str(len(preferences)) + ") unequal to the number of reasonings (" + 
              str(len(reasonings)) + ")\n")
        exit(0)

    #generate prompt string
    stlList = []
    validData = []
    for key in data[0]:
        validData.append(key)
    
    for preference in range(len(preferences)):
        #LLM
        client = Model(model=model)
        
        prompt = createDataPrompt(preferences[preference], reasonings[preference], ego_name, ado_name, 
                                  data, stride, scenario)
        print("data prompt sent: " + str(time.time() - timer))
        timer = time.time()
        malformedSTL = client.chat(prompt, out=logFile)

        #gen STL
        preOrderPrompt = genPreOrderPrompt(preOrderShots)
        stlFeedback = []
        print("pre-order prompt sent: " + str(time.time() - timer))
        timer = time.time()
        propSTL = client.chat(preOrderPrompt, out=logFile)

        #now re-prompt until syntactically correct STL is generated
        for i in range(promptLimit):
            propSTL = findSTL(propSTL)
            #is STL in bracket form
            if propSTL[0] != False:
                stlFeedback = checkSTL(propSTL, validData)
            #if STL is not in brackets, use feedback in re-prompt
            else:
                stlFeedback = propSTL
            if stlFeedback[0] == False:
                print("stl: " + stlFeedback[1] + "\n")
                stlFeedback = genErrorPrompt(stlFeedback[1])
                print("error prompt "  + str(i) + " sent: " + str(time.time() - timer))
                timer = time.time()
                propSTL = client.chat(stlFeedback, out=logFile)
            else:
                checkPrompt = genSTLCheckPrompt(stlFeedback)
                propSTL = client.chat(checkPrompt, out=logFile)
                if 'yes' in propSTL:
                    break
                else:
                    propSTL[0] = False
                    

        #given that there are generally multiple STL preferences, some failing is acceptable, comment out to mute warning
        if stlFeedback[0] == False:
            print("Warning: On preference [" + str(preference[0]) + " or " + str(preference[1]) + "] stl failed to generate\n")
        else:
            stlList.append(stlFeedback)

    #Return STL
    #client.outputChat(logFile)
    logFile.close()
    return stlList


if __name__ == '__main__':
    #Syntax for preferences and reasonings can be viewed in the NL2PrefSTL function description
    experiment = 'overtake' # name of the experiment
    data_name = experiment + "_trajectories.pkl"
    pref_name = experiment + "_pref.csv"
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
        data[trajectory] = trajDat(pklDat, trajectory)
        
    
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
    
    stl = NL2PrefSTL(preferences=preferences[0:1], reasonings=reasonings[0:1], ego_name=ego_name, ado_name=ado_name, 
               data=data, scenario=scenario, stride=12,
                promptLimit=3, preOrderShots=3)

    for i in stl:
        print(i)
