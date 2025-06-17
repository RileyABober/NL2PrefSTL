from extract_info import *

#data is an array of pytorch tensors, stride is the number of time intervals before the next output
#info is the set of labels of data to be included in the promp: ex "[speed, relative_speed, lateral]"
def addTrajInfo(data, stride, ego_name, ado_name):
    prompt = ""
    #match desired data label to actual data. Calculates the data desired from the ego and ado trajectories
    #also sets name format for the prompt
    for d in data:
        description = ""
        match d:
            case "ego_trajectory":
                description = ego_name + " trajectory"
            case "ado_trajectory":
                description = ado_name + " trajectory"
            case "speed":
                #vehicle speed
                description = ego_name + " speed"
            case "acceleration":
                #vehicle acceleration
                description = ego_name + " acceleration"
            case "jerk":
                #vehicle jerk (derivative of acceleration)
                description = ego_name + " jerk"
            case "distance":
                #distance of ego from ado 
                description = ego_name + " distance from " + ado_name
            case "relative_speed":
                #relative speed of ego to ado
                description = ego_name + " speed relative to " + ado_name
            case "longitudinal":
                #longitudinal speed
                description = ego_name + " horizontal speed"
            case _:
                print("Invalid data request: \"" + str(d) + "\"")
                exit(0)
        dataMetric = data[d][::stride]

        prompt += description + ": "
        prompt += str(dataMetric) + "\n\n"

    return prompt

#returns a prompt for an LLM to interperet. one preference and one reasoning provided along with the data and stride that will
#be included for the LLM to reason with. the scenario, ego_name, and ado_name are meant to provide context to the LLM on what
#the data means. Provides a 1 shot example. Returns a string of the prompt information.
def createDataPrompt(preference, reasoning, ego_name, ado_name, data, stride, scenario):
    #first add context of scenario
    strideInterval = str(stride * 0.1) + " second"
    description = ""
    match scenario:
        case "overtake":
            description = "a " + ego_name + " overtaking a slower car scenario"
        case "pedestrian":
            description = "a " + ego_name + "stopping for a pedestrian scenario"
        case _:
            print("Invalid scenario provided: '" + scenario + "'\n")
            exit(0)
    
    prompt = "The following is two trajectories for a " + description + " taken at " + strideInterval + " intervals:\n"

    prompt += "Trajectory information: \n"

    for trajectory in preference:
        prompt += "Trajectory " + str(trajectory) + ": \n"
        info = addTrajInfo(data[trajectory], stride, ego_name, ado_name)
        prompt += info
    
    prompt += "Trajectory " + str(preference[0]) + " is preferred with the stated reason:\n"
    prompt += "\"" + reasoning + "\"\n\n"

    #description of STL
    prompt += "Given this information about two trajectories in a " + scenario + " and the user preference about which trajectory is preferred, "
    prompt += "provide a single STL formula that would enforce the users preferences."
    prompt += "Signal temporal logic (STL) is a form of temporal logic with the following operators: negation, imply, and, equal, until, always[], eventually[], or."
    prompt += "In this scenario the available atomic propositions are speed, acceleration, jerk, distance, longitudinal, relative."
    prompt += "jerk is the rate of acceleration. longitudinal is the horizontal speed of the vehicle, "
    prompt +="relative is the speed of the ego vehicle relative to the overtaken car. Distance is the distance of the ego vehicle relative to the overtaken car."
    prompt += "Use only numerical values in the atomic propositions and do not consider safety constraints as it is unessesary in this instance.\n"

    #one-shot STL
    prompt += "Given trajectory A and trajectory B where A accelerates around 10 and B accelerates around 5 when the distance from the car being "
    prompt += "overtaken is 40. Trajectory B is preferred with the stated reasoning: \n"
    prompt += "\" I don't like how quickly the car accelerates into the adjacent lane when passing the car in A\"\n"
    prompt += "STL: distance < 40 imply acceleration < 10"

    return prompt

#Returns a string requesting that STL should be made pre-order form with <shots> shot implementation with shots < 3
def genPreOrderPrompt(shots):
    #check to see if more than the allowed 3 shots are provided
    if shots > 3:
        print("Error: maximum of 5 shots allowed while " + str(shots) + " requested\n")
        exit(0)

    shot1 = "STL: distance < 40 imply acceleration < 10\n pre-order: {imply, distance < 40, acceleration < 10}\n\n"
    shot2 = "STL: acceleration > 0 and speed < 40\n pre-order: {and, acceleration > 0, speed < 40}\n\n"
    shot3 = "STL: always[0, infinity] (speed < 50 or acceleration < -5)\n pre-order: {always [0 infinity], or, speed < 50, acceleration < -5}\n\n"
    shotList = [shot1, shot2, shot3]

    prompt = "Given the STL generated from the previous prompt, re-arrange the STL to be in pre-order format.\n"
    prompt += "The final STL should be within { } brackets {pre-order STL} there should only be one set of such brackets"
    prompt += "and no other such brackets should be present in the text.\n"
    prompt += "This format mandates the propositions for an operator go after the operator. Examples of this include:\n"
    for i in range(shots):
        prompt += shotList[i]
    
    return prompt

#if syntactically incorrect stl is detected, advice is provided (where the error occured). This is used to create a prompt that
#is intended to, when provided to the LLM, correct the incorrect syntax.
def genErrorPrompt(advice):
    prompt = "The provided pre-order STL is not syntactically correct. " 
    prompt += "Please edit your response given the following offending element and recommendation: \n"
    prompt += advice  + "\n\n"

    prompt += "Remember that all STL must only be composed of data of the following labels: 'speed', 'acceleration', 'jerk', 'relative', 'distance', 'longitudinal' and nothing else.\n"
    prompt += "data can be compared only with constants in the form <data> <sign> <constant>.\n" 
    prompt += "The operators in STL are: 'and', 'or', 'not', 'imply', 'always', 'eventually', 'until' and nothing else.\n"

    return prompt

def genSTLCheckPrompt(stl):
    prompt = "can the stl provided in your previous prompt be written as follows i pre-order forma?\n"
    stlStr = "["
    for element in stl:
        stlStr += str(element)
        stlStr += ", "
    stlStr = stlStr[:-2]
    stlStr += "]"

    prompt += stlStr + "\n"

    prompt += "If so, respond with one word: yes, otherwise, rephrase the provided stl to better suit the intended meaning\n"

    return prompt


def trajDat(pkl, trajectory):
    data = {}
    ego = [torch.tensor(np.array(pkl["ego_trajectory"][trajectory]))[:, 1:3]]
    ado = [torch.tensor(np.array(pkl["ado_trajectory"][trajectory]))[:, 1:3]]
    data["speed"] = get_speed(ego)[0]
    data["acceleration"] = get_acceleration(ego)[0]
    data["jerk"] = get_jerk(ego)[0]
    data["distance"] = get_distance(ego, ado)[0]
    data["relative_speed"] = get_relative_speed(ego, ado)[0]
    data["longitudinal"] = get_longitudinal_speed(ego, ado)[0]
    #return to np
    for key in data:
        data[key] = data[key].numpy()
    
    return data

#LLMs may have difficulty outputting STL alone and requests to do this may relinquish train of thought patterns which benefit accuracy
#Therefore, a search for brackets [] are done to determine where the STL is, if not brackets found, return [False, reasoning]
#input is a prompt intended to have [pre-order stl] format
def findSTL(prompt):
    begin = None
    end = None
    #first find the end of think (relevant string deepseek which starts the actual response message)
    start = prompt.find("</think>")
    print("start: " + str(start) + "\n")
    for index in range(len(prompt) - start):
        c = index + start
        if prompt[c] == '{':
            if begin != None:
                return [False, "More than one { characters detected. Only one set of { } are allowed within the outputted text"]
            begin = c
        if begin != None and prompt[c] == '}':
            if end != None:
                return [False, "More than one } characters detected. Only one set of { } are allowed within the outputted text"]
            end = c + 1
    

    if begin == None:
        return [False, "A final answer must contain square brackets around the final stl answer: format should be {pre-order stl}"]
    if end == None:
        return [False, "The beginning bracket of the stl is found but not the end bracket: format should be {pre-order stl}"]

    return prompt[begin:end]
