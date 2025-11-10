from extract_info import *


def trajectory_information(data, stride, ego_name, ado_name):
    '''
    Returns string of requested data

    Args:
        stride: sample data every <stride> points
        ego_name: name of ego vehicle
        ado_name: name of other vehicle / object

    Returns:
        str: requested data
    '''
    prompt = ""

    for metric in data:
        description = "" 
        match metric:
            case "ego_trajectory":
                description = f"{ego_name} trajectory"
            case "ado_trajectory":
                description = f"{ado_name} trajectory"
            case "speed":
                #vehicle speed
                description = f"{ego_name} speed"
            case "acceleration":
                #vehicle acceleration
                description = f"{ego_name} acceleration"
            case "jerk":
                #vehicle jerk (derivative of acceleration)
                description = f"{ego_name} jerk"
            case "distance":
                #distance of ego from ado 
                description = f"{ego_name} distance from {ado_name}"
            case "relative":
                #relative speed of ego to ado
                description = f"{ego_name} speed relative to {ado_name}"
            case "longitudinal":
                #longitudinal distance from subject
                description = f"{ego_name} horizontal distance from {ado_name}"
            case "lateral":
                #lateral distance from subject
                description = f"{ego_name} forward distance from {ado_name}"
            case _:
                assert False, f"Invalid data request: \"{metric}\""

        selected_metric = data[metric].squeeze(0).numpy()[::stride]

        prompt += description + ": "
        prompt += str(selected_metric) + "\n\n"

    return prompt


def data_prompt(preference, reasoning, ego_name, ado_name, data, stride,
                     scenario, previous_STL = []):
    '''
    create string of requested data with surrounding context

    Args:
        preference: pair [t1, t2] where t1 is preferred
        reasoning: string rationale for preference
        ego_name: name of ego vehicle
        ado_name: name of other vehicle / object
        data: of form trajectory, metric, tensor key-key-value pairs
        stride: use data every <stride> steps
        scenario: string explenation of trajectory scenario
        previous_STL: optional, list of previously generated STL

    Returns:
        str: requested data with context
    '''

    try:
        assert len(preference) == 2, "preference must be a pair (t1, t2)"
    except Exception:
        assert False, "preference must be indexable array-like"

    stride_interval = stride * 0.1

    prompt = f"The following is two trajectories for a {scenario}" \
             f"taken at {stride_interval} second intervals:\n" \
             "Trajectory information: \n"

    for trajectory in preference:
        prompt += f"Trajectory {trajectory}: \n"
        info = trajectory_information(data[trajectory], stride, ego_name, ado_name)
        prompt += info
    
    prompt += f"Trajectory {preference[0]} is preferred with the stated reason:\n \
              \"{reasoning}\"\n\n" \
              f"Given this information about two trajectories in a {scenario} " \
              "and the user preference about which trajectory is preferred, " \
              "provide a single STL formula that would enforce the users preferences. " \
              "Signal temporal logic (STL) is a form of temporal logic with the \
              following operators: negation, and, until, always[], eventually[], or. " \
              "negation, always, eventually have one following proposition while " \
              "or, and, until have two following propositions. always[], eventually[], " \
              "until[] have a time input such that it should be written always [t1, t2] " \
              "where t1 and t2 are integers are t2 is 'infinity'\n " \
              "In this scenario the available atomic propositions are speed, " \
              "acceleration, jerk, distance, longitudinal, relative. " \
              "jerk is the rate of acceleration. longitudinal is the horizontal distance " \
              f"of the vehicle relative to {ado_name}, lateral is the forward distance " \
              f"of the vehicle realtive to {ado_name} " \
              "relative is the speed of the ego vehicle relative to the overtaken car. " \
              "Distance is the distance of the ego vehicle relative to the overtaken car. " \
              "Use only numerical values in the atomic propositions.\n " \
              "Your goal when creating this STL is to create an STL formula " \
              "that has a higher robustness value for the preferred trajectory " \
              "then for the non-preferred trajectory.\n " \
              "Robustnes is the measure of how formula satisfaction is calculated: \n " \
              "robustness(x < c) = c - x\n " \
              "robustness(x or y) = max(robustness(x), robustness(y))\n " \
              "robustness(x and y) = min(robustness(x), robustness(y))\n " \
              "robustness(negation(x)) = -1 * robustness(x)\n " \
              "robustness(always(x)) = minimum robustness of x over all time\n " \
              "robustness(eventually(x)) = maximum robustness of x over all time\n " \
              "robustness(x until y) = maximum over time t for (min(min over x up to t, y at t))\n\n"

    if len(previous_STL) != 0:
        prompt += "The user has expressed preferences about previous trajectory comparisons. " \
                    "In those comparisons the following STL was developed: \n"
        for i in range(len(previous_STL)):
            prompt += f"STL {i + 1}: {previous_STL[i]}\n"
        prompt += "The generated STL should be complementary and preferably " \
                   "logically consistnat with the previously provided STL \n " \
                   "This can be done such that given the new STL and the previously " \
                   "provided STL combined the user preferences are represented\n"

    prompt += "Here are some examples of alligning STL with the preference of the user:\n " \
               "Given trajectories A and B where trajectory A starts with a speed of 30 " \
               "and and relative speed of 10 and accelerates until it reaches " \
               "a speed of 40 and relative speed of 20, and trajectory B with a " \
               "speed of 20, relative speed of 0 and accelerates until it reaches " \
               "a speed of 40 relative speed of 20.\n " \
               "The user prefers A stating \" I like the speed of A. " \
               "B feels too slow for the most part\".\n " \
               "A possible valid STL statement is {always[0, infinity] speed > 35} " \
               "as although both trajectories violate this, trajectory B violates " \
               "it more and would therefore have a lower minimum robustness  " \
               "selected by the always operator.\n\n " \
               "Another example is given trajectory A that keeps a distance of 10 " \
               "at all times from an overtaking car and trajectory B that keeps a " \
               "distance of 20 but for a short period from t=100s to t=120s of time " \
               "comes with a distance of 5 from the car.\n " \
               "The user prefers B stating \"B keeps a larger distance from " \
               "the car in front\" A valid STL would be  " \
               "{distance > 20 until (eventually [100 140] distance > 20)}\n " \
               "This would allow for the short period of distance to ignored when " \
               "calculating robustness as the car would eventually revert to an " \
               "acceptable distance from the car.\n"

    return prompt


def pre_order_prompt(shots):
    '''generate pre-order prompt with <shots> few shot examples'''

    assert shots <= 3, f"Error: maximum of 3 shots allowed while {shots} requested\n"

    shot1 = "STL: negation(distance < 40) or acceleration < 5\n " \
            "pre-order: {or, negation, distance < 40, acceleration < 5}\n\n"
    shot2 = "STL: acceleration > 0 and speed < 40\n " \
            "pre-order: {and, acceleration > 0, speed < 40}\n\n"
    shot3 = "STL: always[0, infinity] (speed < 50 or acceleration < -5)\n " \
            "pre-order: {always [0 infinity], or, speed < 50, acceleration < -5}\n\n"
    shotList = [shot1, shot2, shot3]

    prompt = "Given the STL generated from the previous prompt, " \
             "re-arrange the STL to be in pre-order format.\n " \
             "The final STL should be within { } brackets {pre-order STL} " \
             "there should only be one set of such brackets and no other such brackets " \
             "should be present in the text.\n " \
             "This format mandates the propositions for an operator go " \
             "after the operator. Examples of this include:\n"
    for i in range(shots):
        prompt += shotList[i]

    return prompt


def error_prompt(advice):
    '''Generate error prompt with advice'''

    prompt = f"The provided pre-order STL is not syntactically correct. " \
              "Please edit your response given the following advice: \n" \
              f"{advice}\n\n" \
              "Remember that all STL must only be composed of data of the following labels: " \
              "'speed', 'acceleration', 'jerk', 'relative', 'distance', " \
              "'longitudinal' and nothing else.\n" \
              "data can be compared only with constants in the form " \
              "<data> <sign> <constant>.\n" \
              "The operators in STL are: 'and', 'or', 'negation', 'always', " \
              "'eventually', 'until' and nothing else.\n" \
              "negation, always, eventually have one following proposition " \
              "while or, and, until have two following propositions. " \
              "always[], eventuallu[], until[] have a time input such that it should " \
              "be written always [t1, t2] where t1 and t2 are integers are t2 is 'infinity'\n" \
              "Write your response in text.\n"

    return prompt

def STL_check_prompt(stl):
    prompt = f"Can the STL provided in your previous prompt be written " \
              "as follows in pre-order form?\n {stl} \n" \
              "If so, respond with one word: yes, otherwise, rephrase"  \
              "the provided STL to better suit the intended meaning\n"

    return prompt


#given pkl file data extract data in the form of data_name, data over all trajectories key value pairs
def trajectory_data(pkl):
    data = {}
    ego = [] 
    ado = []
    maxLength = 0
    for k in range(len(pkl["ego_trajectory"])):
        try:
            ego.append(torch.tensor(np.array(pkl["ego_trajectory"][k]))[:, 1:3])
            ado.append(torch.tensor(np.array(pkl["ado_trajectory"][k]))[:, 1:3])
            if len(pkl["ego_trajectory"][k]) > maxLength:
                maxLength = len(pkl["ego_trajectory"][k])
        except KeyError:
            continue

    #data extracts the metrics from the data and extends them to the max length by repeating the last value
    data["distance"] = get_distance(ego, ado)
    data["relative"] = get_relative_speed(ego, ado)
    data["longitudinal"] = get_longitudinal_speed(ego, ado)
    data["lateral"] = get_lateral_speed(ego, ado)

    #because these data are dependent on eachother and errors occur from extending the data before concatenation
    #they must be extended now
    speedList = get_speed(ego)
    accelerationList = get_acceleration(ego)
    jerkList = get_jerk(ego)
    
    speed = torch.ones(len(pkl["ego_trajectory"]), maxLength)
    acceleration = torch.ones(len(pkl["ego_trajectory"]), maxLength)
    jerk = torch.ones(len(pkl["ego_trajectory"]), maxLength)

    for k in range(len(pkl["ego_trajectory"])):
        speed[k, :] = torch.cat((speedList[k], 
                                 speedList[k][-1]
                                 *torch.ones(size=(maxLength - speedList[k].shape[0],1))),axis=0).squeeze(-1)
        acceleration[k, :] = torch.cat((accelerationList[k], 
                                        accelerationList[k][-1]
                                        *torch.ones(size=((maxLength - accelerationList[k].shape[0]),))))
        jerk[k, :] = torch.cat((jerkList[k], 
                                jerkList[k][-1]
                                *torch.ones(size=((maxLength - jerkList[k].shape[0]),))))
    
    data["speed"] = speed
    data["acceleration"] = acceleration
    data["jerk"] = jerk
    
    return data

#retrives signals for robustness calculations
def get_signals(pkl):
    '''return signals from raw pkl file'''
    data = trajectory_data(pkl)
    for key in data:
        data[key] = data[key].unsqueeze(-1).unsqueeze(-1)

    return data

#LLMs may have difficulty outputting STL alone and requests to do this may relinquish train of thought patterns which benefit accuracy
#Therefore, a search for brackets [] are done to determine where the STL is, if not brackets found, return [False, reasoning]
#input is a prompt intended to have [pre-order stl] format
def find_STL(prompt):
    '''find text surrounded by {} brackets'''
    begin = None
    end = None
    #first find the end of think (relevant string deepseek which starts the actual response message)
    start = prompt.find("</think>")
    if start == -1:
        start = 0
    for index in range(len(prompt) - start):
        c = index + start
        if prompt[c] == '{' and begin is None:
            begin = c
        if begin != None and prompt[c] == '}':
            end = c + 1

    if begin is None:
        error = "A final answer must contain brackets { }around the final stl answer"
        return (False, error)
    if end is None:
        error = "The beginning bracket { of the stl is found but not the end bracket }"
        return (False, error)

    return (True, prompt[begin:end])
