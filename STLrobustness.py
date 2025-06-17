import numpy as np
import re

class AP:
    #data is the name of the data provided, sign is <, >, or =. constant is a number constant. Represents AP data sign constant ex: speed < 2
    def __init__(self, data, sign, constant):
        self.data = data
        self.sign = sign
        self.constant = constant

    #data is the np array of the required data specified in name
    def calcRobustness(self, data):
        robustness = []
        for point in data:
            r = 0
            match self.sign:
                case '>':
                    r = point - self.constant 
                case '<':
                    r = self.constant - point
                case '=':
                    r = -1 * abs(point - self.constant)
            robustness.append(r)
        return np.array(robustness)
    
    def __str__(self):
        return self.data + " " + self.sign + " " + str(self.constant)

class Predicate:
    #name is the operator and optional inteval is of the form of an array [time1, time2]
    def __init__(self, name, interval=[]):
        self.name = name
        self.interval = interval

    #Given r1 and r2 (if necessary) are set, calculates the robustness value based on the operator
    def calcRobustness(self, r1, r2=[]):
        match self.name:
            case 'not':
                return -1 * r1
            case 'and':
                return np.minimum(r1, r2)
            case 'or':
                return np.maximum(r1, r2)
            case 'imply':
                return np.maximum(-1 * r1, r2)
            #case 'eventually':
            case '_':
                print("not implimented\n")
                exit(0)

    def __str__(self):
        out = self.name
        if self.interval != []:
            out += " [" + str(self.interval[0]) + " " + str(self.interval[1] + "]")
        return out

#ap should be of the class AP and data should be a pytorch tensor
#returns array of rubustness at each time interval

#input is a potential number, returns True if it is a number false otherwise
def isNum(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

#slt is a proposed STL string of the form [AP/operator, AP/operator,..., AP], validAPData is an array of names for data
#returns spliced STL or [False, violating formula element] if STL is an element is syntactically incorrect.
#returns [False, stl] if the net syntax is incorrect
'''
def checkSTL(stl, validAPData):
    #first check validity of individual elements in list and 
    stlList = []
    stl = stl[1:-1]
    stl = stl.split(', ')
    for element in stl:
        p = None
        #check for operator without timeseries information
        if element in ["not", "and", "or", "imply"]:
            p = Predicate(element)

        e = element.split(" ")
        #check for AP
        if e[0] in validAPData:
            #dataName, sign, constant
            if len(e) != 3:
                return [False, element + " is not composed of three portions with a space seperation, desired format is <data sign constant>"]
            if e[1] not in ['<', '>', '='] or not isNum(e[2]):
                return [False, element + " does not have correct formatting, only valid signs are <, >, ="]
            try:
                p = AP(e[0], e[1], float(e[2]))
            except:
                return [False, element + " constant specification cannot be read as a number, desired format is <data sign constant> where constant is a number"]
                
        #check for operator with timeseries information[]
        if e[0] in ["until", "eventually", "always"]:
            #operator <space> [time1 <space> time2]
            if len(e) != 3:
                return [False, element + " is not composed of three portions with a space seperation, desired format is operator [t1 t2]"]
            if len(e[1]) < 2 or len(e[2]) < 2:
                return [False, element + " invalid spacing in temporal specification, desired format is [t1 t2] where t1, t2 are numbers"]
            if e[1][0] != '[' or e[2][-1] != ']' or not isNum(e[1][1:]) or not isNum(e[2][:-1]):
                return [False, element + " no brackets at beginning or end of temporal specification detected, desired format is [t1 t2]"]
            try:
                #infinity is a valid AP
                if e[2] == "infinity":
                    p = Predicate(e[0], float(e[1]), e[2])
                else:
                    p = Predicate(e[0], [float(e[1]), float(e[2])])
            except:
                return [False, element + " temporal specification within brackets cannot be read as a number, desired format is [t1 t2] where t1 and t2 are numbers"]

        if p == None:
            #general failure without specific rationale discovered so preform general diagnostic
            #likely an attempted AP:
            if '<' in element or '>' in element or '=' in element:
                return [False, element + " AP is incorrectly formated, desired format is <data sign constant>"]
            #Another likely attempt an at AP
            for label in validAPData:
                if label in element:
                    return [False, element + " AP is incorrectly formated, desired format is <data sign constant>"]
            #Likely attempt at a temporal operator
            if 'always' in element or 'until' in element or 'eventually' in element:
                return [False, element + " temporal operator is incorrectly formated, desired format is <operator [t1 t2]>"]
            #Likely attempt at a non-temporal operator
            if 'not' in element or 'and' in element or 'or' in element or 'imply' in element:
                return [False, element + " operator is incorrectly formatted, desired format is <operator> without additional information"]
            
            #if not rationale found, note general rejection
            rejection = "STL is syntactically invalid. Remember that all operators within the { } brackets must be verbatim from the following " \
            "'and', 'or', 'not', 'imply', 'always', 'eventually', 'until' and the STL must be in pre-order format. "
            return [False, rejection]
        #correct syntax discoered
        stlList.append(p)

    #check STL is structure is valid
    #note that there should be one root for the STL preorder tree (count == 1)
    count = 0
    for inv in range(len(stlList)):
        #reverse order to build tree
        element = stlList[len(stlList) - 1 - inv]
        if type(element) == AP:
            count += 1
        if type(element) == Predicate:
            if element.name not in ["not", "always", "eventually"]:
                count -= 1
        #no root nodes detected
        if count < 1:
            return [False, "stl is not in valid pre-order, specifically " + element.name + " does not have a sufficient number of propositions"]
    if count != 1:
        return [False, "stl is not in valid pre-order, specifically there are " + str(count) + " propositions that are not connected via a proposition"]

    return stlList
'''

def checkSTL(stl, validAPData):
    #operators
    regStr = "not|imply|and|or|always|eventually|until"
    #data points for APs
    for data in validAPData:
        regStr += "|" + data
    
    div = re.finditer(regStr, stl)

    split = []
    order = []
    begin = -1
    end = 0
    for iter in div:
        if begin != -1:
            end = iter.start()
            split.append(stl[begin:end])
        begin = iter.end()
        
        order.append(stl[iter.start():iter.end()])
    split.append(stl[begin:])
    print(order)
    print(split)

    stlList = []

    for element in range(len(split)):
        p = None
        info = split[element]
        operator = order[element]

        if operator in ['and', 'or', 'imply', 'not']:
            p = Predicate(operator)
        elif operator in ['always', 'until', 'eventually']:
            nums = []
            numParts = re.finditer("[0-9]|\.|-", info)
            #infinity is a valid entry for the second element
            infCheck = re.finditer("infinity", info)
            infIdx = None
            for inf in infCheck:
                if infIdx != None:
                    return [False, "in " + info + " Expected maximum one infinity after operator, recieved multiple"]
                infIdx = inf.start()

            prev = -2
            num = ""
            for iter in numParts:
                #continuation of a previous number:
                if (iter.start() - 1 == prev or num == "") and info[iter.start()] != '-':
                    num += info[iter.start()]
                #new number started
                else:
                    nums.append(float(num))
                    print(num)
                    #insert in infinity if in between numbers
                    if infIdx != None:
                        if infIdx > iter.start():
                            nums.append('infinity')
                    #start new number string
                    num = info[iter.start()]
                prev = iter.start()
            #more then two numbers detected
            if num != "":
                nums.append(float(num))
                if infIdx != None:
                    nums.append('infinity')
                    
            if len(nums) != 2:
                return [False, "in " + info + " Expected two after numbers but detected " + str(len(nums))
                        + " numbers: " + str(nums)]
            p = Predicate(operator, [nums[0], nums[1]])
        else:
            #'operator' is in validAPData so is detecting an AP
            ops = re.finditer("<|>|=", info)
            numParts = re.finditer("[0-9]|\.|-", info)
            #multiple or no >, <, or = detected
            op = None
            for i in ops:
                if op == None:
                    op = i.start()
                else:
                    #multiple of <, >, = detected
                    return [False, "in " + info + " Expected only one of >, <, = but recieved multiple"]
            if op == None:
                return [False, "in " + info + " Expected one of <, >, = but recieved none"]
            
            prev = -2
            numStr = ""
            for part in numParts:
                if part.start() - 1 != prev or info[part.start()] == '-':
                    if prev == -2:
                        prev = part.start()
                        #number detected before the <, >, or = is present
                        if prev < op:
                            return [False, "in " + info + " a number is detected before the " + info[ops[0].start()] + ". This is invalid"]
                    else:
                        #multiple numbers detected
                        return [False, "in " + info + " multiple numbers detected when only one is expected"]
                numStr += info[part.start()]
            
            if len(numStr) == 0:
                return [False, "in " + info + "no constant detected as is necessary in an atompic proposition"]

            p = AP(operator, info[op], float(numStr))
        
        stlList.append(p)
        
    #check STL is structure is valid
    #note that there should be one root for the STL preorder tree (count == 1)
    count = 0
    for inv in range(len(stlList)):
        #reverse order to build tree
        element = stlList[len(stlList) - 1 - inv]
        if type(element) == AP:
            count += 1
        if type(element) == Predicate:
            if element.name not in ["not", "always", "eventually"]:
                count -= 1
        #no root nodes detected
        if count < 1:
            return [False, "stl is not in valid pre-order, specifically " + element.name + " does not have a sufficient number of propositions"]
    if count != 1:
        return [False, "stl is not in valid pre-order, specifically there are " + str(count) + " propositions that are not connected via a proposition"]

    return stlList

#stl is a list of elements in the Predicate or AP class. Data is a dictionary with [name, np array] elements
def robustness(stl, data):
    robustnessStack = []
    #iterate through elements in reverse calculating robustness and adding to stack such that each new operator takes into acount the
    #robustness of the last two proposition robustness calculated (as is natural to pre-order STL formlas)
    for inv in range(len(stl)):
        #reverse order indexing
        element = len(stl) - 1 - inv
        p = stl[element]
        if type(p) == AP:
            robustnessStack.append(p.calcRobustness(data[p.data]))
        if type(p) == Predicate:
            predRobustness = []
            if p.name in ['not', 'always', 'eventually']:
                predRobustness = p.calcRobustness(robustnessStack[-1])
                robustnessStack.pop()
            else:
                predRobustness = p.calcRobustness(robustnessStack[-1], robustnessStack[-2])
                robustnessStack.pop()
                robustnessStack.pop()
                robustnessStack.append(predRobustness)
                
    return robustnessStack[0]

#metric for chosing preference based on robustness vectors
#calculates average tanh robustness (tahnh(robustness)) and with greater values being more accurate
#metric value
def avgTanhRobustness(r):
    reg = np.tanh(r)
    avg = np.average(reg)
    return avg

#returns an an order of trajectories based on metric across all trajectories of size num trajectories
#larger sum across stl formulae metric between trajectories is prefered
#metricVals is an stl trajectories x stl formulae array of metric values 
def largestSum(metricVals):
    sums = np.sum(metricVals, axis=1)
    sort = np.argsort(sums)
    return sort

#returns an array of length trajectories element i is the ranking of the ith trajectory (0-trajectories - 1 with larger numbers preferred)
#data is a dictionary of dictionaries of trajectory, name, array key, key, value elements
#stl is a list of valid pre-order stl formulae in [Predicate/AP] format
#stlMetric is a metric for a robustness on stl (larger better), crossStlMetric is a metric across stl metrics to rank trajectories
#ego_dat, ado_dat are the trajectories of the ego and other perspectives
def metricTable(stlMetric, crossStlMetric, stl, data):
    numTraj = len(data)
    table = np.zeros(shape=(numTraj,  len(stl)))

    for traj in range(numTraj):
        for s in range(len(stl)):
            table[traj][s] = stlMetric(robustness(stl[s], data[traj]))
            print(str(traj) + ": " + str(table[traj][s]))

    trajMetrics = crossStlMetric(table)
    
    return np.argsort(trajMetrics)
    
    