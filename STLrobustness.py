import numpy as np
import re
from copy import deepcopy
from math import inf
import matplotlib.pyplot as plt

class AP():
    #data is the name of the data provided, sign is <, >, or =. constant is a number constant. Represents AP data sign constant ex: speed < 2
    def __init__(self, data, sign, constant):
        self.data = data
        self.sign = sign
        self.constant = constant

    #data is the np array of the required data specified in name
    def robustness(self, data):
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

class Not():
    def robustness(self, r):
        return -1 * r

    def __str__(self):
        return "not"
    
class And():
    def __init__(self, weights=np.full(2, 0.5)):
        self.weights = weights

    def robustness(self, r1, r2):
        return np.minimum(r1*self.weights[0], r2*self.weights[1])
    
    def set_weights(self, weights):
        self.weights = weights

    def randomize_weights(self):
        self.weights = np.random.rand(len(self.weights))

    def retrieve_weights(self): return self.weights 
    
    def __str__(self):
        return "and"
    
class Or():
    def __init__(self, weights=np.full(2, 0.5)):
        self.weights = weights

    def robustness(self, r1, r2):
        return np.maximum(r1*self.weights[0], r2*self.weights[1])
    
    def randomize_weights(self):
        self.weights = np.random.rand(len(self.weights))

    def retrieve_weights(self): return self.weights 
    
    def __str__(self):
        return "or"
    

class Imply():
    def __init__(self, weights=np.full(2, 0.5)):
        self.weights = weights

    def robustness(self, r1, r2):
        return np.maximum(-1*r1*self.weights[0], r2*self.weights[1])
    
    def set_weights(self, weights):
        self.weights = weights

    def randomize_weights(self):
        self.weights = np.random.rand(len(self.weights))

    def retrieve_weights(self): return self.weights 
    
    def __str__(self):
        return "imply"
    
class Always():
    def __init__(self, interval, weights=None):
        self.interval = interval
        if weights == None:
            weightLen = interval[1] - interval[0] + 1
            self.weights = np.full(weightLen, 0.5)
        else:
            self.weights = weights

    def robustness(self, r):
        t1 = self.interval[0]
        t2 = self.interval[1] + 1
        if len(r) < t1:
            return np.zeros(len(r))
        if len(r) < t2:
            t2 = len(r)
        minRobustness = np.amax(r[t1:t2] * self.weights[t1:t2])
        return np.full(len(r), minRobustness)
    
    def set_weights(self, weights):
        self.weights = weights

    def randomize_weights(self):
        self.weights = np.random.rand(len(self.weights))

    def retrieve_weights(self): return self.weights 
    
    def __str__(self):
        return "always [" + str(self.interval[0]) + " " + str(self.interval[1]) + "]"
    
class Eventually:
    def __init__(self, interval, weights=None):
        self.interval = interval
        if weights == None:
            weightLen = interval[1] - interval[0] + 1
            self.weights = np.full(weightLen, 0.5)
        else:
            self.weights = weights
    
    def robustness(self, r):
        t1 = self.interval[0]
        t2 = self.interval[1] + 1
        if len(r) < t1:
            return np.zeros(len(r))
        if len(r) < t2:
            t2 = len(r)
        maxRobustness = np.amax(r[t1:t2] * self.weights[t1:t2])
        return np.full(len(r), maxRobustness)
    
    def set_weights(self, weights):
        self.weights = weights

    def randomize_weights(self):
        self.weights = np.random.rand(len(self.weights))

    def retrieve_weights(self): return self.weights 
    
    def __str__(self):
        return "eventually [" + str(self.interval[0]) + " " + str(self.interval[1]) + "]"
    
class Until:
    def __init__(self, interval, weights=None):
        self.interval = interval
        if weights == None:
            weightLen = interval[1] - interval[0] + 1
            self.weights = np.full((2, weightLen), 0.5)
        else:
            self.weights = weights

    def robustness(self, r1, r2):
        t1 = self.interval[0]
        t2 = self.interval[1] + 1
        if len(r1) < t1:
            return np.zeros(len(r1))
        if len(r1) < t2:
            t2 = len(r1)
        minR1 = np.zeros(len(r1))
        minAtI = r1[0]
        for i in range(t2):
            if minAtI > r1[i]:
                minAtI = r1[i]
            minR1[i] = minAtI
                    
        transComp = np.minimum(minR1 * self.weights[0][t1:t2], r2 * self.weights[0][t1:t2])
        maxInInterval = np.amax(transComp[t1:t2])
        return np.full(len(r1), maxInInterval)
    
    def set_weights(self, weights):
        self.weights = weights

    def randomize_weights(self):
        self.weights = np.random.rand(len(self.weights))

    def retrieve_weights(self): return self.weights 
    
    def __str__(self):
        return "until [" + str(self.interval[0]) + " " + str(self.interval[1]) + "]"
    
class WSTL:
    #list of APs and Predicates in a valid order (view checkSTL function)
    def __init__(self, stlList):
        self.list = stlList

    def copy(self):
        return WSTL(deepcopy(self.list))
    
    #combines two stl statements
    def add(self, wstl):
        a = And()
        synth = [a]
        for i in wstl.list:
            synth.append(i)
        for i in self.list:
            synth.append(i)
        self.list = synth

    
    def randomize_weights(self):
        for element in self.list:
            if type(element) not in [Not, AP]:
                element.randomize_weights()

    #appends via and operation another STL formula

    #stl is a list of elements in the Predicate or AP class. Data is a dictionary with [name, np array] elements
    def robustness(self, data):
        robustnessStack = []
        #iterate through elements in reverse calculating robustness and adding to stack such that each new operator takes into acount the
        #robustness of the last two proposition robustness calculated (as is natural to pre-order STL formlas)
        for inv in range(len(self.list)):
            #reverse order indexing
            element = len(self.list) - 1 - inv
            p = self.list[element]

            if type(p) == AP:
                robustnessStack.append(p.robustness(data[p.data]))
            else:
                predRobustness = []
                if type(p) in [Not, Eventually, Always]:
                    predRobustness = p.robustness(robustnessStack[-1])
                    robustnessStack.pop()
                    robustnessStack.append(predRobustness)
                else:
                    predRobustness = p.robustness(robustnessStack[-1], robustnessStack[-2])
                    robustnessStack.pop()
                    robustnessStack.pop()
                    robustnessStack.append(predRobustness)
                    
        return np.amin(robustnessStack[0])

def strToPredicate(name, interval=[]):
    match name:
        case "not":
            return Not()
        case "imply":
            return Imply()
        case "and":
            return And()
        case "or":
            return Or()
        case "always":
            return Always(interval)
        case "eventually":
            return Eventually(interval)
        case "until":
            return Until(interval)
    print("Error: Invalid stl name detected\n")
    exit(0)

#stl is a proposed STL string of the form [AP/operator, AP/operator,..., AP], validAPData is an array of names for data
#returns spliced STL or [False, violating formula element] if STL is an element is syntactically incorrect.
#returns [False, stl] if the net syntax is incorrect
#the horizon is the replacement for "infinity" so that actual calculations can occur and weights can be assigned
def checkSTL(stl, validAPData, horizon=500):
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

    stlList = []

    for element in range(len(split)):
        p = None
        info = split[element]
        operator = order[element]

        if operator in ['and', 'or', 'imply', 'not']:
            p = strToPredicate(operator)
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
                    nums.append(int(float(num)))
                    print(num)
                    #insert in infinity if in between numbers
                    if infIdx != None:
                        if infIdx > iter.start():
                            nums.append(horizon)
                    #start new number string
                    num = info[iter.start()]
                prev = iter.start()
            #more then two numbers detected
            if num != "":
                nums.append(int(float(num)))
                if infIdx != None:
                    nums.append(horizon)
                    
            if len(nums) != 2:
                return [False, "in " + info + " Expected two numbers but detected " + str(len(nums))
                        + " numbers: " + str(nums)]
            p = strToPredicate(operator, [nums[0], nums[1]])
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
        else:
            if type(element) not in [Not, Always, Eventually]:
                count -= 1
        #no root nodes detected
        if count < 1:
            return [False, "stl is not in valid pre-order, specifically " + element.name + " does not have a sufficient number of propositions"]
    if count != 1:
        return [False, "stl is not in valid pre-order, specifically there are " + str(count) + " propositions that are not connected via a proposition"]

    return WSTL(stlList)

#returns WSTL formula corresponding with the STL provided in wstl with weights optimized to minimize loss.
#iterations is the number of random assignments checked
#loss must be of the form loss(robustness of preferred, robustness of non-preferred) and returns a float value
#data is of the form {trajectory i: data i, ...} where data i is a dictionary {metric: np vector,...}
#preferences are of the form [[preferred data key, unfavored data key],...]
def random_sampling(wstl: WSTL, data, preferences, loss, iterations):
    optLoss = inf
    optWSTL = None
    currWSTL = wstl
    tempLoss = np.zeros(iterations)
    tempOpt = np.zeros(iterations)
    for itr in range(iterations):
        ###indicator, temporary ------
        #if itr % 1000 == 0:
            #print("iteration: " + str(itr) + " loss: " + str(optLoss) + "\n")
        #calculate robustness
        lossSum = 0
        for pref in preferences:
            favored = pref[0]
            disfavored = pref[1]
            rPref = wstl.robustness(data[favored])
            rNonPref = wstl.robustness(data[disfavored])
            lossSum += loss(rPref, rNonPref)

        #determine if a new optimum is found
        if lossSum < optLoss:
            optWSTL = currWSTL.copy()
            optLoss = lossSum
        
        #randomize current wstl while leaving optimum unchanged
        currWSTL.randomize_weights()
        tempLoss[itr] = lossSum
        tempOpt[itr] = optLoss
    plt.scatter(range(iterations), tempLoss)
    plt.show()
    plt.scatter(range(iterations), tempOpt)
    plt.show()
    print(optLoss)
    return optWSTL

#calculates the difference r2 - r1
def diffLoss(r1, r2):
    return r2 - r1

def hingeLoss(r1, r2):
    return max(r2 - r1 + 1, 0)
        
def countLoss(r1, r2):
    if r1 > r2:
        return 0
    return 1