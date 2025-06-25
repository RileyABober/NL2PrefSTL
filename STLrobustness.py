import numpy as np
import re
import matplotlib.pyplot as plt

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
            case 'always':
                t1, t2 = self.intervalCheck(len(r1))
                if t1 == -1:
                    #interval range has an invalid start
                    return np.zeros(len(r1))
                minRobustness = np.amin(r1[t1:t2])
                return np.full(len(r1), minRobustness)
            case 'eventually':
                t1, t2 = self.intervalCheck(len(r1))
                if t1 == -1:
                    return np.zeros(len(r1))
                maxRobustness = np.amax(r1[t1:t2])
                return np.full(len(r1), maxRobustness)
            case 'until':
                t1, t2 = self.intervalCheck(len(r1))
                if t1 == -1:
                    return np.zeros(len(r1))
                minR1 = np.zeros(len(r1))
                minAtI = r1[0]
                for i in range(t2):
                    if minAtI > r1[i]:
                        minAtI = r1[i]
                    minR1[i] = minAtI
                
                transComp = np.minimum(minR1, r2)
                maxInInterval = np.amax(transComp[t1:t2])
                return np.full(len(r1), maxInInterval)
            case '_':
                print(self.name + " is not a valid name in STL")
                exit(0)

    def intervalCheck(self, signalLength):
        if self.interval[0] > signalLength:
            #interval range has an invalid start
            return [-1, -1]
        if self.interval[1] == 'infinity':
            #set interval to end at signal length
            return [int(self.interval[0]), signalLength]
            #set interval to end at signal length
        if self.interval[1] > signalLength:
            return [int(self.interval[0]), signalLength]
        return [int(self.interval[0]), int(self.interval[1]) + 1]

    def __str__(self):
        out = self.name
        if self.interval != []:
            out += " [" + str(self.interval[0]) + " " + str(self.interval[1] + "]")
        return out
    
class WSTL:
    #list of APs and Predicates in a valid order (view checkSTL function)
    def __init__(self, stlList):
        self.list = stlList

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
                robustnessStack.append(p.calcRobustness(data[p.data]))
            if type(p) == Predicate:
                predRobustness = []
                if p.name in ['not', 'always', 'eventually']:
                    predRobustness = p.calcRobustness(robustnessStack[-1])
                    robustnessStack.pop()
                    robustnessStack.append(predRobustness)
                else:
                    predRobustness = p.calcRobustness(robustnessStack[-1], robustnessStack[-2])
                    robustnessStack.pop()
                    robustnessStack.pop()
                    robustnessStack.append(predRobustness)
                    
        return np.amin(robustnessStack[0])    

#stl is a proposed STL string of the form [AP/operator, AP/operator,..., AP], validAPData is an array of names for data
#returns spliced STL or [False, violating formula element] if STL is an element is syntactically incorrect.
#returns [False, stl] if the net syntax is incorrect
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
                return [False, "in " + info + " Expected two numbers but detected " + str(len(nums))
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

    return WSTL(stlList)

