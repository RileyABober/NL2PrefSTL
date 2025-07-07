import torch
import re

from WSTL import *

#stl is a proposed STL string of the form [Expression/operator, Expression/operator,..., Expression], 
# validExpressionData is an array of names for data
#returns tuple (WSTL formula, signal structure) or [False, violating formula element] if STL is an element is syntactically incorrect.
#returns [False, stl] if the net syntax is incorrect
#the horizon is the replacement for "infinity" so that actual calculations can occur and weights can be assigned
def checkSTL(stl, validExpressionData, horizon=500):
    #operators
    regStr = "negation|and|or|always|eventually|until"
    #data points for Expressions
    for data in validExpressionData:
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

        if operator in ['and', 'or', 'negation']:
            p = (operator,)
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
                    #print(num)
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
            p = (operator, (nums[0], nums[1]))
        else:
            #'operator' is in validExpressionData so is detecting an Expression
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

            p = (operator, info[op], float(numStr))

        
        stlList.append(p)
        
    #check STL is structure is valid
    #note that there should be one root for the STL preorder tree (count == 1)
    count = 0
    for inv in range(len(stlList)):
        #reverse order to build tree
        element = stlList[len(stlList) - 1 - inv][0]
        if element in validExpressionData:
            count += 1
        else:
            if element not in ["negation", "always", "eventually"]:
                count -= 1
        #no root nodes detected
        if count < 1:
            return [False, "stl is not in valid pre-order, specifically " + element + " does not have a sufficient number of propositions"]
    if count != 1:
        return [False, "stl is not in valid pre-order, specifically there are " + str(count) + " propositions that are not connected via a proposition"]

    #manufacture STL and structure for data in form of a tuple of tuples of data names in accordance with the format in WSTL.py
    #generic signal to initialize lengths
    signalInit = torch.ones(horizon)
    signalInit = signalInit.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)    

    #traverse the preorder with a stack allows for reading of the tree
    formulaStack = []
    signalStack = []
    for inv in range(len(stlList)):
        #reverse order to build tree
        element = stlList[len(stlList) - 1 - inv]
        numArgs = 0
        if element[0] in ["negation", "always", "eventually"]:
            numArgs = 1
        if element[0] in ["and", "or", "until"]:
            numArgs = 2
        inputArgs = []
        signalArgs = []
        for index in range(numArgs):
            inputArgs.append(formulaStack[-1])
            signalArgs.append(signalStack[-1])
            formulaStack.pop()
            signalStack.pop()
        if numArgs == 1:
            signalStack.append(signalArgs[0])
        elif numArgs == 2:
            signalStack.append(tuple(signalArgs))

        match element[0]:
            case 'negation':
                formulaStack.append(Negation(inputArgs[0]))
            case 'and':
                formulaStack.append(And(inputArgs[0], inputArgs[1]))
            case 'or':
                formulaStack.append(Or(inputArgs[0], inputArgs[1]))
            case 'always':
                formulaStack.append(Always(inputArgs[0], element[1]))
            case 'eventually':
                formulaStack.append(Eventually(inputArgs[0], element[1]))
            case 'until':
                formulaStack.append(Until(inputArgs[0], inputArgs[1], element[1]))
            case _: #Expression
                expression = Expression(element[0], signalInit)
                match element[1]:
                    case '<': formulaStack.append(expression <= element[2])
                    case '>': formulaStack.append(expression >= element[2])
                    case '=': formulaStack.append(expression == element[2])
                    case _: 
                        assert False, "Implementation error, stlCheck detected invalid operator " + element[0]
                signalStack.append(element[0])
                
    #The stack should be empty except for one element (as is determined in validating that the STL is valid)
    #returned arguments are the WSTL formula and the signals in the form of provided names from validExpressionData
    return (formulaStack[0], signalStack[0])

#input tuple of signal names in WSTL signal format for the WSTL input.
#data is in the form of name, torch tensor shape(trajectories, length, 1, 1) key value pairs
def convertSignalTuple(signalNames, data):    
    #level order traversal
    stack = []
    queue = [signalNames]
    while len(queue) != 0:
        levelCount = len(queue)
        for node in range(levelCount):
            if type(queue[0]) == tuple: #further branching
                count = 0
                for branch in queue[0]:
                    count += 1
                    queue.append(branch)
                stack.append(count)
            else:
                stack.append(queue[0])
            queue.pop(0)
    
    #now stack is a list of strings and numbers with strings denoting the data name
    #and numbers representing the tuple size of a node
    heldQueue = []
    for inv in range(len(stack) - 1, -1, -1):
        if type(stack[inv]) == int:
            #fill sub aray and reverse
            subArray = []
            for held in range(stack[inv]):
                subArray.append(heldQueue[0])
                heldQueue.pop(0)
            subArray.reverse()

            heldQueue.append(tuple(subArray))
        else:
            #add the tensor for the data represented by the name
            heldQueue.append(data[stack[inv]])
    
    return heldQueue[0]


            
