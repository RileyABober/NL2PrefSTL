import torch
import re
import random
import numpy as np

import WSTL

class Signal_WSTL():
    '''
    Saves signal structure of WSTL for easier computation
    
    Attributes:
        wstl: WSTL.WSTL_Formula
        signal_names: tuple structure of signals with string names instead of
                      torch tensor signals

    Methods:
        robustness: calculates robustness of wstl
        set_weights: sets weights of wstl
        convert_signal_tuple: creates valid signals tuples from raw signal data
    '''
    def __init__(self, wstl: WSTL.WSTL_Formula, signal_names: tuple):
        self.signal_names = signal_names
        self.wstl = wstl

    def robustness(self, data, scale=-1, t=0) -> torch.Tensor:
        """
        Returns WSTL weighted robustness value for given input signals
        and for all weight valuation samples at t=0, by default.
        Note that robustness is computed per each time instant.

        Args:
            data (dictionary or dataframe): name, Input signals key value pairs.
            scale (int): Scaling factor for robustness computation.
            t (int): Time instance for which to compute the robustness.

        Returns:
            torch.Tensor: WSTL weighted robustness values.
        """

        converted_signals = self.convert_signal_tuple(data)
        return self.wstl.robustness(converted_signals, scale=scale, t=t)
    
    def set_weights(self, data, samples, seed=None):
        '''
        set weights of self.wstl formula

        Args:
            data (dictionary or dataframe): name, Input signals key value pairs.
            samples: number of weight evaluations
            seed: seed for reproducability
        '''
        if seed is None:
            seed = random.random()

        converted_signals = self.convert_signal_tuple(data)
        self.wstl.set_weights(converted_signals, w_range=[0.01, 1.01],
                                     no_samples=samples, random=True, seed=seed)

    def convert_signal_tuple(self, data):    
        '''
        Given tuple of signal_names returns tuple of same structure with signals
        from data in place of names

        Args:
            data: dictionary of (name, signal) key value pairs 

        Returns:
            signal tuple
        '''

        stack = []
        queue = [self.signal_names]
        while len(queue) != 0:
            level_count = len(queue)
            for node in range(level_count):
                if isinstance(queue[0], tuple): 
                    count = 0
                    for branch in queue[0]:
                        count += 1
                        queue.append(branch)
                    stack.append(count)
                else:
                    stack.append(queue[0])
                queue.pop(0)
        
        held_queue = []
        for inv in range(len(stack) - 1, -1, -1):
            if isinstance(stack[inv], int):
                sub_array = []
                for held in range(stack[inv]):
                    sub_array.append(held_queue[0])
                    held_queue.pop(0)
                sub_array.reverse()

                held_queue.append(tuple(sub_array))
            else:
                held_queue.append(data[stack[inv]].unsqueeze(-1).unsqueeze(-1))
        #Note the second [0] index is a hot fix, idk why it works
        return held_queue[0]
    
def compute_accuracy(wstl, data, weight_evaluations, train, test, metric, batches=1, seed=None):
        '''
        Gives test, train, and net accuracy based on metric evaluation
          over weight_evaluations weights

        Args:
            data: data organized in form of {name, [trajectory, signal]} key value pairs
            weight_evaluations: number of weight evaluations performed
            train: list of numerical pairs where each element is a trajectory in data
            test: list of numerical pairs where each element is a trajectory in data
            metric: in class Metric, performs ranking of robustness across trajectories
            batches: splits robustness tests into weight_evaluations / batches 
            seed: randomness for reproducability set to None automatically

        Returns:
            (accuracy, train_accuracy, test_accuracy)
        '''
        for key in data:
            assert isinstance(data[key], torch.Tensor), "values in dict must be torch.tensor"
        assert isinstance(weight_evaluations, int), "weight_evaluations must be an int"
        assert isinstance(metric, Metric), "metric must be of the class Metric"
        assert weight_evaluations >= 1, "weight_evaluations must be >= 1"
        assert isinstance(batches, int), "batches must be an integer quantity"
        assert batches >= 1, "batches must be >= 1"

        robustness_values = None
        for batch in range(batches):
            evals = best_slice(wstl, data, int(weight_evaluations / batches), train, metric, seed)
            if robustness_values is None:
                robustness_values = evals
            else:
                robustness_values = torch.cat((evals, robustness_values), dim=2)

        best_eval = metric(evals, train)

        accuracy = [0, 0, 0] #train, test, net
        for pair in train:
            if best_eval[pair[0]] > best_eval[pair[1]]:
                accuracy[0] += 1
                accuracy[2] += 1

        test_accuracy = None
        test_length = 0
        if test is not None:
            for pair in test:
                if best_eval[pair[0]] > best_eval[pair[1]]:
                    accuracy[1] += 1
                    accuracy[2] += 1
            test_length = len(test)
            test_accuracy = accuracy[1] / test_length

        return (accuracy[0] / len(train), test_accuracy, 
                accuracy[2] / (len(train) + test_length))

def best_slice(wstl: Signal_WSTL, data, weight_evaluations, train, metric, seed=None):
    wstl.set_weights(data, weight_evaluations, seed)
    robustness = wstl.robustness(data)
    best_slice =  metric(robustness, train)
    return best_slice.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class Metric:
    '''Callable classes returning an optimal slice of robustness data'''
    def __call__(self, robustness: torch.Tensor, preferences) -> torch.Tensor:
        '''
        Find robustness values over trajectories that maximize the metric 

        Args:
            robustness: torch.tensor (trajectories, 1, weight evaluations, 1)
            preferences: tuple or list of trajectory pairs of format
                         (preferred, not preferred): (int, int)

        Returns: 
            torch.tensor: shape=(trajectories,) of robustness values 
        '''
        assert isinstance(robustness, torch.Tensor), \
            "robustness must be a torch tensor"

        assert robustness.dim() == 4, \
            "robustness must be shape (trajectories, 1, weight evaluations, 1)"

        assert robustness.shape[1] == 1 and robustness.shape[3] == 1, \
            "robustness must be shape (trajectories, 1, weight evaluations, 1)"

        return self.calc_metric(robustness, preferences)

    def calc_metric(self, robustness, preferences) -> AssertionError:
        assert False, "calc_metric is not defined for base class"

class Max_Difference(Metric):
    def __init__(self):
        super().__init__()
    
    def calc_metric(self, robustness: torch.Tensor, preferences) -> torch.Tensor:
        '''
        Maximize avg difference between preferred and non-preferred trajectories

        Args:
            robustness: torch tensor (trajectories, 1, weight evaluations, 1)
            preferences: tuple or list of pairs of format
                         (preferred, not preferred): (int, int)

        Returns: 
            torch.tensor shape=(trajectories,) of robustness values 
        '''

        difference_sum = torch.zeros(size=(robustness.shape[2],))
        for preference in preferences:
            preferred = robustness[preference[0], 0, :, 0]
            non_preferred = robustness[preference[1], 0, :, 0]
            difference_sum = difference_sum + (preferred - non_preferred)
        return robustness[:, :, difference_sum.argmax(), :].squeeze()
    
class Max_Correct(Metric):
    def __init__(self):
        super().__init__()

    def calc_metric(self, robustness: torch.Tensor, preferences) -> torch.Tensor:
        '''
        Maximize number of correctly classified preference pairs

        Args:
            robustness: torch tensor (trajectories, 1, weight evaluations, 1)
            preferences: tuple or list of pairs of format
                         (preferred, not preferred): (int, int)

        Returns: 
            torch.tensor shape=(trajectories,) of robustness values 
        '''

        correct_count = torch.zeros(size=(robustness.shape[2],))
        for preference in preferences:
            preferred = robustness[preference[0], 0, :, 0]
            non_preferred = robustness[preference[1], 0, :, 0]
            correct_count = correct_count + (torch.ge(preferred, non_preferred))
            

        return robustness[:, :, correct_count.argmax(), :].squeeze()

def metric_accuracy(metric_evaluation, preferences) -> float:
    '''Calculates accuracy based metric_evaluation and preferences'''

    count = float(len(preferences))
    correct_count = 0
    for preferred, non_preferred in preferences:
        if metric_evaluation[preferred] > metric_evaluation[non_preferred]:
            correct_count += 1

    return correct_count / count

def process_temporal_operator(operator, info, horizon):
    '''
    Checks if the info and operator are compatible and if so returns formatted
    information about the operator

    Args:
        operator: string: always, until, eventually
        info: string potentially containing temporal information

    Returns: 
        if processed correctly: (True, (operator, (t1, t2)))
        if unable to process: (False, explenation string)
    '''

    assert operator in ["always", "until", "eventually"], \
           "inputed operator must be one of ['always', 'until', 'eventually]"

    nums = []
    num_parts = re.finditer("[0-9]|\.", info)
    #infinity is a valid entry for the second element
    inf_check = re.finditer("infinity", info)
    inf_index = None
    for inf in inf_check:
        if inf_index != None:
            error = f"in {info} Expected maximum one infinity after operator, recieved multiple"
            return (False, error)
        inf_index = inf.start()

    prev = None
    num = ""
    for iter in num_parts:
        if iter.start() - 1 == prev or num == "":
            num += info[iter.start()]

        else:
            try:
                nums.append(int(float(num)))
            except Exception:
                error = f"in {info} numbers are formated incorrectly"
                return (False, error)
            
            if inf_index != None:
                if inf_index > iter.start():
                    nums.append(horizon)
            #start new number string
            num = info[iter.start()]
        prev = iter.start()

    if num != "":
        try:
            nums.append(int(float(num)))
        except:
            error = f"in {info} numbers are formatted incorrectly"
            return (False, error)
        if inf_index != None:
            nums.append(horizon)
                    
    if len(nums) != 2: 
        if len(nums) == 0: 
            nums = [0, horizon]
        elif len(nums) == 1: 
            nums = [0, nums[0]]
        else:
            error = f"in {operator} Expected two numbers but detected {len(nums)}"
            return (False, error)
        
    return (True, (operator, (nums[0], nums[1])))

def process_expression(name, info):
    '''
    Extracts information if the info is compatable with an Expression format

    Args:
        name: string, name of signal
        info: string, potentially containing expression information

    Returns:
        if processed correctly: (True, ())
        if unable to process: (False, explenation string)
    '''

    operators = re.finditer("<|>", info)
    num_parts = re.finditer("[0-9]|\.|-", info)

    operator = None
    for i in operators:
        if operator is None:
            operator = i.start()
        else:
            error = f"in {info} Expected only one of >, < but recieved multiple"
            return (False, error)
    if operator is None:
        error = f"in {info} Expected one of <, > but recieved none"
        return (False, error)
            
    prev = -2
    numStr = ""
    for part in num_parts:
        if part.start() - 1 != prev or (info[part.start()] == '-' and numStr != ""):
            if prev != -2:
                error = f"in {info} multiple numbers detected when only one is expected"
                return (False, error)
        prev = part.start()
        numStr += info[part.start()]
    if len(numStr) == 0:
        error = f"in {info} no constant detected in expression"
        return (False, error)
    
    try:
        return (True, (name, info[operator], float(numStr)))
    except:
        error = f"in {info} constant is incorretly formated"
        return [False, error]

def check_stl_structure(stl):
    '''
    Determines if stl elements can be converted into a valid structure

    Args:
        stl: a list of operators of format [(name, info...),...]
    
    Returns: 
        if stl is correctly structured: (True, None)
        else:                           (False, description)

    '''
    count = 0
    for inv in range(len(stl)):
        element = stl[len(stl) - 1 - inv][0]
        if element in ["and", "or", "until"]:
            count -= 1
        elif element not in ["negation", "always", "eventually"]:
            count += 1
        if count < 1:
            error = f"stl is not in valid pre-order as {element}" \
                     "does not have a sufficient number of following expressions"
            return (False, error)
    if count != 1:
        error = f"stl is not in valid pre-order as not all operators are connected"
        return (False, error)
    
    return (True, None)

def structure_stl(stl, horizon) -> tuple[WSTL.WSTL_Formula, tuple]:
    '''
    Converts list of stl operators with information into a single WSTL formula
    stl must be able to pass check_stl_structure(stl) to function

    Args:
        stl: list  of expressions: (name, operator, constant)
                      temporal operators: (name, (t1, t2))
                      logical operators: (name)
    
    Returns:
        Signal_WSTL
    '''

    generic_signal = torch.ones(horizon)
    generic_signal = generic_signal.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)    

    formula_stack = []
    signal_stack = []
    for inv in range(len(stl)):
        #reverse order to build tree
        element = stl[len(stl) - 1 - inv]
        num_args = 0
        if element[0] in ["negation", "always", "eventually"]:
            num_args = 1
        if element[0] in ["and", "or", "until"]:
            num_args = 2
        inputArgs = []
        signalArgs = []
        for index in range(num_args):
            inputArgs.append(formula_stack[-1])
            signalArgs.append(signal_stack[-1])
            formula_stack.pop()
            signal_stack.pop()
        if num_args == 1:
            signal_stack.append(signalArgs[0])
        elif num_args == 2:
            signal_stack.append(tuple(signalArgs))

        match element[0]:
            case 'negation':
                formula_stack.append(WSTL.Negation(inputArgs[0]))
            case 'and':
                formula_stack.append(WSTL.And(inputArgs[0], inputArgs[1]))
            case 'or':
                formula_stack.append(WSTL.Or(inputArgs[0], inputArgs[1]))
            case 'always':
                formula_stack.append(WSTL.Always(inputArgs[0], element[1]))
            case 'eventually':
                formula_stack.append(WSTL.Eventually(inputArgs[0], element[1]))
            case 'until':
                formula_stack.append(WSTL.Until(inputArgs[0], inputArgs[1], element[1]))
            case _: #Expression
                expression = WSTL.Expression(element[0], generic_signal)
                match element[1]:
                    case '<': formula_stack.append(expression <= element[2])
                    case '>': formula_stack.append(expression >= element[2])
                    case _: 
                        assert False, \
                            f"in structure_stl, detected invalid operator {element[0]}"
                signal_stack.append(element[0])

    return Signal_WSTL(formula_stack[0], signal_stack[0])
    

def check_stl(stl, valid_signals, horizon=500):
    '''
    Given a an stl string searches for a valid STL formula. If found returns
    tuple of form (STL, Tuple of signal names). If none found found returns
    (None, string explenation for why STL is invalid)

    Args:
        stl: STL string to be validated
        valid_signals: list of strings that are valid signal names
        horizon: length of temporal operators that have the infinity argument

    Returns:
        
        if stl is invalid: (None, string explanation)
        else:              (Signal_WSTL, None)
    '''

    reg_string = "negation|and|or|always|eventually|until"
    for data in valid_signals:
        reg_string += "|" + data
    
    key_words = re.finditer(reg_string, stl)

    split = []
    order = []
    begin = -1
    end = 0
    for iter in key_words:
        if begin != -1:
            end = iter.start()
            split.append(stl[begin:end])
        begin = iter.end()
        order.append(stl[iter.start():iter.end()])
    split.append(stl[begin:])

    stl_list = []
    for element in range(len(split)):
        p = None
        info = split[element]
        operator = order[element]

        if operator in ['and', 'or', 'negation']:
            p = (operator,)
        elif operator in ['always', 'until', 'eventually']:
            valid, description = process_temporal_operator(operator, info, horizon)
            if not valid:
                return (None, description)
            p = description
        else:
            valid, description = process_expression(operator, info)
            if not valid:
                return (None, description)
            p = description

        stl_list.append(p)
        
    valid, description = check_stl_structure(stl_list)
    if not valid:
        return (None, description)

    return (structure_stl(stl_list, horizon), None)


def link_STL(stl_list) -> Signal_WSTL:
    '''
    links stl statements and signal tuples

    Args:
        stl_list: list of Signal_WSTL formulae

    Returns:
        Signal_WSTL: connected WSTL
    '''

    if len(stl_list) == 0:
        return (stl_list)
    if len(stl_list) == 1:
        return stl_list[0]
    linked_stl = stl_list[0].wstl
    linked_signals = stl_list[0].signal_names

    for stl in stl_list[1:]:
        linked_stl = WSTL.And(stl.wstl, linked_stl)
        linked_signals = (stl.signal_names, linked_signals)
    
    return Signal_WSTL(linked_stl , linked_signals)

def accuracy_over_splits(wstl_list, data, splits, weight_evaluations, preferences, 
                         metric, batches=1, seed=None):
    '''
    Gives test, train, and net accuracy based on metric evaluation
        over weight_evaluations weights

    Args:
        wstl_list: list of valid wstl formulae
        data: data organized in form of {name, [trajectory, signal]} key value pairs
        splits: list of integers where integer <= len(preferences)
        weight_evaluations: number of weight evaluations performed
        preferences: list of numerical pairs where preferred trajectory is listed first
        metric: in class Metric, performs ranking of robustness across trajectories
        batches: splits robustness tests into weight_evaluations / batches 
        seed: randomness for reproducability set to None automatically

    Returns:
        accuracy splits across all splits.
    '''
    accuracy_list = {}
    valid_STL = list(wstl_list.keys())
    first_valid_STL = valid_STL[0]
    wstl = wstl_list[first_valid_STL]

    for split in range(1, valid_STL[-1]+1, 1):
        if split != first_valid_STL and split in valid_STL:
            wstl = link_STL([wstl, wstl_list[split]])
        print(wstl.wstl)
        print(wstl.signal_names)
        print(split)
        if split in splits:
            train = preferences[:split]
            if split != len(preferences)-1:
                test = preferences[split:]
            else:
                test = None
            accuracy_list[split] = compute_accuracy(wstl, data, weight_evaluations, train, test, metric, batches, seed)
    
    return accuracy_list