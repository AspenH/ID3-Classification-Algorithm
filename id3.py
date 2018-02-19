'''
Programmed by: Slate Hayes, Aspen Henry, Jake Thomas
2/19/2018
Description: This program constructs a deceision tree for data classification using the ID3 Algorithm.
'''
import math
import pprint as pp
import copy

# Returns a list of all classes in the data set
def get_class_list(training_data):
    classes = []
    for tup in training_data:
        if tup[1] not in classes:
            classes.append(tup[1])
    return classes

# Returns the info needed to classify an object
def info_at_root(training_data, attr_list):
    i = 0 # info needed to classify object
    classes = {} # Dictionary to store the frequencies at which each class occurs at the root
    for tup in training_data:
        if tup[1] not in classes.keys():
            classes[tup[1]] = 1
        else:
            classes[tup[1]] += 1

    # c - class name, f - frequency
    # p - probability of that class in training_data
    for c, f in classes.items():
        p = f / len(training_data)
        i -= p * math.log2(p)

    return i

# Returns info needed to classify an object with a value for a given attribute
def info_for_value_of_attr(training_data, value, class_freqs):
    classes = get_class_list(training_data)
    value_i = 0 # info needed to classify object
    total_freq = 0 # Keep track of the total frequencies of each class
    for c, v in class_freqs.items():
        if c == value:
            for x in v.values():
                total_freq += x

    # perform the summation calculation for each class
    for c in classes:
        p = class_freqs[value][c] / total_freq
        if p != 0:
            value_i -= p * math.log2(p)
        else:
            value_i = 0

    return value_i

# Returns the expected info needed to classify a random sample using a given attribute - magic
# EX: values = { "Senior": { False: 2, True: 3}, ... }
def entropy(training_data, attr_list, attr):
    e = 0 # entropy

    classes = get_class_list(training_data)

    # the parent dictionary that will contain the
    # dictionary of frequencies for each class for every value for this attribute
    values = {}

    # build the class frequencies for each value
    for tup in training_data:
        for a in tup[0]:
            if a == attr:
                if tup[0][a] not in values:
                    values[tup[0][a]] = None
                    class_freqs = {}
                    for c in classes:
                        class_freqs[c] = 0

                    class_freqs[tup[1]] += 1
                    values[tup[0][a]] = class_freqs
                else:
                    values[tup[0][a]][tup[1]] += 1

    # dictionary that will hold the total frequencies
    total_freqs_for_values = {}

    for val, c_freq in values.items():
        if val not in total_freqs_for_values:
            total_freqs_for_values[val] = 0
        for f in c_freq.values():
                total_freqs_for_values[val] += f

    # perform the entropy summation for each value of the attribute
    for val, freq in values.items():
        e += ( total_freqs_for_values[val] / len(training_data) * info_for_value_of_attr(training_data, val, values))

    return e

# Returns the info gain for the given attribute
def gain(training_data, attr_list, attr):
    return info_at_root(training_data, attr_list) - entropy(training_data, attr_list, attr)

# Returns the attribute with the maximum info gain
def find_best_attr(training_data, attr_list):
    max_gain = -1 # the numerical max gain
    max_attr = None # the attribute that gives us the max gain

    for attr in attr_list:
        attr_gain = gain(training_data, attr_list, attr)
        if attr_gain > max_gain:
            max_gain = attr_gain
            max_attr = attr

    return max_attr

# ID3 algorithm
def id3(training_data, attr_list):
    node = [] # this list gets converted to a tuple when we're ready to add it to the tree
    branches = {} # Dictionary to store the branches to children nodes

    # Base case 1: If every tuple has the same class, return a node with that class
    for tup_index in range(len(training_data)):
        if tup_index != len(training_data)-1:
            if training_data[tup_index][1] != training_data[tup_index+1][1]:
                break
        else:
            if training_data[tup_index-1][1] == training_data[tup_index][1]:
                return training_data[tup_index][1]

    # Base case 2: If the attribute list is empty, return a node with the class of the majority
    if len(attr_list) == 0:
        tuples = [tup for tup in training_data]
        class_freq = {t[1]: 0 for t in tuples}
        for tup in tuples:
            if tup[1] in class_freq.keys():
                class_freq[tup[1]] += 1
        majority = list(class_freq.keys())[0]
        for k, v in class_freq.items():
            if v > class_freq[majority]:
                majority = k
        return majority

    # Main part
    test_attr = find_best_attr(training_data, attr_list)

    # construct the shell of the node
    node.append(test_attr)
    node.append(branches)

    # get rid of the attribute that we're splitting on
    attr_list.remove(test_attr)

    # Get all the values we know of for test_attr
    values = []

    for tup in training_data:
        for attr in tup[0]:
            if attr == test_attr and tup[0][attr] not in values:
                values.append(tup[0][attr])

    for val in values:
        bad_training_data = copy.deepcopy(training_data)

        new_training_data = []

        for t in range(len(bad_training_data)):
            if bad_training_data[t][0][test_attr] == val:
                new_training_data.append(bad_training_data[t])

        branches[val] = id3(new_training_data, attr_list)

    tuples = [tup for tup in training_data]
    class_freq = {t[1]: 0 for t in tuples}
    for tup in tuples:
        if tup[1] in class_freq.keys():
            class_freq[tup[1]] += 1
    new_training_data_majority = list(class_freq.keys())[0]
    for k, v in class_freq.items():
        if v > class_freq[new_training_data_majority]:
            new_training_data_majority = k

    branches[None] = new_training_data_majority

    return tuple(node)


def classify(dt, tup):

    # we're assuming that when a node is no longer a tuple, we've reached the leaf node
    while isinstance(dt, tuple):
        attr = dt[0] # get the attribute that we're checking against the test sample
        values = list(dt[1].keys()) # get all the possible values for that attribute

        # if the test sample has an unknown value or attribute, panic and return the majority class at that node
        # otherwise, we step down the decision tree and repeat
        if attr in tup.keys() and tup[attr] in values:
            for val in values:
                if tup[attr] == val:
                    dt = dt[1][val]
        else:
            return dt[1][None]

    return dt

if __name__ == "__main__":

    training_data = [
        ({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'}, False),
        ({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'}, False),
        ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
        ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
        ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
        ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, False),
        ({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, True),
        ({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, False),
        ({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
        ({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'}, True),
        ({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),
        ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, True),
        ({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'}, True),
        ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False)
    ]

    attr_list = [] # this list stores all of the attributes that are considered at each step of the algorithm

    for tup in training_data:
        for i in tup[0].keys():
            if i not in attr_list:
                attr_list.append(i)

    # construct the decision tree
    dt = id3(training_data, attr_list)

    # display the decision tree
    print("\t-------- Decision Tree --------\n")
    pp.pprint(dt)
    print("\n\n\t-------- Test sample classifications --------\n")

    # these are the test samples that we need to classify
    test_samples = [{"level": "Junior", "lang": "Java", "tweets": "yes", "phd": "no"},
                    {"level": "Junior", "lang": "Java", "tweets": "yes", "phd": "no"},
                    {"level": "Intern"},
                    {"level": "Senior"}]

    # display the class for each test sample
    for sample in range(len(test_samples)):
        cls = classify(dt, test_samples[sample])
        print("Test sample {}: {}\n\t- Class: {}\n".format(sample, test_samples[sample], cls))
