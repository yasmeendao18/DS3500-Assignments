"""
Emma Shek, Sabrina Zhou, Lilian Uong, Felix Yang, Yasmeen Dao
DS3500 / fanTAstic
Evolutionary Computing / 4
3/22/23 / 3/28/23
"""

from evo import Evo
import random as rnd
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import chain

# define data here
sect_data = pd.read_csv('data/sections.csv')
tas_data = pd.read_csv('data/tas.csv')

# objective 1
def over_allocation(L, max=tas_data['max_assigned']):
    """ Objective: Count the number of penalities for overallocating a TA """
    diff = np.sum(L, axis=1) - max
    return sum(x for x in diff if x > 0)

# objective 2
def conflicts(L):
    """ Objective: Count the number of time conflicts """
    ta_info = [list(zip(ta, sect_data['daytime'])) for ta in L]

    # For each ta, filter the list to only include sections that tas were assigned to
    ta_assigns = [list(filter(lambda x: 0 not in x, item)) for item in ta_info]

    # For each row in array, does TA have conflict
    return sum([1 for item in ta_assigns if Counter(item) != Counter(list(set(item)))])

# objective 3
def under_support(L, min_ta=sect_data['min_ta']):
    """ Objective: Count the number of penalities for undersupporting a section """
    ones_counts = np.sum(L, axis=0)
    L = np.vstack((L, min_ta))
    L = np.vstack((L, ones_counts))
    return sum([x - y for x, y in zip(L[-2], L[-1]) if y < x])

# objective 4
def unwilling(L):
    """ Objective: count the number of times you allocate a TA to a section they are unwilling to support """

    # mold into workable dataframe
    test_data = pd.DataFrame(L)
    test_data.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    ta_willingness = tas_data.loc[:, '0':'16']

    # convert "U" to 1 and convert "P" and "W" to 0
    bool_ta = ta_willingness.applymap(lambda x: 1 if x == 'U' else 0)

    # multiply new bool_ta df with test df
    unwilling_result = bool_ta & test_data

    # count the sum of values in df to see when "U" was in the same location as 1
    count = unwilling_result.sum().sum()
    
    return count

# objective 5
def unpreferred(L):
    """ Objective: count the number of times you allocate a TA to a section where they said 'willing' but not 'preferred' """
    
    # mold into workable dataframe
    test_data = pd.DataFrame(L)
    test_data.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    ta_willingness = tas_data.loc[:, '0':'16']

    # convert "U" to 1 and convert "P" and "W" to 0
    bool_ta = ta_willingness.applymap(lambda x: 1 if x == 'W' else 0)

    # multiply new bool_ta df with test df
    unwilling_result = bool_ta & test_data

    # count the sum of values in df to see when "U" was in the same location as 1
    count = unwilling_result.sum().sum()
    
    return count

# generate random solution
def generate_rand_sol(size=(43,17)):
    return np.random.randint(2, size=size)

# build agents
def swap_tas(solutions):
    """ Swap two random values column-wise of the 2d array """
    L = solutions[0]
    j = rnd.randrange(0, L.shape[1])
    idx1 = np.where(L[:,j] == 1)[0]
    idx0 = np.where(L[:,j] == 0)[0]
    if len(idx1) == 0 or len(idx0) == 0:
        return L
    i = np.random.choice(idx0)
    k = np.random.choice(idx1)
    L[k][j], L[i][j] = L[i][j], L[k][j]
    return L

def swap_section(solutions):
    """ Swap two random values row-wise of the 2d array """
    L = solutions[0]
    i = rnd.randrange(0, L.shape[0])
    idx1 = np.where(L[i] == 1)[0]
    idx0 = np.where(L[i] == 0)[0]
    if len(idx1) == 0 or len(idx0) == 0:
        return L
    j = np.random.choice(idx0)
    k = np.random.choice(idx1)
    L[i][j], L[i][k] = L[i][k], L[i][j]
    return L

def remove_ta_section(solutions):
    """ remove a ta from a section that they are assigned to, if they 
        are not assigned to anything, then check a different ta """
    L = solutions[0]
    # array empty check
    if np.sum(L) == 0:
        return L
    
    # select random TA
    i = rnd.randrange(0, L.shape[0])
    idx = np.where(L[i] == 1)[0] # check assigned sections
    while len(idx) == 0:
        i = rnd.randrange(0, L.shape[0])
        idx = np.where(L[i] == 1)[0]
    j = np.random.choice(idx)
    L[i][j] = 0
    return L

def remove_ta(solutions):
    """ remove a ta from a section (could already not be assigned) """
    L = solutions[0]
    i = rnd.randrange(0, L.shape[0])
    j = rnd.randrange(0, L.shape[1])
    L[i][j] = 0
    return L

def add_ta_section(solutions):
    """ assign a ta to a section not assigned to, if they 
        are assigned to everything, check a different ta """
    L = solutions[0]
    # array full check
    if np.sum(L) == L.shape[0] * L.shape[1]:
        return L
    
    # select random TA
    i = rnd.randrange(0, L.shape[0])
    idx = np.where(L[i] == 0)[0] # check unassigned sections
    while len(idx) == 0:
        i = rnd.randrange(0, L.shape[0])
        idx = np.where(L[i] == 0)[0]
    j = np.random.choice(idx)
    L[i][j] = 1
    return L

def add_ta(solutions):
    """ assign a ta to a section (could already be assigned) """
    L = solutions[0]
    i = rnd.randrange(0, L.shape[0])
    j = rnd.randrange(0, L.shape[1])
    L[i][j] = 1
    return L

# get scores from all of the above functions for a certain solution
def scores(L):
    ''' determine scores for test cases '''
    ss = {}
    ss['over'] = over_allocation(L)
    ss['conf'] = conflicts(L)
    ss['under'] = under_support(L)
    ss['unwill'] = unwilling(L)
    ss['unpref'] = unpreferred(L)
    return ss

def output_solutions(evo):
    ''' output the solutions as csv with solutions '''
    # unpack the evo framework
    combined_dict = defaultdict(list)
    cvals = evo.pop.keys()
    sol = evo.pop.values()
    evals = list(chain.from_iterable(cvals))
    for key, value in evals:
        combined_dict[key].append(value)
        
    # build dataframe
    out_df = pd.DataFrame(combined_dict)
    out_df['solution'] = list(sol)
    out_df['groupname'] = ['TAlendar'] * len(out_df)
    out_df[['groupname', 'overallocation', 'conflicts', 'undersupport', 'unwilling', 'unpreferred']].to_csv('final_output.csv')
    return out_df
    

# run evolutionary framework
def main():
    
    # Create framework
    E = Evo()

    # Register some objectives
    E.add_fitness_criteria('overallocation', over_allocation)
    E.add_fitness_criteria('conflicts', conflicts)
    E.add_fitness_criteria('undersupport', under_support)
    E.add_fitness_criteria("unwilling", unwilling)
    E.add_fitness_criteria("unpreferred", unpreferred)

    # Register some agents
    E.add_agent("add_ta", add_ta, k=1)
    E.add_agent("add_ta_to_secion", add_ta_section, k=1)
    E.add_agent("remove_ta", remove_ta, k=1)
    E.add_agent("remove_ta_from_section", remove_ta_section, k=1)
    E.add_agent("swap_tas", swap_tas, k=1)
    E.add_agent("swap_section", swap_section, k=1)

    # Seed the population with an initial random solution
    L = generate_rand_sol(size=(len(tas_data), len(sect_data)))
    E.add_solution(L)
    print(E)

    # Run the evolver
    E.evolve(10000000 , 100, 100000, time_limit=600)

    # Print final results
    output_solutions(E)

if __name__ == '__main__':
    main()

