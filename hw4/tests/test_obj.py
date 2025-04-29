"""
Unit tests for stack.py
"""
import deTAngling as dta
import pandas as pd
import numpy as np
import pytest


test1 = np.array(pd.read_csv('data/test1.csv', header=None))
test2 = np.array(pd.read_csv('data/test2.csv', header=None))
test3 = np.array(pd.read_csv('data/test3.csv', header=None))

@pytest.fixture
def s1():
    return dta.scores(test1)

@pytest.fixture
def s2():
    return dta.scores(test2)

@pytest.fixture
def s3():
    return dta.scores(test3)

def test1_scores(s1):
    assert s1['over'] == 37, 'wrong overallocation function'
    assert s1['conf'] == 8, 'wrong conflict function'
    assert s1['under'] == 1, 'wrong underallocation function'
    assert s1['unwill'] == 53, 'wrong unwillingness function'
    assert s1['unpref'] == 15, 'wrong unpreferred function'

def test2_scores(s2):
    assert s2['over'] == 41, 'wrong overallocation function'
    assert s2['conf'] == 5, 'wrong conflict function'
    assert s2['under'] == 0, 'wrong underallocation function'
    assert s2['unwill'] == 58, 'wrong unwillingness function'
    assert s2['unpref'] == 19, 'wrong unpreferred function'
    
def test3_scores(s3):
    assert s3['over'] == 23, 'wrong overallocation function'
    assert s3['conf'] == 2, 'wrong conflict function'
    assert s3['under'] == 7, 'wrong underallocation function'
    assert s3['unwill'] == 43, 'wrong unwillingness function'
    assert s3['unpref'] == 10, 'wrong unpreferred function'
