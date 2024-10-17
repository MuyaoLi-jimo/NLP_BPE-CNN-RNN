"""
TODO: 
① 构造指定大小的list
② 计算时间
③ write an report
"""
import autograder
import copy
import numpy as np
import random
import string
from typing import Literal
import autograder

def flatten_list_sample(num:float=1e2):
    num_int = int(num)
    inputs = []
    outputs = []
    while num_int>0:
        k = np.random.randint(0, num_int + 1) 
        e = [i for i in range(k)]
        inputs.append(e)
        outputs.extend(e)
        num_int -= k
    return (inputs,outputs)

def char_count_sample(num:float=1e2):
    num_int = int(num)
    inputs = ""
    outputs = {}
    for i in range(num_int):
        random_letter = random.choice(string.ascii_lowercase)

        inputs+=random_letter
        temp = outputs.get(random_letter,0)
        outputs[random_letter]=temp+1
    return (inputs,outputs)

def create_sample(exercises,function_name: Literal["flatten_list", "char_count"],num:float=1e2):
    new_exercises = exercises
    if function_name=="flatten_list":
        new_exercises[function_name]["test_cases"].append(flatten_list_sample(num=num))
    elif function_name == "char_count":
        new_exercises[function_name]["test_cases"].append(char_count_sample(num=num))
    return new_exercises

if __name__ == "__main__":
    exercises = autograder.exercises
    num = 1.2
    for i in range(2,9,1):
        num = num*10
        exercises = create_sample(exercises=exercises,function_name="flatten_list",num=num)
        exercises = create_sample(exercises=exercises,function_name="char_count",num=num)
    submission_file = 'submission.py'
    user_module = autograder.load_module(submission_file) 
    results = autograder.run_tests(user_module, exercises)
    autograder.print_results(results)