import os
import sys
import subprocess
from  apithon import APIthonCodeExecuter

# For a finding to be considered valid, it must work on Python 3.8.10
# We will only accept findings that affect Python 3.8.10
REQUIRED_VERSION = "3.8.10"

def check_python_version(required_version):
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if current_version != required_version:
        raise Exception(f"This script requires Python {required_version}. Found Python {current_version}.")

#Make sure we don't pass in any functions, as that would bypass the allowlist of methods, and is out of scope for the Bug Bounty 
#Requires Python 3.8.10
def execute_apithon(code_to_run):
    code_executer = APIthonCodeExecuter({})
    try:
        result = code_executer.execute_code(code_to_run)
        return_value = result.return_value
        err = result.err #error value if err is not None
        line = result.line #line of error if err is not None
        if err is not None:
            print("Encountered error; printing data")
            print(f'{type(err)}{err.args}\n')
            print('Encountered on: ')
        else:
            print("Result is ", result) #prints entire object
            print("Return value is ", result.return_value) #prints the return value
            print("Log data", result.print_log) #prints the log from the code run; print statements are included
            print("====")
    except:
        print("Code size error, skipping sample")
    return

# Make sure we are in a Python 3.8.10 execution environment
check_python_version(REQUIRED_VERSION)

#basic print example
execute_apithon("print('hello world')")
multi_line_example = '''
a = 15 * 5
b = 1 + a / 2 - 4
print(a+b)
'''
#multi line example
execute_apithon(multi_line_example)

#example exception example: This should throw a NotImplementedError
execute_apithon('import ast')