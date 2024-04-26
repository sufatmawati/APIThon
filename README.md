# Requirements

This requires Python version 3.8.10

# How to Run

See `example_use.py` for how to use the class.
Please note that **we do not consider** functions passed into the `APIthonCodeExecuter` class during initialization as a valid finding.

# What's in scope?
Since the scope could change from time to time, we recommend looking [here](https://hackerone.com/moveworks/policy_scopes) prior to writing and running any tests. 

# File Breakdown

- `example_use.py` - How to use the apithon class
- `apithon_impl.py` - The underlying base operations that can be used by apithon
- `apithon.py` - The apithon class itself
- `exceptions.py` - Exceptions apithon can throw