"""Prompt templates for the seed coding harness."""

SOLVE_PROMPT = """You are solving a Python programming exercise.

# Instructions
{instructions}

# The file you must write: {filename}
Its current contents (a stub you must replace):
```python
{stub}
```

# The tests that will judge you
```python
{tests}
```

Write the complete final contents of {filename}. Match the names, signatures
and exception types the tests use exactly. Reply with one Python code block
and nothing else.
"""

REPAIR_PROMPT = """Your solution to a Python exercise failed its tests.

# Instructions
{instructions}

# What you submitted as {filename}
```python
{attempt}
```

# Test output
```
{output}
```

Diagnose the failure and write the corrected complete contents of {filename}.
Reply with one Python code block and nothing else.
"""
