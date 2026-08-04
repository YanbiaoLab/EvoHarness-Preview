<!-- polyglot_py task_sys_msg v1 (2026-08-04).
     Injected into every mutation prompt. Deliberately NOT written here:
     how many attempts to spend, what to put in the prompts, whether to read
     the tests, or how to spend the test-run budget. Those are the search
     space; naming the answers would make every arm identical. -->

# Task: evolve the coding harness, not the code it writes

You are mutating a three-file program that solves Python exercises from
Aider's polyglot benchmark. It is scored on how many exercises' real test
suites pass — you never see the reference solutions, and the score always
comes from a test run performed after your harness returns.

## The genome

- `solver.py` — must define `solve(task, llm, tools)`, the entry point.
- `policy.py` — control decisions (how many attempts, how to read a reply).
- `prompts.py` — the wording sent to the model.

The three-file split exists so a mutation can change the control flow without
re-emitting the prompts, or rewrite a prompt without touching control flow.
Only `solver.py` is imported by the runner; the other two are yours to use,
restructure, or ignore, as long as the imports stay consistent.

## The contract

```python
def solve(task, llm, tools) -> dict:
    ...
    return {"files": {filename: source}}
```

`task` is a dict:

- `task["slug"]` — the exercise name, e.g. `"bowling"`.
- `task["instructions"]` — the exercise text, exactly as a human competitor
  sees it (markdown, often with examples).
- `task["files"]` — `{filename: stub source}`, the file(s) you must write.
  Most exercises have exactly one; a few have more.
- `task["tests"]` — `{filename: source}`, the actual test files that will
  judge the result. They are given to you in full.

`llm.complete(stage, prompt, system="")` returns the model's reply as a
string. `stage` is a free-form label used for logging.

`tools.run_tests(files)` runs the exercise's tests against a `{filename:
source}` mapping and returns `{"passed": bool, "output": str}` — the real
pytest output, truncated to the last 4000 characters.

Return every file the exercise needs, not just the ones you changed. Files you
do not return are absent when the tests run.

## The budgets, per exercise

- at most 6 `llm.complete` calls
- at most 3 `tools.run_tests` calls
- 60 s per test run; exceeding any cap raises inside your harness, and an
  exercise whose harness raises scores zero for that exercise only

The budget is per exercise, not shared across the set: spending nothing on an
easy exercise buys nothing on a hard one.

## How you are scored

fitness = (exercises whose tests pass) / (exercises attempted). The evaluation
also reports, per exercise, whether it passed, a short failure category
(`syntax-error`, `import-error`, `assertion-failed`, `test-timeout`,
`harness-…` when your own code raised), and how much of each budget was spent.

An exercise that fails because your harness crashed is categorised apart from
one that fails because the code was wrong — if most exercises fail inside the
harness, the run is marked as not measuring the program at all.

## Rules

- Standard library only, plus whatever the exercise's own tests import. No
  network access besides `llm`.
- Do not attempt to read the reference solution, the corpus directory, or
  anything outside the workspace you are handed; files written outside it are
  rejected.
- Keep `solve` synchronous and single-threaded.
