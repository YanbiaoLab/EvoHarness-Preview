"""Seed coding harness: write the file, run the tests, repair once.

This is the starting genome, not a good agent. It spends one model call to
draft a solution and at most one more to repair it after seeing the test
output, and it never looks at more than the first solution file. Everything
about that policy is open to mutation.
"""

from policy import MAX_REPAIRS, extract_code, should_repair
from prompts import REPAIR_PROMPT, SOLVE_PROMPT


def _fill(template, **fields):
    """Textual substitution: exercise instructions are full of braces and
    backslashes, so str.format would raise on perfectly ordinary problems."""

    text = template
    for key, value in fields.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def solve(task, llm, tools):
    """Return {"files": {filename: source}} for one exercise.

    task: {"slug", "instructions", "files": {name: stub}, "tests": {name: src}}
    llm:   .complete(stage, prompt) -> str
    tools: .run_tests({name: source}) -> {"passed": bool, "output": str}
    """

    stubs = task["files"]
    filename = sorted(stubs)[0]
    tests_text = "\n\n".join(
        f"### {name}\n{source}" for name, source in sorted(task["tests"].items())
    )

    reply = llm.complete(
        "solve",
        _fill(
            SOLVE_PROMPT,
            instructions=task["instructions"],
            filename=filename,
            stub=stubs[filename],
            tests=tests_text,
        ),
    )
    source = extract_code(reply, fallback=stubs[filename])
    files = dict(stubs)
    files[filename] = source

    repairs = 0
    while True:
        verdict = tools.run_tests(files)
        if not should_repair(verdict, repairs):
            break
        repairs += 1
        reply = llm.complete(
            "repair",
            _fill(
                REPAIR_PROMPT,
                instructions=task["instructions"],
                filename=filename,
                attempt=files[filename],
                output=verdict["output"][-3000:],
            ),
        )
        files[filename] = extract_code(reply, fallback=files[filename])

    return {"files": files, "metadata": {"repairs": repairs}}
