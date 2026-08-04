"""Control decisions for the seed coding harness, kept apart from the prompts
so a mutation can change how many attempts to spend without rewriting the
wording, and the other way round."""

# How many repair rounds to spend after the first attempt fails.
MAX_REPAIRS = 1


def should_repair(verdict: dict, repairs_done: int) -> bool:
    """Spend another attempt on this exercise?"""

    if verdict.get("passed"):
        return False
    return repairs_done < MAX_REPAIRS


def extract_code(text: str, fallback: str = "") -> str:
    """Pull the source out of a model reply.

    Models fence code inconsistently — ```python, bare ```, or nothing at all.
    Returning the raw reply on a miss is deliberate: a reply that IS code with
    no fence should still be tried, and a reply that is prose will simply fail
    its tests, which is information rather than a crash.
    """

    marker = "```"
    if marker not in text:
        return text.strip() or fallback
    blocks = []
    parts = text.split(marker)
    for index in range(1, len(parts), 2):
        block = parts[index]
        newline = block.find("\n")
        if newline != -1 and block[:newline].strip().isalpha():
            block = block[newline + 1:]      # drop the language tag
        blocks.append(block)
    if not blocks:
        return text.strip() or fallback
    return max(blocks, key=len).strip() or fallback
