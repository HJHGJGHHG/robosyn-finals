"""Public policy outcome vocabulary, independent of the evaluator and robot."""

from collections.abc import Mapping

FAILURE_KEY = "rsc_episode_failure"
FAILURE_CODES = frozenset({
    "perception_insufficient",
    "planning_infeasible",
    "unsafe_policy_state",
    "non_finite_plan",
    "policy_cannot_continue",
})


def failure_info(code):
    """Return an info mapping for an unrecoverable current episode."""
    if not isinstance(code, str) or code not in FAILURE_CODES:
        raise ValueError("Unknown policy episode failure code")
    return {FAILURE_KEY: {"code": code}}


def read_failure(info):
    """Return a recognized code, or None for an ordinary evaluation result."""
    if not isinstance(info, Mapping) or FAILURE_KEY not in info:
        return None
    report = info[FAILURE_KEY]
    if not isinstance(report, Mapping) or set(report) != {"code"}:
        raise ValueError("Malformed policy episode failure")
    code = report["code"]
    failure_info(code)
    return code


def failure_from_eval(result):
    """Read the optional failure field from the public four-value eval result."""
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        return None
    return read_failure(result[1])
