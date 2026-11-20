"""A negotiated terminal result for an explicitly recognized policy failure."""

from .policy_outcome import failure_info

CAPABILITY = "policy-episode-end-v1"
REASONS = {
    "perception_insufficient": "The policy could not obtain usable perception.",
    "planning_infeasible": "The policy could not generate a feasible plan.",
    "unsafe_policy_state": "The policy detected an unsafe state.",
    "non_finite_plan": "The policy could not produce a finite action plan.",
    "policy_cannot_continue": "The policy reported that the episode cannot continue.",
}


class PolicyEpisodeFailure(Exception):
    def __init__(self, code):
        failure_info(code)
        self.code = code
        super().__init__(REASONS[code])
