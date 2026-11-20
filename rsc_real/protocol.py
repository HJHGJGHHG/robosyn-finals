"""Authoritative checks on the LOCAL side; remote model claims are not scores."""
import time
import numpy as np
from .bridge import action

PROTOCOL = 'rsc-real-v1'
IDENTITY_KEYS = ('request_id', 'bundle_sha', 'task', 'task_config_sha')


def identity_matches(actual, expected):
    if actual.get('protocol') != PROTOCOL or any(actual.get(k) != expected[k] for k in IDENTITY_KEYS):
        raise ValueError('Wrong model endpoint identity')


class ActionGuard:
    def __init__(self, attempt_id, lower, upper, max_age_seconds=2):
        self.attempt_id = attempt_id
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        if self.lower.shape != (14,) or self.upper.shape != (14,) or not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper):
            raise ValueError('Explicit verified 14-dimensional joint/gripper bounds required')
        if not 0 < max_age_seconds <= 30:
            raise ValueError('Invalid local action age limit')
        self.max_age = max_age_seconds
        self.steps = 0
        self.failed = False

    def execute(self, response, received_at, command):
        if self.failed or self.steps >= 1000:
            raise ValueError('Attempt cannot issue more commands')
        try:
            if response.get('attempt_id') != self.attempt_id or response.get('seq') != self.steps:
                raise ValueError('Action attempt/sequence mismatch')
            age = time.monotonic() - received_at
            if not 0 <= age <= self.max_age:
                raise ValueError('Action expired locally')
            value = action(response['actions'])[0]
            if np.any(value < self.lower) or np.any(value > self.upper):
                raise ValueError('Action exceeds verified joint/gripper bounds')
            # Consume before sending: a transport exception may occur AFTER motion.
            self.steps += 1
            command(value.copy())
            return value
        except BaseException:
            self.failed = True
            raise
