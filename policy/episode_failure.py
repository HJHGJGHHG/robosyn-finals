"""Explicit policy outcomes shared by the simulator and the serial robot adapter."""
import logging
import time

import numpy as np
import torch

from rsc_real.policy_outcome import failure_info


class PlanningInfeasible(RuntimeError):
    """The policy exhausted the allowed plans for this request."""


class PerceptionInsufficient(ValueError):
    """Validly shaped model output cannot define the required task geometry."""


class NonFinitePlan(RuntimeError):
    """The planner produced non-finite commands."""


def end_episode(model, obs, started, error, code='planning_infeasible'):
    """Finish before handoff and retain the original reason and planner evidence."""
    info = failure_info(code)
    model.episode_failure = {
        'code': code,
        'reason': str(error),
        'step': model._submission_steps,
        'details': getattr(error, 'diagnostics', getattr(error, 'rejections', None)),
    }
    cause = error.__cause__ or error.__context__
    if cause is not None:
        model.episode_failure['prior_reason'] = str(cause)
    logging.getLogger(__name__).warning('Policy episode ended at step %d: %s (%s)',
                                        model._submission_steps, error, code)
    info['policy_diagnostic'] = model.episode_failure
    info['elapsed_steps'] = np.int64(model._submission_steps)
    return obs, info, True, [time.perf_counter() - started]


def validate_prediction(prediction):
    for key, value in prediction.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise PerceptionInsufficient(f'Nonfinite prediction: {key}')
