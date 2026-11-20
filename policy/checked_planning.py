"""Use the SDK's TOPPRA settings while preserving solver and graph exceptions."""
import numpy as np
import toppra as ta
from toppra import constraint
from embodichain.lab.sim.planners.toppra_planner import _build_constraint_arrays

from .episode_failure import NonFinitePlan, PlanningInfeasible


def joint_trajectory(waypoints, steps, velocity=.2, acceleration=.5):
    q = np.asarray(waypoints, dtype=np.float64)
    if q.ndim != 2 or len(q) < 2 or not np.isfinite(q).all() or int(steps) != steps or steps < 2:
        raise ValueError('Invalid joint trajectory input')
    if all(np.linalg.norm(a - b) <= 1e-3 for a, b in zip(q, q[1:])):
        return np.repeat(q[:1], int(steps), axis=0).astype(np.float32)
    indices = [0]
    for i in range(1, len(q)):
        if np.max(np.abs(q[i] - q[indices[-1]])) >= 1e-6:
            indices.append(i)
    if indices[-1] != len(q) - 1:
        indices.append(len(q) - 1)
    q = q[indices]
    vlims, alims = _build_constraint_arrays(velocity, acceleration, q.shape[1])
    path = ta.SplineInterpolator(np.linspace(0, 1, len(q)), q)
    solver = ta.algorithm.TOPPRA(
        [constraint.JointVelocityConstraint(vlims), constraint.JointAccelerationConstraint(alims)],
        path, parametrizer='ParametrizeConstAccel', gridpt_min_nb_points=max(100, 10 * len(q)))
    trajectory = solver.compute_trajectory()
    if trajectory is None or trajectory.duration <= 0:
        raise PlanningInfeasible('Independent TOPPRA planning failed')
    if not np.isfinite(trajectory.duration):
        raise NonFinitePlan('Nonfinite TOPPRA duration')
    frames = np.asarray([trajectory.eval(t) for t in np.linspace(0, trajectory.duration, int(steps))],
                        dtype=np.float32)
    if frames.shape != (steps, q.shape[1]):
        raise ValueError(f'Unexpected TOPPRA positions shape {frames.shape}')
    if not np.isfinite(frames).all():
        raise NonFinitePlan('Nonfinite TOPPRA positions')
    return frames


def graph_actions(context):
    """The SDK turns exceptions in node callbacks into False. Preserve their cause."""
    graph = context.graph_compose.copy()
    errors = []
    rejected = []

    def observe(name, generator):
        def call(*args, **kwargs):
            try:
                accepted = generator(*args, **kwargs)
                if not accepted:
                    rejected.append(name)
                return accepted
            except Exception as exc:
                errors.append(exc)
                raise
        return call

    for nodes in context.action_bank.conf['node'].values():
        for item in nodes:
            name, = item
            node = graph.nodes[name]
            node['generator'] = observe(name, node['generator'])
    actions = context.action_bank.create_action_list(context, graph, context.packages)
    if errors:
        raise errors[0]
    if actions is None:
        if not rejected:
            raise RuntimeError('Action graph returned no plan without a node rejection')
        context.rejected_nodes.extend(rejected)
    return actions
