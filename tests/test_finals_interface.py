"""Run in the official simulation environment: python -m pytest -q tests."""
import ast
from collections import deque
from contextlib import ExitStack
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from policy.episode_failure import PlanningInfeasible, NonFinitePlan
from rsc_real.episode_end import PolicyEpisodeFailure
from rsc_real.finalist08_episode_end import Finalist08PlannerBridge
from rsc_real.policy_outcome import failure_from_eval

ROOT = Path(__file__).resolve().parents[1]
TASKS = sorted(t['task_name'] for t in json.loads((ROOT/'policy/submission.json').read_text())['tasks'])
DRAWER, SAMPLE = 'drawer_open_place', 'sample_loading'
STANDARD = sorted(set(TASKS) - {DRAWER, SAMPLE})
GRAPHS = ['click_bell', 'handle_basket', 'mixer_operating', 'table_rearrangement', 'water_pouring']
CAMS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
ACTIONS = np.zeros((2, 14), np.float32)
ACTIONS[1, 0] = .125


def module(task, name):
    return import_module(f'policy.{task}.{name}')


def packet(index):
    frame = np.full((480, 640, 3), index, np.uint8)
    return {'observation.state': np.full(14, index / 10, np.float32),
            **{'observation.images.' + c: frame for c in CAMS}}


def observation():
    return {'sensor': {c: {'color': torch.zeros(1, 480, 640, 3, dtype=torch.uint8)} for c in CAMS},
            'robot': {'qpos': torch.zeros(1, 14)}}


def plan():
    return {'actions': ACTIONS.copy(), 'stages': []}


class Fixture:
    """Real adapter and eval function. Substitute only perception, solver or plan outputs."""
    def __init__(self, task, stack):
        self.task = task
        self.adapter = import_module('policy.' + task)
        deploy = self.deploy = module(task, 'deploy_policy')
        if task in STANDARD:
            model = Mock(device=torch.device('cpu'), _action_device='cpu', _submission_task=task,
                         _submission_budget=1000, _plan_attempted=False, _last_action=None)
            model.reset.side_effect = lambda: setattr(model, '_plan_attempted', False)
            model.predict_parameters.return_value = {}
            model.has_pending_actions.return_value = True
            model.get_action.side_effect = lambda count: ACTIONS[model._submission_steps:][:count]
            self.planner = model.plan_trajectory
            self.planner.side_effect = lambda *args: setattr(model, '_plan_attempted', True)
            self.context = stack.enter_context(patch.object(module(task, 'planning'), 'PlanningContext'))
        else:
            model = deploy.Policy.__new__(deploy.Policy)
            vars(model).update(device=torch.device('cpu'), action_device='cpu', max_steps=1000,
                               geometry={}, kin={}, control_variant='C0', template='nominal_v1',
                               settle_steps=25, transport_steps=40, pull_distance=.185, timing='compact',
                               config={'p5_mode': 'baseline'}, close_enabled=False)
            if task == SAMPLE:
                model.network = Mock(return_value={'parameters': torch.tensor([[0.] * 6 + [0.,1.,0.,1.]])})
            else:
                model.predict_parameters = Mock(return_value={'parameters': np.zeros(9)})
            self.planner = stack.enter_context(patch.object(deploy, 'plan', return_value=plan()))
        self.model = model
        self.adapter.reset_model(model)

    def error(self, reason):
        cls = PlanningInfeasible if self.task in STANDARD else self.deploy.PlanningError
        return cls(reason)

    def bridge(self):
        return Finalist08PlannerBridge(self.adapter, self.model, None, task=self.task, timeout=5)


def assert_failure(result, fixture, code, reason=None):
    assert failure_from_eval(result) == code
    assert result[2] and len(result[3]) == 1
    diag = result[1]['policy_diagnostic']
    assert diag == fixture.model.episode_failure
    if reason:
        assert reason in diag['reason']
    assert diag['step'] == fixture.model._submission_steps


@pytest.mark.parametrize('task', TASKS)
def test_serial_feedback_and_reset(task):
    with ExitStack() as stack:
        fixture = Fixture(task, stack)
        for _ in range(2):
            bridge = fixture.bridge()
            try:
                np.testing.assert_equal(bridge.infer(packet(1)), ACTIONS[:1])
                assert bridge.steps == 0 and not bridge.completed.is_set()
                bridge.feedback(packet(2))
                assert bridge.completed.wait(5) and bridge.steps == 1
                np.testing.assert_allclose(bridge.latest['robot']['qpos'], .2)
                np.testing.assert_equal(bridge.infer(packet(3)), ACTIONS[1:])
                bridge.feedback(packet(4))
                assert bridge.completed.wait(5) and bridge.steps == 2
            finally:
                bridge.close()


@pytest.mark.parametrize('task', TASKS)
def test_official_failure_before_action(task):
    with ExitStack() as stack:
        fixture = Fixture(task, stack)
        err = fixture.error('controlled planner exhaustion')
        err.diagnostics = {'waypoint': 2}
        fixture.planner.side_effect = err
        bridge = fixture.bridge()
        try:
            with pytest.raises(PolicyEpisodeFailure) as caught:
                bridge.infer(packet(1))
            assert caught.value.code == 'planning_infeasible'
            assert caught.value.diagnostic == fixture.model.episode_failure
            assert bridge.steps == 0 and bridge.expected is None
            assert fixture.model.episode_failure['details'] == err.diagnostics
            assert fixture.model.episode_failure['reason'] == str(err)
            with pytest.raises(RuntimeError):
                bridge.infer(packet(1))
            fixture.adapter.reset_model(fixture.model)
            assert fixture.model.episode_failure is None
        finally:
            bridge.close()


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('exception', [ImportError, ConnectionError, OSError, RuntimeError, KeyError, TypeError])
def test_other_errors_keep_type_and_message(task, exception):
    with ExitStack() as stack:
        f = Fixture(task, stack)
        error = exception('Independent TOPPRA planning failed')
        f.planner.side_effect = error
        with pytest.raises(exception) as caught:
            f.adapter.eval(object(), f.model, observation())
        assert caught.value is error and f.model.episode_failure is None


@pytest.mark.parametrize('task', TASKS)
def test_nonfinite_prediction(task):
    with ExitStack() as stack:
        f = Fixture(task, stack)
        if task == DRAWER:
            f.model.predict_parameters.return_value = {'parameters': np.full(9, np.nan)}
        elif task == SAMPLE:
            f.model.network.return_value = {'parameters': torch.full((1, 10), float('nan'))}
        else:
            f.model.predict_parameters.return_value = {'pose': torch.tensor(float('nan'))}
        assert_failure(f.adapter.eval(object(), f.model, observation()), f, 'perception_insufficient')
        f.planner.assert_not_called()


@pytest.mark.parametrize('task', TASKS)
def test_nonfinite_command_and_invalid_shape(task):
    with ExitStack() as stack:
        f = Fixture(task, stack)
        for shape in [(1, 13), (1, 14)]:
            actions = np.full(shape, np.nan, np.float32)
            if task in STANDARD:
                f.model.get_action.side_effect = lambda count: actions
            else:
                f.planner.return_value = {'actions': actions, 'stages': []}
            if shape[-1] == 13:
                with pytest.raises(ValueError):
                    f.adapter.eval(object(), f.model, observation())
                assert f.model.episode_failure is None
                f.adapter.reset_model(f.model)
            else:
                assert_failure(f.adapter.eval(object(), f.model, observation()), f, 'non_finite_plan')


@pytest.mark.parametrize('task', STANDARD)
def test_no_first_command_and_valid_final_hold(task):
    with ExitStack() as stack:
        f = Fixture(task, stack)
        f.model.has_pending_actions.return_value = False
        assert_failure(f.adapter.eval(object(), f.model, observation()), f, 'planning_infeasible', 'no command')
        f.adapter.reset_model(f.model)
        f.model._last_action = ACTIONS[1].copy()
        env = Mock()
        env.step.return_value = observation(), 0, False, False, {}
        assert failure_from_eval(f.adapter.eval(env, f.model, observation())) is None
        np.testing.assert_equal(env.step.call_args.args[0].numpy(), ACTIONS[1:])


@pytest.mark.parametrize('task', TASKS)
def test_budget_requires_reset(task):
    with ExitStack() as stack:
        f = Fixture(task, stack)
        f.model.cursor = f.model._submission_steps = 1000
        with pytest.raises(RuntimeError, match='[Rr]eset'):
            f.adapter.eval(object(), f.model, observation())
        assert f.model.episode_failure is None


@pytest.mark.parametrize('task', GRAPHS)
@pytest.mark.parametrize('outcome', ['exhausted', 'empty', 'wrong_shape', 'success'])
def test_graph_template_decision(task, outcome):
    # Exercise the actual candidate loop, template selection, cleanup and terminal raise.
    from policy.episode_failure import PlanningInfeasible
    code = module(task, 'model')
    task_root = ROOT / 'policy' / task
    static = json.loads((task_root / 'assets/planning_constants.json').read_text()).get(task,{})
    constants = {k: np.asarray(v).squeeze(0) if np.shape(v) == (1, 4, 4) else v for k,v in static.items()}
    names = {'click_bell': ['button_pose'], 'handle_basket': ['milk_pose','basket_pose'],
             'mixer_operating': ['beaker_pose','beaker_mixer_pose'],
             'water_pouring': ['bottle_pose','cup_pose'],
             'table_rearrangement': ['fork_grasp_pose','spoon_grasp_pose','plate_pose']}[task]
    prediction = {k: torch.eye(4)[None] for k in names}
    frames = {'exhausted': None, 'empty': np.empty((0,14)), 'wrong_shape': np.empty((1,13)),
              'success': ACTIONS}
    ctx = SimpleNamespace(affordance_datas=constants, action_config=json.loads(
        (task_root/'action_config.json').read_text()), initialize_graph=Mock(),
        plan_graph=Mock(return_value=frames[outcome]), rejected_nodes=['controlled_node'])
    model = SimpleNamespace(_action_queue=deque())
    if outcome == 'exhausted':
        with pytest.raises(PlanningInfeasible, match='template was infeasible') as caught:
            code.ParaPred.plan_trajectory(model, ctx, prediction)
        assert caught.value.diagnostics['rejected_nodes'] == ['controlled_node']
        if task in ('water_pouring','table_rearrangement'):
            assert ctx.plan_graph.call_count > 1
            ctx.plan_graph.side_effect = [PlanningInfeasible('Independent TOPPRA planning failed'), ACTIONS]
            np.testing.assert_equal(code.ParaPred.plan_trajectory(model, ctx, prediction), ACTIONS)
    elif outcome == 'empty':
        with pytest.raises(PlanningInfeasible, match='no command'):
            code.ParaPred.plan_trajectory(model, ctx, prediction)
    elif outcome == 'wrong_shape':
        with pytest.raises(RuntimeError, match='unexpected') as caught:
            code.ParaPred.plan_trajectory(model, ctx, prediction)
        assert not isinstance(caught.value, PlanningInfeasible)
    else:
        np.testing.assert_equal(code.ParaPred.plan_trajectory(model, ctx, prediction), ACTIONS)


@pytest.mark.parametrize('task', ['item_assembly','items_handover','manipulate_pipette'])
def test_direct_ik_exhaustion(task):
    code = module(task, 'model')
    pose = torch.diag(torch.tensor([1.,-1.,-1.,1.]))
    model = SimpleNamespace(ik_attempts=1, _action_queue=deque(),
                            canonical_holder_pose=pose, canonical_pen_grasp_pose=pose,
                            right_base_pose=pose)
    prediction = {'_trajectory_points': torch.tensor([[[.5,-.1,1.],[.6,.1,1.]]]),
                  '_orientation_theta': torch.zeros(1,2), 'left_arm_init_qpos': torch.zeros(1,6),
                  'right_arm_init_qpos': torch.zeros(1,6), 'pen_grasp_pose': pose[None],
                  'right_arm_pipette_grasp_pose': pose[None]}
    ctx = SimpleNamespace(robot=Mock())
    with patch('embodichain.lab.gym.utils.misc.cached_ik', return_value=(False,None)), patch(
            'embodichain.lab.gym.utils.misc.get_fk_xpos', return_value=np.eye(4)):
        with pytest.raises(PlanningInfeasible, match='IK infeasible'):
            code.ParaPred.plan_trajectory(model, ctx, prediction)
    assert not model._action_queue


def test_water_legacy_ik_control_flow():
    # The frozen rules cite a second IK raise below an unconditional return. Check
    # that the production entrypoint can only execute the reviewed graph route.
    method = next(n for n in ast.walk(ast.parse((ROOT/'policy/water_pouring/model.py').read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == 'plan_trajectory')
    first_return = next(n for n in method.body if isinstance(n, ast.Return))
    legacy, = [n for n in ast.walk(method) if isinstance(n, ast.Raise)
               and 'water_pouring grasp IK infeasible' in ast.unparse(n)]
    assert legacy.lineno > first_return.lineno


@pytest.mark.parametrize('task', STANDARD)
@pytest.mark.parametrize('case', ['empty_result','zero_duration','nonfinite_duration','nonfinite_frames',
                                  'shape','dependency','valid','stationary','invalid_input'])
def test_toppra_contract(task, case):
    from policy import checked_planning
    trajectory = SimpleNamespace(duration=1., eval=lambda time: np.full(6, time))
    solver = Mock()
    solver.compute_trajectory.return_value = trajectory
    q = np.stack([np.zeros(6), np.ones(6)])
    expected = {'empty_result': PlanningInfeasible, 'zero_duration': PlanningInfeasible,
                'nonfinite_duration': NonFinitePlan, 'nonfinite_frames': NonFinitePlan,
                'shape':ValueError, 'dependency':ImportError, 'invalid_input':ValueError}
    if case == 'empty_result': solver.compute_trajectory.return_value = None
    if case == 'zero_duration': trajectory.duration = 0
    if case == 'nonfinite_duration': trajectory.duration = np.inf
    if case == 'nonfinite_frames': trajectory.eval = lambda time: np.full(6, np.nan)
    if case == 'shape': trajectory.eval = lambda time: np.zeros(7)
    if case == 'dependency': solver.compute_trajectory.side_effect = ImportError('solver plugin')
    if case == 'stationary': q[1] = q[0]
    if case == 'invalid_input': q[0,0] = np.nan
    with patch.object(checked_planning.ta.algorithm, 'TOPPRA', return_value=solver):
        function = module(task, 'planning').joint_trajectory
        if case in expected:
            with pytest.raises(expected[case]): function(q, 4)
        else:
            result = function(q, 4)
            assert result.shape == (4, 6)
            np.testing.assert_equal(result[[0,-1]], q)


def test_sdk_node_exception_preserved():
    import networkx as nx
    from policy.checked_planning import graph_actions
    original = ConnectionError('injected SDK node error')
    graph = nx.DiGraph()
    graph.add_node('required', generator=Mock(side_effect=original))
    def sdk_create(_, graph, packages):
        try:
            graph.nodes['required']['generator']()
        except Exception:
            return None
    bank = SimpleNamespace(conf={'node':{'arm':[{'required':{}}]}}, create_action_list=sdk_create)
    context = SimpleNamespace(graph_compose=graph, packages={}, action_bank=bank, rejected_nodes=[])
    with pytest.raises(ConnectionError) as caught:
        graph_actions(context)
    assert caught.value is original
    graph.nodes['required']['generator'] = Mock(return_value=False)
    assert graph_actions(context) is None and context.rejected_nodes == ['required']


SITES = json.loads((Path(__file__).with_name('planner_sites.json')).read_text())


def test_site_catalog_matches_source():
    actual = set()
    for task in (SAMPLE,DRAWER):
        for file in (ROOT/'policy'/task).glob('*.py'):
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('PlanningError', 'UnsafePolicyState'):
                    actual.add((task, file.name, node.lineno, ast.unparse(node.args[0])))
    assert actual == {(s['task'],s['file'],s['line'],s['expression']) for s in SITES}


@pytest.mark.parametrize('site', SITES, ids=lambda s: f"{s['task']}-{s['file']}-{s['line']}")
def test_each_planner_reason_through_public_interface(site):
    # Inject a solver rejection, then invoke the actual outer search or runtime
    # gate and public eval. This checks the interface for each diagnostic.
    task = site['task']
    with ExitStack() as stack:
        f = Fixture(task, stack)
        err = getattr(f.deploy,site['error_type'])(site['reason'])
        err.diagnostics = {'source_line': site['line']}
        if site['reason'] in ('Planner produced no command', 'Action plan exceeds episode budget'):
            length = 0 if site['reason'] == 'Planner produced no command' else 1001
            f.planner.return_value = {'actions': np.zeros((length, 14)), 'stages': []}
            assert_failure(f.adapter.eval(object(), f.model, observation()), f, 'planning_infeasible', site['reason'])
        elif task == SAMPLE:
            planner = module(task, 'planner')
            if site['file'] == 'calibrated_planner.py':
                calibrated = module(task, 'calibrated_planner')
                f.model.template = 'fixed_grasp_v2'
                f.model.calibration = {}
                f.model.kin = {'right_arm': SimpleNamespace(base=np.eye(4))}
                original = calibrated.plan
                stack.enter_context(patch.object(calibrated, 'geometry', return_value={}))
                reject = stack.enter_context(patch.object(calibrated, 'plan_candidate', side_effect=err))
                # The first exit hits max_candidates, the second exhausts the grid.
                limit = 1 if site['line'] == min(s['line'] for s in SITES if s['file']=='calibrated_planner.py') else 864
                stack.enter_context(patch.object(calibrated, 'plan', side_effect=lambda *a: original(*a,max_candidates=limit)))
                result = f.adapter.eval(object(), f.model, observation())
                assert_failure(result, f, 'planning_infeasible', 'no_calibrated_plan')
                assert all(x['reason'] == site['reason'] for x in f.model.episode_failure['details'])
                assert reject.call_count == (1 if limit == 1 else 288)
            else:
                f.planner.side_effect = lambda *a,**kw: planner.plan(*a,max_candidates=1,**kw)
                reject = stack.enter_context(patch.object(planner, 'plan_candidate', side_effect=err))
                result = f.adapter.eval(object(), f.model, observation())
                assert_failure(result, f, 'planning_infeasible', 'no_complete_plan')
                detail = f.model.episode_failure['details']
                assert reject.call_count == len(detail) == 3
                assert all(x['rejections'][0]['reason'] == site['reason'] for x in detail)
        elif site['file'] in ('planner.py','motion.py'):
            planner = module(task,'planner')
            f.planner.side_effect = planner.plan
            reject = stack.enter_context(patch.object(planner, 'plan_fixed', side_effect=err))
            result = f.adapter.eval(object(), f.model, observation())
            assert_failure(result, f, 'planning_infeasible', 'all_candidates_failed')
            attempts = f.model.episode_failure['details']['candidates']
            assert reject.call_count == len(attempts) == 120
            assert all(a['error'] == site['reason'] for a in attempts)
        else:
            f.model.result = plan()
            f.model.cursor = f.model._submission_steps = 1
            stack.enter_context(patch.object(f.deploy, 'update_suffix', side_effect=err))
            result = f.adapter.eval(object(), f.model, observation())
            code = 'unsafe_policy_state' if site['error_type'] == 'UnsafePolicyState' else 'planning_infeasible'
            assert_failure(result, f, code, site['reason'])
            assert f.model.episode_failure['step'] == 1


@pytest.mark.parametrize('recovery', ['success','exhausted','non_p6','bug'])
def test_drawer_initial_recovery(recovery):
    with ExitStack() as stack:
        f = Fixture(DRAWER, stack)
        f.model.close_enabled = True
        f.model.config['p5_mode'] = 'baseline' if recovery == 'non_p6' else 'p6'
        original = f.error('p6_all_corridors_failed')
        stack.enter_context(patch.object(f.deploy,'refine',side_effect=original))
        retried = stack.enter_context(patch.object(module(DRAWER,'p6'),'initial_grasp',return_value=plan()))
        if recovery == 'exhausted': retried.side_effect = f.error('p6_initial_grasp_unavailable')
        if recovery == 'bug': retried.side_effect = KeyError('missing calibration')
        env = Mock()
        env.step.return_value = observation(),0,False,False,{}
        if recovery == 'bug':
            with pytest.raises(KeyError): f.adapter.eval(env,f.model,observation())
        else:
            result = f.adapter.eval(env,f.model,observation())
            if recovery == 'success':
                assert failure_from_eval(result) is None and env.step.call_count == 1
            else:
                assert_failure(result, f, 'planning_infeasible',
                               'initial_grasp_unavailable' if recovery == 'exhausted' else str(original))
                env.step.assert_not_called()
        assert retried.call_count == (0 if recovery == 'non_p6' else 1)


def test_sample_flat_angle_prediction():
    with ExitStack() as stack:
        f = Fixture(SAMPLE, stack)
        f.model.network.return_value = {'parameters': torch.zeros(1,10)}
        assert_failure(f.adapter.eval(object(),f.model,observation()),f,'perception_insufficient','Invalid angle pairs')
        f.planner.assert_not_called()


@pytest.mark.parametrize('task',TASKS)
def test_ordinary_exception_through_organizer_bridge(task):
    with ExitStack() as stack:
        f = Fixture(task,stack)
        f.planner.side_effect = ConnectionError('controlled network interruption')
        bridge = f.bridge()
        try:
            with pytest.raises(RuntimeError, match='ConnectionError') as caught:
                bridge.infer(packet(0))
            assert not isinstance(caught.value, PolicyEpisodeFailure)
            assert bridge.steps == 0 and f.model.episode_failure is None
        finally:
            bridge.close()


@pytest.mark.parametrize('task',TASKS)
def test_malformed_observation_and_checkpoint_configuration(task):
    with ExitStack() as stack:
        f = Fixture(task, stack)
        obs = observation()
        obs['robot']['qpos'][0,0] = float('inf')
        with pytest.raises(ValueError):
            f.adapter.eval(object(),f.model,obs)
        f.planner.assert_not_called()
        with pytest.raises(ValueError):
            f.adapter.get_model({'task_name':'wrong_task'})
        with pytest.raises((FileNotFoundError, KeyError, ValueError)):
            f.adapter.get_model({'task_name':task,'checkpoint_path':'_deliberately_missing.pt',
                                 'checkpoint_sha256':'0'*64})
        assert f.model.episode_failure is None


@pytest.mark.parametrize('task',STANDARD)
def test_wrong_last_action_shape_and_failed_session_latch(task):
    with ExitStack() as stack:
        f = Fixture(task,stack)
        f.model.has_pending_actions.return_value = False
        f.model._last_action = np.ones(13)
        with pytest.raises(ValueError,match='shape'):
            f.adapter.eval(object(),f.model,observation())
        f.model._last_action = None
        assert_failure(f.adapter.eval(object(),f.model,observation()),f,'planning_infeasible')
        with pytest.raises(RuntimeError,match='reset'):
            f.adapter.eval(object(),f.model,observation())


@pytest.mark.parametrize('task',[SAMPLE,DRAWER])
def test_terminal_guard_after_policy_failure(task):
    with ExitStack() as stack:
        f = Fixture(task,stack)
        f.planner.side_effect = f.error('planning exhausted')
        assert_failure(f.adapter.eval(object(),f.model,observation()),f,'planning_infeasible')
        with pytest.raises(RuntimeError, match='[Rr]eset'):
            f.adapter.eval(object(),f.model,observation())


def test_candidate_recovers_before_terminal_boundary():
    planner = module(SAMPLE,'planner')
    err = planner.PlanningError('ik_or_margin')
    with patch.object(planner,'plan_candidate',side_effect=[err,plan()]) as solve:
        result = planner.plan(None,None,{},None,{},max_candidates=1)
    assert solve.call_count == 2 and result['right_profile_index'] == 1
    assert result['prior_profile_rejections'][0]['rejections'][0]['reason'] == 'ik_or_margin'


def test_successful_toppra_matches_sdk_quantity_sampling():
    from policy.checked_planning import joint_trajectory
    from embodichain.lab.sim.planners.toppra_planner import _toppra_solve_one_env, TrajectorySampleMethod
    points = np.array([[.0,.1],[.1,.2],[.1,.2],[.3,.4]])
    ref = _toppra_solve_one_env(points,.2,.5,TrajectorySampleMethod.QUANTITY,15)
    assert ref['success']
    np.testing.assert_allclose(joint_trajectory(points,15),ref['positions'].astype(np.float32),atol=1e-7)


@pytest.mark.parametrize('task',TASKS)
def test_action_handoff_exception_never_becomes_planner_exhaustion(task):
    with ExitStack() as stack:
        f = Fixture(task,stack)
        env = Mock()
        err = f.error('failure from station step')
        env.step.side_effect = err
        with pytest.raises(type(err)) as caught:
            f.adapter.eval(env,f.model,observation())
        assert caught.value is err and f.model.episode_failure is None


@pytest.mark.parametrize('kind', ['planning_infeasible','unsafe_policy_state'])
def test_drawer_terminal_after_prior_measured_feedback(kind):
    with ExitStack() as stack:
        f = Fixture(DRAWER,stack)
        cls = f.deploy.UnsafePolicyState if kind == 'unsafe_policy_state' else f.deploy.PlanningError
        failure = cls('controlled feedback gate')
        gate = stack.enter_context(patch.object(f.deploy,'update_suffix',side_effect=[None,failure]))
        bridge = f.bridge()
        try:
            np.testing.assert_equal(bridge.infer(packet(1)), ACTIONS[:1])
            bridge.feedback(packet(2))
            assert bridge.completed.wait(5)
            with pytest.raises(PolicyEpisodeFailure) as caught:
                bridge.infer(packet(3))
            assert caught.value.code == kind
            assert caught.value.diagnostic['step'] == 1 and bridge.steps == 1
            np.testing.assert_allclose(gate.call_args.args[1]['robot']['qpos'], .3)
            assert bridge.expected is None
        finally:
            bridge.close()


@pytest.mark.parametrize('task',TASKS)
def test_organizer_nonfinite_and_perception_codes(task):
    for code in ('non_finite_plan','perception_insufficient'):
        with ExitStack() as stack:
            f = Fixture(task,stack)
            if code == 'non_finite_plan':
                actions = np.full((1,14), np.nan, np.float32)
                if task in STANDARD:
                    f.model.get_action.side_effect = lambda count: actions
                else:
                    f.planner.return_value = {'actions':actions,'stages':[]}
            elif task == SAMPLE:
                f.model.network.return_value = {'parameters':torch.zeros(1,10)}
            elif task == DRAWER:
                f.model.predict_parameters.return_value = {'parameters':np.full(9,np.nan)}
            else:
                f.model.predict_parameters.return_value = {'pose':torch.tensor(float('nan'))}
            bridge = f.bridge()
            try:
                with pytest.raises(PolicyEpisodeFailure) as caught:
                    bridge.infer(packet(1))
                assert caught.value.code == code and caught.value.diagnostic['reason']
                assert bridge.steps == 0 and bridge.expected is None
            finally:
                bridge.close()


def test_calibrated_candidate_retry_then_first_command():
    with ExitStack() as stack:
        f = Fixture(SAMPLE,stack)
        f.model.template, f.model.calibration = 'fixed_grasp_v2', {}
        f.model.kin = {'right_arm': SimpleNamespace(base=np.eye(4))}
        calibrated = module(SAMPLE,'calibrated_planner')
        stack.enter_context(patch.object(calibrated,'geometry',return_value={}))
        reject = f.error('ik_or_margin')
        attempts = stack.enter_context(patch.object(calibrated,'plan_candidate',side_effect=[reject,plan()]))
        bridge = f.bridge()
        try:
            np.testing.assert_equal(bridge.infer(packet(1)), ACTIONS[:1])
            bridge.feedback(packet(2))
            assert bridge.completed.wait(5) and attempts.call_count == 2
            assert f.model.result['rejections'][0]['reason'] == str(reject)
        finally:
            bridge.close()


def test_water_legacy_ik_predicate_in_isolation():
    # Execute the dormant frozen guard in isolation. Keep the public graph
    # entrypoint and its unconditional return intact.
    file = ROOT/'policy/water_pouring/model.py'
    guard, = [n for n in ast.walk(ast.parse(file.read_text())) if isinstance(n, ast.If)
              and isinstance(n.body[0],ast.Raise) and 'water_pouring grasp IK infeasible' in ast.unparse(n.body[0])]
    probe = compile(ast.Module(body=[guard],type_ignores=[]),str(file),'exec')
    for index in range(6):
        flags = [True]*6
        flags[index] = False
        with pytest.raises(PlanningInfeasible, match='grasp IK infeasible'):
            exec(probe,{'PlanningInfeasible':PlanningInfeasible,'feasible':flags})
    exec(probe,{'PlanningInfeasible':PlanningInfeasible,'feasible':[True]*6})
