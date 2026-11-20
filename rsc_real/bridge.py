"""Runs in an untrusted model container, never in the robot controller process.

Each env.step yields one action and waits for actual post-command observation.
This preserves public adapters that update their history inside eval().
"""
import queue
import threading
from types import SimpleNamespace
import numpy as np
from .episode_end import PolicyEpisodeFailure
from .policy_outcome import failure_from_eval

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def observation(value):
    out = {'sensor': {}, 'robot': {}}
    for name in CAMERAS:
        image = np.asarray(value['sensor'][name]['color'])
        if image.dtype != np.uint8 or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
            raise ValueError('Expected batched uint8 RGB images')
        if not 1 <= image.shape[1] <= 2160 or not 1 <= image.shape[2] <= 3840:
            raise ValueError('Image dimensions exceed contract')
        out['sensor'][name] = {'color': image.copy()}
    for key in ('qpos', 'qvel'):
        if key not in value['robot'] and key == 'qvel':
            continue
        data = np.asarray(value['robot'][key])
        if data.shape != (1, 14) or data.dtype.kind not in 'fiu' or not np.isfinite(data).all():
            raise ValueError('Expected finite [1,14] robot state')
        out['robot'][key] = data.astype(np.float32)
    return out


def action(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.shape == (14,):
        value = value[None, :]
    if value.shape != (1, 14) or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
        raise ValueError('Expected one finite joint action [1,14]')
    if np.abs(value).max(initial=0) > np.finfo(np.float32).max:
        raise ValueError('Action outside float32 range')
    return value.astype(np.float32)


def to_torch(value):
    import torch
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        return {k: to_torch(v) for k, v in value.items()}
    return value


class Stopped(BaseException):
    pass


class FeedbackBridge:
    def __init__(self, adapter, model, instruction, convert=to_torch, timeout=120):
        self.adapter, self.model = adapter, model
        self.convert, self.timeout = convert, timeout
        self.inputs, self.outputs = queue.Queue(1), queue.Queue(1)
        self.stopped = threading.Event()
        self.worker = None
        self.steps = 0
        self.latest = None
        self.unwrapped = self
        self.device, self.num_envs = 'cpu', 1
        # Wire gripper commands use normalized [0,1]. These are interface
        # bounds, not inferred physical arm limits; the onsite client enforces
        # its separately approved hardware limits before executing commands.
        lower = np.full(14, -np.inf, dtype=np.float32)
        upper = np.full(14, np.inf, dtype=np.float32)
        lower[[6, 13]], upper[[6, 13]] = 0.0, 1.0
        self.single_action_space = SimpleNamespace(shape=(14,), low=lower, high=upper)
        self.action_space = SimpleNamespace(shape=(1, 14))
        self._current_instruction = instruction

    def _take(self):
        while not self.stopped.is_set():
            try:
                return self.inputs.get(timeout=.1)
            except queue.Empty:
                continue
        raise Stopped()

    def step(self, value):
        if self.stopped.is_set() or self.steps >= 1000:
            raise Stopped()
        self.outputs.put(('action', action(value)), timeout=self.timeout)
        self.latest = self._take()
        self.steps += 1
        return self.convert(self.latest), 0., self.convert(np.array([False])), self.convert(np.array([self.steps >= 1000])), {'elapsed_steps': self.steps}

    def get_wrapper_attr(self, name):
        if name == 'is_task_success':
            return lambda: False
        if name in ('device', 'num_envs', 'action_space', 'single_action_space', '_current_instruction'):
            return getattr(self, name)
        raise AttributeError('Unsupported real environment attribute: ' + name)

    def _run(self):
        try:
            self.adapter.reset_model(self.model)
            self.outputs.put(('reset_done', None), timeout=self.timeout)
            self.latest = self._take()
            while not self.stopped.is_set() and self.steps < 1000:
                before = self.steps
                result = self.adapter.eval(self, self.model, self.convert(self.latest))
                code = failure_from_eval(result)
                if code is not None:
                    if before != self.steps:
                        raise ValueError('Policy failure must precede actions in this eval call')
                    raise PolicyEpisodeFailure(code)
                if before == self.steps:
                    raise ValueError('Adapter returned without calling env.step')
        except Stopped:
            pass
        except PolicyEpisodeFailure as failure:
            self.outputs.put(('episode_end',failure.code),timeout=self.timeout)
        except BaseException as exc:
            try:
                self.outputs.put(('error', type(exc).__name__), timeout=1)
            except queue.Full:
                pass

    def prepare(self, *, ready_timeout=None):
        if self.stopped.is_set():
            raise RuntimeError('Bridge is closed; reset with a new attempt')
        if self.worker is None:
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()
            try:
                kind, value = self.outputs.get(timeout=self.timeout if ready_timeout is None else ready_timeout)
            except queue.Empty as exc:
                self.close()
                raise TimeoutError('Policy reset timeout') from exc
            if kind != 'reset_done':
                self.close()
                raise RuntimeError('Policy reset failed: ' + str(value))

    def infer(self, obs):
        self.prepare()
        obs = observation(obs)
        try:
            self.inputs.put(obs, timeout=self.timeout)
            kind, value = self.outputs.get(timeout=self.timeout)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError('Policy response timeout') from exc
        if kind == 'episode_end':
            self.close()
            raise PolicyEpisodeFailure(value)
        if kind != 'action':
            self.close()
            raise RuntimeError('Policy failed: ' + value)
        return value

    def close(self):
        self.stopped.set()
        if self.worker:
            self.worker.join(timeout=2)
            if self.worker.is_alive():
                raise RuntimeError('Policy did not stop; container restart required')
