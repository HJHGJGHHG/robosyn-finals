"""Explicit one-command chunks for USTC's reviewed one-step evaluator contract."""
import ast
import hashlib
import inspect
import queue
import re
import textwrap
import threading

from .bridge import FeedbackBridge, action, to_torch
from .chunks import ChunkBridge

ADAPTER = 'ustc-single-step-v1'


def review(adapter):
    function = adapter.eval
    if function.__name__ != 'evaluate' or not re.fullmatch(
            r'policy\.[a-z_]+\.submission_runtime', function.__module__):
        raise ValueError('Single-step adapter requires the reviewed USTC runtime')
    source = textwrap.dedent(inspect.getsource(function))
    definition = ast.parse(source).body[0]
    steps = [node for node in ast.walk(definition) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'step'
             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'env']
    direct_steps = [statement for statement in definition.body if isinstance(statement, ast.Assign)
                    and statement.value in steps]
    if len(steps) != 1 or len(direct_steps) != 1 or any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(definition)):
        raise ValueError('Reviewed single-step evaluator must call env.step once outside loops')
    return {'kind': ADAPTER, 'source_sha256': hashlib.sha256(source.encode()).hexdigest()}


class SingleStepChunkBridge(ChunkBridge):
    def __init__(self, adapter, model, instruction, convert=to_torch, timeout=120, *, reviewer=review):
        FeedbackBridge.__init__(self, adapter, model, instruction, convert, timeout)
        self.adaptation = reviewer(adapter)
        self.evaluate = adapter.eval
        self.feedbacks = queue.Queue(2)
        self.expected = None
        self.cursor = 0
        self.active = False
        self.completed = threading.Event()
        self.completed.set()

    def action_chunk(self, values):
        raise ValueError('Single-step evaluator cannot publish a precomputed chunk')

    def step(self, value):
        if self.expected is not None:
            raise ValueError('Single-step evaluator attempted multiple commands')
        self.expected = action(value).copy()
        self.cursor = 0
        self.outputs.put(('actions', self.expected.copy()), timeout=self.timeout)
        return super().step(value)
