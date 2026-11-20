"""Source-pinned one-command bridge for finalist_08's explicit policy outcomes."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

from .episode_end import PolicyEpisodeFailure
from .single_step_chunk import SingleStepChunkBridge

# Generated from the reviewed correction; these hashes are independent of the
# mutable task manifests shipped alongside checkpoints.
PIN_FILE = Path(__file__).with_name('finalist08_sources.json')
PIN_SHA256 = 'fd0bb3bc6f9195fdbb08f1676785ec7881657d72489f71198c956d8f75bca9a8'


def review_policy(adapter, task):
    data = PIN_FILE.read_bytes()
    if hashlib.sha256(data).hexdigest() != PIN_SHA256:
        raise ValueError('Unreviewed finalist08 source manifest')
    pins = json.loads(data)
    evaluator = Path(inspect.getsourcefile(adapter.eval)).resolve()
    root = evaluator.parents[1]
    entry = pins['tasks'][task]
    if evaluator != root / task / entry['entrypoint']:
        raise ValueError('Unexpected finalist08 evaluator')
    for name, digest in {**pins['shared'], **entry['sources']}.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Unreviewed finalist08 source: {name}')
    tree = ast.parse(textwrap.dedent(inspect.getsource(adapter.eval)))
    steps = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'step'
             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'env']
    if len(steps) != 1 or any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree)):
        raise ValueError('Expected one command from the reviewed evaluator')
    return {'kind': 'finalist08-planner-episode-end-v1', 'feedback': 'measured-per-action'}


class Finalist08PlannerBridge(SingleStepChunkBridge):
    def __init__(self, adapter, model, instruction, *, task, **options):
        super().__init__(adapter, model, instruction,
                         reviewer=lambda selected: review_policy(selected, task), **options)

    def infer(self, payload):
        try:
            return super().infer(payload)
        except PolicyEpisodeFailure as error:
            error.diagnostic = self.model.episode_failure
            self.close()
            raise
