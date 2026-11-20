"""Chunk-boundary adaptation without replacing observations with fake state."""
import ast
import hashlib
import inspect
import queue
import textwrap
import threading
import traceback
import numpy as np
from .episode_end import PolicyEpisodeFailure
from .policy_outcome import failure_from_eval
from .bridge import CAMERAS, FeedbackBridge, Stopped, action, observation, to_torch

CHUNK_PROTOCOL = 'rsc-real-chunk-v2'


def public_observation(payload):
    """The robot speaks one fixed OpenPI-style schema, independent of policy."""
    state=np.asarray(payload['observation.state'])
    sensors={name:{'color':np.asarray(payload['observation.images.'+name])[None]} for name in CAMERAS}
    return observation({'sensor':sensors,'robot':{'qpos':state[None]}})


def chunk_array(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    result=np.asarray(value)
    if result.ndim!=2 or result.shape[1]!=14 or not 1<=len(result)<=1000:
        raise ValueError('Expected a nonempty action chunk [H,14], H <= 1000')
    if result.dtype.kind not in 'fiu' or not np.isfinite(result).all():
        raise ValueError('Invalid action chunk values')
    with np.errstate(over='ignore'):result=result.astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Chunk exceeds float32 range')
    return result


def intercept_eval(function):
    """Wrap only the explicit precomputed-action loop used by public templates.

    Keep the original inference, slicing, action conversion and loop body. Refuse
    unknown layouts instead of silently fabricating step results or using H=1.
    """
    source=textwrap.dedent(inspect.getsource(function))
    tree=ast.parse(source);definition=tree.body[0]
    if not isinstance(definition,ast.FunctionDef) or definition.decorator_list:
        raise ValueError('Chunk adapter requires an undecorated public eval function')
    candidates=[]
    for loop in definition.body:
        if not isinstance(loop,ast.For) or not isinstance(loop.target,ast.Name) or not isinstance(loop.iter,ast.Name) or loop.orelse:
            continue
        first=loop.body[0]
        call=first.value if isinstance(first,(ast.Assign,ast.Expr)) else None
        if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and
                isinstance(call.func.value,ast.Name) and call.func.value.id=='env' and
                call.func.attr=='step' and len(call.args)==1 and
                isinstance(call.args[0],ast.Name) and call.args[0].id==loop.target.id):
            candidates.append(loop)
    steps=[n for n in ast.walk(definition) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and isinstance(n.func.value,ast.Name) and n.func.value.id=='env' and n.func.attr=='step']
    if len(candidates)!=1 or len(steps)!=1:
        raise ValueError('Policy needs an explicit reviewed chunk adapter: no unique precomputed action loop')
    loop=candidates[0]
    if any(isinstance(n,(ast.While,ast.For,ast.Return,ast.Yield,ast.YieldFrom)) for stmt in loop.body for n in ast.walk(stmt)):
        raise ValueError('Nested control flow in action loop requires review')
    for node in ast.walk(loop):
        if isinstance(node,ast.Continue):raise ValueError('Conditional action skipping requires review')
        if isinstance(node,ast.Break):
            parent=next((p for p in ast.walk(loop) if isinstance(p,ast.If) and node in p.body),None)
            condition=ast.unparse(parent.test) if parent else ''
            if not parent or parent.orelse or len(parent.body)!=1 or condition not in (
                    'truncated', 'truncated.any()', 'torch.as_tensor(truncated).any()',
                    "env.get_wrapper_attr('is_task_success')()"):
                raise ValueError('Early chunk termination requires a reviewed adapter')
    step_assignment=loop.body[0]
    if isinstance(step_assignment,ast.Assign):
        target=step_assignment.targets[0]
        if not isinstance(target,(ast.Tuple,ast.List)) or len(target.elts)!=5 or any(isinstance(e,ast.Starred) for e in target.elts):
            raise ValueError('Expected the public five-value env.step contract')
    loop.iter=ast.Call(func=ast.Attribute(value=ast.Name(id='env',ctx=ast.Load()),attr='action_chunk',ctx=ast.Load()),args=[loop.iter],keywords=[])
    ast.fix_missing_locations(tree)
    namespace={}
    exec(compile(tree,inspect.getsourcefile(function) or '<chunk-adapter>','exec'),function.__globals__,namespace)
    return namespace[definition.name],{'kind':'precomputed-action-loop-v1',
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'adapted_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()}


class ChunkBridge(FeedbackBridge):
    def __init__(self,adapter,model,instruction,convert=to_torch,timeout=120):
        super().__init__(adapter,model,instruction,convert,timeout)
        self.evaluate,self.adaptation=intercept_eval(adapter.eval)
        self.feedbacks=queue.Queue(2)
        self.expected=None;self.cursor=0;self.active=False
        self.completed=threading.Event();self.completed.set()

    def action_chunk(self,values):
        originals=list(values)
        values=chunk_array(np.concatenate([action(v) for v in originals],axis=0))
        count=min(len(values),1000-self.steps)
        if count<=0:raise Stopped()
        self.expected=values[:count].copy();self.cursor=0
        self.outputs.put(('actions',self.expected.copy()),timeout=self.timeout)
        return originals[:count]

    def step(self,value):
        if self.expected is None or self.cursor>=len(self.expected):
            raise ValueError('Policy step outside its declared chunk')
        if not np.array_equal(action(value)[0],self.expected[self.cursor]):
            raise ValueError('Policy mutated an action after chunk handoff')
        while not self.stopped.is_set():
            try:actual=self.feedbacks.get(timeout=.1);break
            except queue.Empty:continue
        else:raise Stopped()
        self.latest=actual;self.steps+=1;self.cursor+=1
        return self.convert(actual),0.,self.convert(np.array([False])),self.convert(np.array([self.steps>=1000])),{'elapsed_steps':self.steps}

    def reviewed_failure(self, error):
        return None

    def _run(self):
        try:
            self.adapter.reset_model(self.model)
            self.outputs.put(('reset_done',None),timeout=self.timeout)
            while not self.stopped.is_set() and self.steps<1000:
                self.latest=self._take();self.expected=None
                try:
                    result = self.evaluate(self,self.model,self.convert(self.latest))
                    code = failure_from_eval(result)
                except Exception as error:
                    failure = self.reviewed_failure(error) if self.expected is None else None
                    if failure is None:
                        raise
                    code = failure.code
                if code is not None:
                    if self.expected is not None:
                        raise ValueError('Policy failure must precede action handoff for this inference')
                    self.outputs.put(('episode_end', code), timeout=self.timeout)
                    self.completed.set()
                    return
                if self.expected is None or self.cursor!=len(self.expected):
                    raise ValueError('Policy did not consume the complete handed-off chunk')
                self.completed.set()
        except Stopped:pass
        except BaseException as error:
            traceback.print_exc()
            self.stopped.set();self.completed.set()
            try:self.outputs.put(('error',type(error).__name__),timeout=1)
            except queue.Full:pass

    def infer(self,payload):
        self.prepare()
        if not self.completed.wait(self.timeout) or self.stopped.is_set():
            raise RuntimeError('Previous chunk feedback or policy processing incomplete')
        self.completed.clear()
        self.inputs.put(public_observation(payload),timeout=self.timeout)
        kind,value=self.outputs.get(timeout=self.timeout)
        if kind=='episode_end':raise PolicyEpisodeFailure(value)
        if kind!='actions':raise RuntimeError('Chunk policy failed: '+str(value))
        return value

    def feedback(self,payload):
        if self.stopped.is_set():raise RuntimeError('Chunk policy stopped')
        self.feedbacks.put(public_observation(payload),timeout=self.timeout)

    def finish(self):
        if not self.completed.wait(self.timeout) or self.stopped.is_set():
            raise RuntimeError('Final chunk history did not complete')
        self.close()


class ChunkGuard:
    def __init__(self,attempt_id,lower,upper,fps=10,max_age_seconds=2):
        from .protocol import ActionGuard
        self.base=ActionGuard(attempt_id,lower,upper,max_age_seconds)
        if fps<=0:raise ValueError('Positive official action frequency required')
        self.fps=fps;self.pending=None;self.index=0;self.sequence=0

    @property
    def steps(self):return self.base.steps

    @property
    def lower(self):return self.base.lower

    @property
    def upper(self):return self.base.upper

    @property
    def failed(self):return self.base.failed

    def accept(self,response,received):
        try:
            if self.pending is not None or self.failed:raise ValueError('Previous chunk is unfinished')
            if response.get('attempt_id')!=self.base.attempt_id or response.get('seq')!=self.sequence:
                raise ValueError('Chunk identity or sequence mismatch')
            values=chunk_array(response['actions']).copy()
            if self.steps+len(values)>1000 or np.any(values<self.base.lower) or np.any(values>self.base.upper):
                raise ValueError('Chunk exceeds approved action bounds or budget')
            self.pending=values;self.received=received;self.index=0
            return values.copy()
        except BaseException:
            self.base.failed=True;raise

    def execute(self,value,command):
        import time
        try:
            if self.failed or self.pending is None or not np.array_equal(np.asarray(value),self.pending[self.index]):
                raise ValueError('Unapproved or changed chunk action')
            age=time.monotonic()-self.received
            if age<0 or age>self.base.max_age+self.index/self.fps:raise ValueError('Chunk execution stalled')
            expected=self.pending[self.index].copy()
            self.base.steps+=1;self.index+=1
            command(expected)
            if self.index==len(self.pending):self.pending=None;self.sequence+=1
        except BaseException:
            self.base.failed=True;raise
