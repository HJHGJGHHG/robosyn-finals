# Finalist 08 serial interface correction

Request `real_80ba4e01d73e45`, revision 2. The policy base is
`2f6a0062b04774888bbcd7f9022be316b95459dd`. The organizer bundle reference is
`657adbc367875ebd560cb230b2211f99a0fab5aa`. All ten policy paths, checkpoint
revisions, checksums, units and deployment YAML bindings are retained.

## Interface and results

The team bridge uses the organizer's single-command transport. Six supplied files
(`policy_outcome.py`, `episode_end.py`, `bridge.py`, `chunks.py`, `protocol.py`
and `single_step_chunk.py`) match the archive byte for byte. The revised
`rsc_real/finalist08_episode_end.py` verifies the corrected adapter and planner
sources against the pinned source manifest. A task returns the organizer's
four-value eval result. The nested `rsc_episode_failure` field carries the code.
Detailed planner diagnostics are retained on the model, in a sibling info field,
and on the locally raised terminal exception. A failure precedes action handoff
in that call. A new session resets all terminal state.

The 75 server ran **365 interface checks: 365 passed, zero failed, zero skipped**
in 50.79 seconds. PyTorch emitted 35 JIT deprecation warnings. The tests use the
actual exported task eval/reset functions, production failure routing and the
organizer bridge. The station packets are test fixtures. Tests control solver
returns, model outputs and feedback to exercise specific decisions. Results for
each case appear in [the JUnit record](results/interfaces.xml).

| Task | Policy-level condition and expected response | Interface evidence |
|---|---|---|
| Click Bell | The graph rejects the button template, the trajectory solver exhausts, or the graph supplies zero commands. End with `planning_infeasible`. | Graph exhaustion, empty result, valid result, malformed shape, solver and bridge tests passed. |
| Handle Basket | The predicted handle grasp graph rejects the template or yields no command. TOPPRA failure also terminates. | Same graph, solver and bridge checks passed. |
| Mixer Operating | The mixer template graph rejects the estimated configuration or yields no command. TOPPRA failure also terminates. | Same graph, solver and bridge checks passed. |
| Table Rearrangement | Try the available approach lengths and orientations. On candidate rejection continue the search. On exhaustion end with the template reason and rejected nodes or underlying solver reason. | Candidate exhaustion, a later successful candidate, zero commands and malformed output checked. |
| Water Pouring | Try the grasp-template pairs and propagate the last solver rejection on exhaustion. | Graph decision, retry, terminal, invalid-output and success paths checked. The legacy grasp-IK guard below the production return is tested in isolation for each of the six IK flags. A structural test checks its position relative to the production return. |
| Item Assembly | Any required grasp/lift IK endpoint remains infeasible after the seed attempts, or TOPPRA cannot connect the endpoints. | The public planner's IK guard and trajectory-result branches were exercised. |
| Items Handover | A required waypoint remains unreachable after IK and retreat alternatives, or a trajectory has no feasible time parameterization. | Direct IK exhaustion, TOPPRA outcomes, empty command, valid terminal hold and transport checks passed. |
| Manipulate Pipette | A required grasp, transfer, press-search or retreat waypoint remains infeasible after the applicable attempts, or TOPPRA exhausts. | Direct IK exhaustion after fallback search, TOPPRA outcomes and command boundary checks passed. |
| Drawer Open Place | All initial route candidates fail, refinement and its permitted initial-grasp recovery fail, or a suffix cannot be planned. End with `planning_infeasible`. Measured contact, tracking, release or clearance gate violations marked unsafe terminate with `unsafe_policy_state`. | The 120-candidate search, startup fallback, failed/recovered suffix scenarios, prior-action feedback boundary, invalid perception and nonfinite command paths were checked. |
| Sample Loading | Plan from the supplied first RGB/qpos packet. Exhausting nominal profiles or the calibrated search returns `planning_infeasible`. Invalid predicted angle pairs or nonfinite estimates return `perception_insufficient`. Empty, over-budget or nonfinite plans terminate before action handoff. | Nominal profile exhaustion/recovery, both calibrated exits, later calibrated success, normal first action, feedback and reset checks passed. |

The [source-site inventory](planner_sites.json) enumerates 56 Drawer and 32 Sample
planner reason sites and their conditional expressions. The suite checks this
inventory against the source tree. It injects each reason at a solver or
execution boundary and checks terminal behavior and preservation of diagnostic
details. Loop exhaustion and representative retry paths are also executed in the
actual outer planners. This is interface-level coverage of each reason.
It supplies no physical success-rate estimate.

All ten tasks have checks for malformed input, reset/budget guards, nonfinite
perception, nonfinite commands, wrong command dimensions and ordinary dependency,
network, OS and programming faults. The same words as a planner failure inside
an ordinary exception remain an error. The graph adapter preserves SDK node
callback exceptions even if the SDK converts them to a negative graph result.
The joint planner uses the SDK's constraints and TOPPRA sampling settings and
allows solver exceptions to propagate. A successful reference trajectory is
compared numerically with the SDK.

## Run

Install the official simulation dependencies and copy both `policy/` and
`rsc_real/` into the pinned official evaluation checkout
(`45d9b9c4c78c66ef340a055697f3ad833ce567f4`). Copy this `tests/` directory
alongside them. From that checkout run:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_finals_interface.py --junitxml=interfaces.xml
```

For real evaluation, select `rsc_real.finalist08_episode_end.Finalist08PlannerBridge`
for the reviewed task and negotiate the organizer's `policy-episode-end-v1`
capability. Keep its per-action feedback contract and the registered task's
action units and bounds. The organizers control deployment and queue admission.
