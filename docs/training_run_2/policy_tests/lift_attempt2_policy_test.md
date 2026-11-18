# Policy test after the lift stage

Weights tested: the stage's best-scoring evaluation (1030 at step 141000). 20 fresh episodes, deterministic actions, fresh seed, in the sim. Raw numbers: `lift_results.json`; videos: `videos_lift/`.

| | random actions | run 2, this stage | run 2, first attempt | run 1, same stage |
|---|---|---|---|---|
| closest the gripper gets to the cube (cm) | 4.2 | 0.2 | 0.6 | 0.7 |
| share of the episode within 2 cm of the cube | 1% | 21% | 41% | 40% |
| share of the episode pinching the cube | 0% | 88% | 62% | 66% |
| average best lift (cm) | 0.3 | 3.2 | 2.0 | 1.8 |
| best single lift (cm) | 0.8 | 8.5 | 2.9 | 2.7 |
| episodes that reached 5 cm | 0% | 5% | 0% | 0% |
| average cube pulled toward the base (cm) | n/a | 0.9 | -1.4 | 10.9 |
| average distance the cube ended from where it started, any direction (cm) | n/a | 3.6 | 22.0 | 12.9 |
| episodes where the cube ended more than 8 cm from its start | n/a | 5% | 95% | 95% |

Average share of the episode held at 5 cm or higher: 0%.

## Flags
- not dragging the cube in (0.9 cm on average; run 1 did this for every policy, 9 to 12 cm)
- keeps the cube near where it started (3.6 cm on average from its start; run 2's first lift attempt: about 23 cm)
- pinches the cube (88% of the episode)
- REACHED the 5 cm target; best lift 8.5 cm (run 1 best: 2.7 cm, run 2 first attempt: 2.9 cm)
