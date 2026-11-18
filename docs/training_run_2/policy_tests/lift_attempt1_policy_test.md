# Policy test after the lift stage

Weights tested: the stage's best-scoring evaluation (723 at step 31000). 20 fresh episodes, deterministic actions, fresh seed, in the sim. Raw numbers: `lift_results.json`; videos: `videos_lift/`.

| | random actions | run 2, this stage | run 1, same stage |
|---|---|---|---|
| closest the gripper gets to the cube (cm) | 4.2 | 0.6 | 0.7 |
| share of the episode within 2 cm of the cube | 1% | 41% | 40% |
| share of the episode pinching the cube | 0% | 62% | 66% |
| average best lift (cm) | 0.3 | 2.0 | 1.8 |
| best single lift (cm) | 0.8 | 2.9 | 2.7 |
| episodes that reached 5 cm | 0% | 0% | 0% |
| average cube pulled toward the base (cm) | n/a | -1.4 | 10.9 |
| episodes where it was pulled in more than 5 cm | n/a | 0% | 100% |

Average share of the episode held at 5 cm or higher: 0%.

## Flags
- not dragging the cube in (-1.4 cm on average; run 1 did this for every policy, 9 to 12 cm)
- pinches the cube (62% of the episode)
- never reached 5 cm; best lift 2.9 cm (run 1 best: 2.7 cm)
