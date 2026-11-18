# Policy test after the grasp stage

Weights tested: the stage's best-scoring evaluation (365 at step 101000). 20 fresh episodes, deterministic actions, fresh seed, in the sim. Raw numbers: `grasp_results.json`; videos: `videos_grasp/`.

| | random actions | run 2, this stage | run 1, same stage |
|---|---|---|---|
| closest the gripper gets to the cube (cm) | 4.2 | 0.3 | 0.2 |
| share of the episode within 2 cm of the cube | 1% | 83% | 92% |
| share of the episode pinching the cube | 0% | 81% | 88% |
| average best lift (cm) | 0.3 | 0.7 | 0.9 |
| best single lift (cm) | 0.8 | 1.2 | 1.0 |
| episodes that reached 5 cm | 0% | 0% | 0% |
| average cube pulled toward the base (cm) | n/a | -3.8 | 9.5 |
| episodes where it was pulled in more than 5 cm | n/a | 0% | 100% |

Average share of the episode held at 5 cm or higher: 0%.

## Flags
- not dragging the cube in (-3.8 cm on average; run 1 did this for every policy, 9 to 12 cm)
- pinches the cube (81% of the episode)
