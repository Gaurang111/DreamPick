# Policy test after the reach stage

Weights tested: the stage's best-scoring evaluation (183 at step 91000). 20 fresh episodes, deterministic actions, fresh seed, in the sim. Raw numbers: `reach_results.json`; videos: `videos_reach/`.

| | random actions | run 2, this stage | run 1, same stage |
|---|---|---|---|
| closest the gripper gets to the cube (cm) | 4.2 | 0.6 | 0.3 |
| share of the episode within 2 cm of the cube | 1% | 74% | 86% |
| share of the episode pinching the cube | 0% | 36% | 1% |
| average best lift (cm) | 0.3 | 0.7 | 0.4 |
| best single lift (cm) | 0.8 | 1.1 | 0.5 |
| episodes that reached 5 cm | 0% | 0% | 0% |
| average cube pulled toward the base (cm) | n/a | -1.0 | 11.6 |
| episodes where it was pulled in more than 5 cm | n/a | 0% | 100% |

Average share of the episode held at 5 cm or higher: 0%.

## Flags
- not dragging the cube in (-1.0 cm on average; run 1 did this for every policy, 9 to 12 cm)
