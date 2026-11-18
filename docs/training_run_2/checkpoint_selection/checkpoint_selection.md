# Lift checkpoint selection

Each full checkpoint (world model + actor) of the lift stage, tested on the same 200 fresh cube positions (deterministic actions). Ranked by: share of the episode held at 5 cm or higher, then share of episodes that end holding the cube at 5 cm or higher, then fewest episodes where the cube ends more than 8 cm from its start.

| rank | step | held at 5 cm+ | ends holding | reached 5 cm | avg best lift (cm) | pinching | cube ends >8 cm away |
|---|---|---|---|---|---|---|---|
| 1 | 291000 | 82% | 91% | 96% | 7.0 | 85% | 6% |
| 2 | 241000 | 80% | 90% | 95% | 6.6 | 83% | 6% |
| 3 | 151000 | 1% | 0% | 9% | 3.5 | 80% | 10% |
| 4 | 141000 | 0% | 0% | 6% | 3.1 | 82% | 6% |
