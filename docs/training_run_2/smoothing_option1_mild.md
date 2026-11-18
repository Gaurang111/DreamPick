# Option 1: smoothing the policy's commands (open-loop replay in the sim)

The step-291k lift policy was recorded on 50 cube positions; each recorded joint-target trajectory was smoothed (low-pass filter time constant, limit on how fast the commanded speed may change, optional slow-down) and replayed open-loop from the same start state. Motion numbers are the median over episodes of the 5 arm joints' measured motion in the sim.

| setting | reached 5 cm | ends holding | cube ends >8 cm away | peak accel (rad/s^2) | peak speed (rad/s) | speed while holding (rad/s) | direction flips while holding |
|---|---|---|---|---|---|---|---|
| no smoothing (original) | 94% | 92% | 4% | 57 | 1.90 | 0.21 | 48% |
| accel limit 40 only | 94% | 90% | 4% | 57 | 1.90 | 0.18 | 43% |
| accel limit 25 only | 50% | 14% | 18% | 45 | 1.66 | 0.19 | 44% |
| light filter 0.05 s | 18% | 10% | 10% | 31 | 1.30 | 0.12 | 30% |
| filter 0.1 s | 4% | 4% | 10% | 23 | 1.20 | 0.07 | 27% |
| filter 0.1 s + accel 25 | 4% | 4% | 10% | 23 | 1.20 | 0.07 | 27% |
| 2x slower only (no filter) | 62% | 22% | 24% | 27 | 1.27 | 0.12 | 23% |
| 1.5x slower only (no filter) | 70% | 40% | 20% | 37 | 1.32 | 0.12 | 31% |
