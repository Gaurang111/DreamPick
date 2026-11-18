# Option 1: smoothing the policy's commands (open-loop replay in the sim)

The step-291k lift policy was recorded on 50 cube positions; each recorded joint-target trajectory was smoothed (low-pass filter time constant, limit on how fast the commanded speed may change, optional slow-down) and replayed open-loop from the same start state. Motion numbers are the median over episodes of the 5 arm joints' measured motion in the sim.

| setting | reached 5 cm | ends holding | cube ends >8 cm away | peak accel (rad/s^2) | peak speed (rad/s) | speed while holding (rad/s) | direction flips while holding |
|---|---|---|---|---|---|---|---|
| no smoothing (original) | 92% | 84% | 14% | 54 | 2.00 | 0.22 | 48% |
| tremor filter only | 2% | 0% | 2% | 15 | 0.99 | nan | nan% |
| filter + accel 20 | 2% | 0% | 2% | 15 | 0.99 | nan | nan% |
| filter + accel 10 | 2% | 0% | 2% | 15 | 0.99 | nan | nan% |
| filter + accel 10, 1.5x slower | 6% | 0% | 2% | 12 | 0.75 | 0.08 | 12% |
| filter + accel 8, 2x slower | 8% | 6% | 0% | 11 | 0.66 | 0.04 | 12% |
| strong filter + accel 5, 2x slower | 0% | 0% | 2% | 7 | 0.47 | nan | nan% |
