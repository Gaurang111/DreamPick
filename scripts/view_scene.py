"""Open MuJoCo's live interactive viewer on the pick-cube scene.

Lets you orbit/zoom the camera, drag joints by hand, and inspect the model -
no RL, no Gymnasium, just the raw MuJoCo scene. Close the window to exit.
"""

import sys
from pathlib import Path

import mujoco
import mujoco.viewer

SCENE_PATH = Path(__file__).resolve().parents[1] / "assets" / "scene_pick_cube.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Opening viewer on {SCENE_PATH.name} - close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    sys.exit(main())
