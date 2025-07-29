
import sys
import carb
import numpy as np
from pathlib import Path
from typing import Optional
import isaacsim.core.api.objects
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.controllers.base_controller import BaseController

# import MPlib
# add Isaac Sim's Python site-packages dynamically
this_file = Path(__file__).resolve()
isaacsim_root = this_file
while isaacsim_root.name != "isaacsim-4.5.0":
    isaacsim_root = isaacsim_root.parent
site_packages = isaacsim_root / "kit" / "python" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

if site_packages.exists() and str(site_packages) not in sys.path:
    sys.path.append(str(site_packages))

from mplib import Planner, Pose

ext_root = this_file.parents[2] # Goes to MPlib_extension/
urdf_path = ext_root / "assets/urdf/panda/franka_panda.urdf"
srdf_path = ext_root / "assets/urdf/panda/franka_panda.srdf"

class FrankaMplibController(BaseController):
    def __init__(self, name: str, robot_articulation: SingleArticulation, physics_dt=1/60.0):
        super().__init__(name)
        self._robot = robot_articulation
        self._physics_dt = physics_dt
        self._planner = Planner(
            urdf=str(urdf_path),
            move_group="panda_hand",
            srdf=str(srdf_path),
            verbose=False
        )
        self._action_sequence = []
        self._last_solution = None

    def _make_new_plan(self, target_pos, target_orn=None, finger_override=None):
        if self._robot is None:
            print("Robot articulation not set.")
            return []

        current_joints = self._robot.get_joint_positions() 
        finger_q = np.array(finger_override) if finger_override is not None else current_joints[7:]
    
        # Use MPLib IK to find a goal joint config
        goal_pose = Pose(
            p=np.array(target_pos).reshape(3, 1),
            q=np.array(target_orn).reshape(4, 1)
        )

        status, q_goals = self._planner.IK(goal_pose=goal_pose, start_qpos=current_joints)
        if status != "Success" or q_goals is None:
            print("[MPLib] IK failed!")
            return None
        
        plan_result = self._planner.plan_pose(goal_pose, current_joints.tolist())
        if plan_result.get("status", "") != "Success" or len(plan_result.get("position", [])) == 0:
            carb.log_warn("MPLib failed to find path.")
            return None
        
        actions = []

        # Linearly interpolate joint positions over time
        positions = plan_result["position"]
        num_points = len(positions)
        if num_points < 2:
            print("[MPLib] Not enough points to interpolate.")
            return []

        for i in range(num_points - 1):
            start = np.array(positions[i])
            end = np.array(positions[i + 1])
            step_size = 0.1
            num_steps = max(2, min(20, int(np.ceil(np.linalg.norm(end - start) / step_size))))

            for alpha in np.linspace(0, 1, num_steps):
                interp = (1 - alpha) * start + alpha * end
                full_q = np.concatenate([interp, finger_q])
                actions.append(ArticulationAction(joint_positions=full_q))

        self._action_sequence = actions
        self._last_solution = actions[-1].joint_positions if actions else None
        return actions

    def forward(self, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray] = None) -> ArticulationAction:
        if not self._action_sequence:
            self._action_sequence = self._make_new_plan(target_end_effector_position, target_end_effector_orientation)
            if not self._action_sequence:
                return ArticulationAction()

        action = self._action_sequence.pop(0)
        return action

    def reset(self):
        self._action_sequence = []


