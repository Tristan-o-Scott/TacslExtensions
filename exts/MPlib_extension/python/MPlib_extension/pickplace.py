from isaacsim.examples.interactive.base_sample import BaseSample
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.types import ArticulationAction
from MPlib_extension.controller import FrankaRrtController
from omni.isaac.core.objects import DynamicCuboid
from MPlib_extension.task import FrankaTask
from pxr import PhysxSchema
import numpy as np
import asyncio


class MyTaskRunner(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._controller = None
        self._articulation_controller = None

    def setup_scene(self):
        world = self.get_world()
        scene = world.scene

        table = scene.add(
            DynamicCuboid(
                prim_path="/World/table",
                name="table",
                position=np.array([0.5, 0.0, 0.0]),
                scale=np.array([0.8, 0.8, 0.05]),    
                color=np.array([0.5, 0.3, 0.1]),
                mass=0.0     
            )
        )

        task = FrankaTask("Plan To Target Task", enable_target_cube=False)
        world.add_task(task)

        self._franka_task = task

        self.blocks = []
        block_positions = [
            np.array([0.4, 0.3, 0.06]),
            np.array([0.2, -0.3, 0.04]),
            np.array([0.6, 0.1, 0.07]),
        ]
        block_colors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        block_heights = [0.12, 0.08, 0.14]
        for i in range(3):
            height = block_heights[i]
            pos = block_positions[i]
            pos[2] = 0.5 * height
            block = scene.add(
                DynamicCuboid(
                    prim_path=f"/World/cube_{i+1}",
                    name=f"cube_{i+1}",
                    position=pos,
                    scale=np.array([0.04, 0.04, height]),
                    color=block_colors[i],
                    mass=0.1,
                )
            )
            self.blocks.append(block)

            cube_prim = world.stage.GetPrimAtPath(block.prim_path)
            physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(cube_prim)
            #physx_api.CreateDisableGravityAttr(True)
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        self._controller.reset()
        return

    def world_cleanup(self):
        self._controller = None
        return

    async def setup_post_load(self):
        self._franka_task = list(self._world.get_current_tasks().values())[0]
        self._task_params = self._franka_task.get_params()

        my_franka = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        self._robot = my_franka
        my_franka.disable_gravity()

        self._controller = FrankaRrtController(name="franka_rrt_controller", robot_articulation=my_franka)
        self._articulation_controller = my_franka.get_articulation_controller()
        
        self._finger_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        self._finger_indices = [self._robot.dof_names.index(j) for j in self._finger_joint_names]

        return

    async def _on_follow_target_event_async(self):
        world = self.get_world()
        self._pass_world_state_to_controller()
        await world.play_async()

        if not world.physics_callback_exists("sim_step"):
            world.add_physics_callback("sim_step", self._on_follow_target_simulation_step)

        for i, block in enumerate(self.blocks):
            print(f"[INFO] Planning pick and place for block {i+1}")
            
            block_pos, _ = block.get_world_pose()

            # Compute grasping and placement poses
            block_scale = block.get_local_scale()
            block_height = block_scale[2]

            grasp_z_offset = 0.25 * block_height + 0.01
            grasp = block_pos + np.array([0, 0, grasp_z_offset])
            above = grasp + np.array([0, 0, 0.15])

            place = block_pos + np.array([0.05, 0.0, grasp_z_offset])
            place_above = place + np.array([0, 0, 0.10])

            orientation = euler_angles_to_quat(np.array([np.pi, 0, 0]))

            try:
                # These steps are made to replicate the demo for MPlib and franka_osc_mplib for isaac gym
                # --- 1. Move above block ---
                self._controller.reset()
                self._controller._make_new_plan(above, orientation)
                while self._controller._action_sequence:
                    action = self._controller.forward(above, orientation)
                    self._articulation_controller.apply_action(action)
                    await asyncio.sleep(1.0 / 60.0)

                # --- 2. Open gripper ---
                print(f"[INFO] Opening gripper for block {i+1}")
                joint_positions = list(self._robot.get_joint_positions())
                for idx in self._finger_indices:
                    joint_positions[idx] = 0.04
                self._articulation_controller.apply_action(ArticulationAction(joint_positions=joint_positions))
                await asyncio.sleep(0.5)

                # --- 3. Move down to grasp ---
                self._controller.reset()
                self._controller._make_new_plan(grasp, orientation)
                while self._controller._action_sequence:
                    action = self._controller.forward(grasp, orientation)
                    self._articulation_controller.apply_action(action)
                    await asyncio.sleep(1.0 / 60.0)

                # --- 4. Close gripper ---
                print(f"[INFO] Grasping block {i+1}")
                joint_positions = list(self._robot.get_joint_positions())
                for idx in self._finger_indices:
                    joint_positions[idx] = 0.001
                self._articulation_controller.apply_action(ArticulationAction(joint_positions=joint_positions))
                await asyncio.sleep(0.5)

                # --- 5. Lift back to hover ---
                self._controller.reset()
                self._controller._make_new_plan(above, orientation)
                while self._controller._action_sequence:
                    action = self._controller.forward(above, orientation)
                    self._articulation_controller.apply_action(action)
                    await asyncio.sleep(1.0 / 60.0)

                # --- 6. Move above placement ---
                self._controller.reset()
                self._controller._make_new_plan(place_above, orientation)
                while self._controller._action_sequence:
                    action = self._controller.forward(place_above, orientation)
                    self._articulation_controller.apply_action(action)
                    await asyncio.sleep(1.0 / 60.0)

                # --- 7. Move down to place ---
                self._controller.reset()
                self._controller._make_new_plan(place, orientation)
                while self._controller._action_sequence:
                    action = self._controller.forward(place, orientation)
                    self._articulation_controller.apply_action(action)
                    await asyncio.sleep(1.0 / 60.0)

                # --- 8. Open gripper to release ---
                print(f"[INFO] Releasing block {i+1}")
                joint_positions = list(self._robot.get_joint_positions())
                for idx in self._finger_indices:
                    joint_positions[idx] = 0.04
                self._articulation_controller.apply_action(ArticulationAction(joint_positions=joint_positions))
                await asyncio.sleep(0.5)

                # --- 9. Lift back up ---
                self._controller.reset()
                self._controller._make_new_plan(place_above, orientation)
                while self._controller._action_sequence:
                    action = self._controller.forward(place_above, orientation)
                    self._articulation_controller.apply_action(action)
                    await asyncio.sleep(1.0 / 60.0)

            except Exception as e:
                print(f"[ERROR] Failed to pick and place block {i+1}: {e}")
                continue

    def _pass_world_state_to_controller(self):
        self._controller.reset()
        for wall in self._franka_task.get_obstacles():
            self._controller.add_obstacle(wall)

    def _on_follow_target_simulation_step(self, step_size):
        observations = self._world.get_observations()
        actions = self._controller.forward(
            target_end_effector_position=observations[self._task_params["target_name"]["value"]]["position"],
            target_end_effector_orientation=observations[self._task_params["target_name"]["value"]]["orientation"],
        )
        kps, kds = self._franka_task.get_custom_gains()
        self._articulation_controller.set_gains(kps, kds)
        self._articulation_controller.apply_action(actions)
        return

    def _on_add_wall_event(self):
        world = self.get_world()
        current_task = list(world.get_current_tasks().values())[0]
        cube = current_task.add_obstacle()
        return

    def _on_remove_wall_event(self):
        world = self.get_world()
        current_task = list(world.get_current_tasks().values())[0]
        obstacle_to_delete = current_task.get_obstacle_to_delete()
        current_task.remove_obstacle()
        return

    def _on_logging_event(self, val):
        world = self.get_world()
        data_logger = world.get_data_logger()
        if not world.get_data_logger().is_started():
            robot_name = self._task_params["robot_name"]["value"]
            target_name = self._task_params["target_name"]["value"]

            def frame_logging_func(tasks, scene):
                return {
                    "joint_positions": scene.get_object(robot_name).get_joint_positions().tolist(),
                    "applied_joint_positions": scene.get_object(robot_name)
                    .get_applied_action()
                    .joint_positions.tolist(),
                    "target_position": scene.get_object(target_name).get_world_pose()[0].tolist(),
                }

            data_logger.add_data_frame_logging_func(frame_logging_func)
        if val:
            data_logger.start()
        else:
            data_logger.pause()
        return

    def _on_save_data_event(self, log_path):
        world = self.get_world()
        data_logger = world.get_data_logger()
        data_logger.save(log_path=log_path)
        data_logger.reset()
        return
