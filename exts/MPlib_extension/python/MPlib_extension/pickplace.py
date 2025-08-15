from isaacsim.examples.interactive.base_sample import BaseSample
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from MPlib_extension.controller import FrankaMplibController
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Usd, UsdShade, UsdPhysics, Sdf, Gf
from omni.isaac.core.objects import DynamicCuboid
from MPlib_extension.task import FrankaTask
import numpy as np
import asyncio

class PickPlaceTaskRunner(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._controller = None
        self._articulation_controller = None

    def setup_scene(self):
        world = self.get_world()
        scene = world.scene
        stage = world.stage

        # add table
        scene.add(
            DynamicCuboid(
                prim_path="/World/table",
                name="table",
                position=np.array([0.5, 0.0, 0.025]),
                scale=np.array([0.8, 0.8, 0.05]),    
                color=np.array([0.5, 0.3, 0.1]),
                mass=0.0     
            )
        )

        # add franka task
        task = FrankaTask("Plan To Target Task", enable_target_cube=False)
        world.add_task(task)
        self._franka_task = task
        self._franka = task.get_franka_robot()

        # add blocks
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

        def create_physics_material(stage, material_path, static_friction, dynamic_friction, restitution, color):
            # Create a new Material prim at the given path
            material_prim = stage.DefinePrim(material_path, "PhysicsMaterial")

            # physical material properties
            material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
            material_api.CreateStaticFrictionAttr().Set(static_friction)
            material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
            material_api.CreateRestitutionAttr().Set(restitution)

            # visual properties for the blocks (for colors and appearance)
            usd_material = UsdShade.Material.Define(stage, material_path)
            shader_path = material_path + "/Shader"
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

            # bind shader to material
            usd_material.CreateSurfaceOutput().ConnectToSource(shader_output)

            return material_prim

        def bind_material_to_prim(stage, prim, material_path):
            material_prim = stage.GetPrimAtPath(material_path)
            if not material_prim:
                print(f"[ERROR] Material {material_path} does not exist.")
                return
            material = UsdShade.Material(material_prim)
            UsdShade.MaterialBindingAPI(prim).Bind(material)   

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
                    mass=0.04
                )
            )

            material_prim_path = f"/World/Materials/HighFrictionBlock_{i+1}"
            if not stage.GetPrimAtPath(material_prim_path):
                create_physics_material(
                    stage,
                    material_path=material_prim_path,
                    static_friction=200.0,
                    dynamic_friction=10.0,
                    restitution=0.0,
                    color=block_colors[i]
                )
    
            cube_prim = world.stage.GetPrimAtPath(block.prim_path)
            bind_material_to_prim(stage, cube_prim, material_prim_path)
            # physx_api.CreateDisableGravityAttr(False)

            self.blocks.append(block)     
        
    async def setup_pre_reset(self):
        self._controller.reset()
        return

    def world_cleanup(self):
        self._controller = None
        return

    async def setup_post_load(self):
        self._franka_task = list(self._world.get_current_tasks().values())[0]
        self._task_params = self._franka_task.get_params()

        self._robot = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        self._robot.disable_gravity()
        self._articulation_controller = self._robot.get_articulation_controller()
        if self._articulation_controller is None:
            print("Waiting for articulation controller to initialize...")
            await asyncio.sleep(0.1)
            self._articulation_controller = self._robot.get_articulation_controller()

        self._articulation_controller.switch_control_mode("position")
        self._controller = FrankaMplibController("mplib_controller", self._robot, physics_dt=1 / 60.0)

        kps, kds = self._franka_task.get_custom_gains()
        self._articulation_controller.set_gains(kps=kps, kds=kds)

        self._finger_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        self._finger_indices = [self._robot.dof_names.index(j) for j in self._finger_joint_names]

        if not self._world.physics_callback_exists("sim_step"):
            self._world.add_physics_callback("sim_step", self._on_sim_step)
        return

    async def test_mplib_plan_and_execute(self):
        # Test MPLib motion planning and execution by clicking test button (test this on first run)
        print("[TEST] Starting MPLib motion test...")
        # 1. Ensure robot is ready
        self._robot.set_joint_positions(np.array([0.0, -0.4, 0.0, -1.0, 0.0, 1.0, 0.7, 0.04, 0.04]))
        await asyncio.sleep(0.3)

        # 2. Build controller
        controller = FrankaMplibController("test_mplib", self._robot)
        self._articulation_controller.switch_control_mode("position")

        # 3. Set target pose
        target_pos = np.array([0.5, 0.0, 0.3])
        target_orn = euler_angles_to_quat(np.array([np.pi, 0, 0]))

        # 4. Call plan function
        actions = controller._make_new_plan(target_pos, target_orn)

        if not actions:
            print("[ERROR] MPLib failed to generate a plan.")
            return

        # 5. Execute plan
        for i, action in enumerate(actions):
            if action.joint_positions is None:
                print(f"[ERROR] Action {i} has no joint positions.")
                continue
            self._articulation_controller.apply_action(action)
            await asyncio.sleep(1.0 / 60.0)
            
        print("[TEST] MPLib motion test complete.")

    async def _on_follow_target_event_async(self):
        world = self.get_world()
        self._pass_world_state_to_controller()
        await world.play_async()

        if not world.physics_callback_exists("sim_step"):
            world.add_physics_callback("sim_step", self._on_follow_target_simulation_step)

        for i, block in enumerate(self.blocks):
            print(f"[INFO] Planning pick and place for block {i+1}")
            
            block_pos, _ = block.get_world_pose()
            block_height = block.get_local_scale()[2]

            grasp_z_offset = 0.25 * block_height + 0.105
            grasp = block_pos + np.array([0.0, 0.0, grasp_z_offset])
            above = grasp + np.array([0, 0, 0.15])
            place = block_pos + np.array([0.05, 0.0, grasp_z_offset ])
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
                    joint_positions[idx] = 0.0
                self._articulation_controller.apply_action(ArticulationAction(joint_positions=joint_positions))
                await asyncio.sleep(0.5)

                # --- 5. Lift back to hover ---
                self._controller.reset()
                closed_fingers = [0.001, 0.001]
                self._controller._make_new_plan(above, orientation, finger_override=closed_fingers)
                while self._controller._action_sequence:
                    action = self._controller.forward(above, orientation)
                    self._articulation_controller.apply_action(action)
                    await asyncio.sleep(1.0 / 40.0)

                # --- 6. Move above placement ---
                self._controller.reset()
                closed_fingers = [0.001, 0.001]
                self._controller._make_new_plan(place_above, orientation, finger_override=closed_fingers)
                while self._controller._action_sequence:
                    action = self._controller.forward(place_above, orientation)
                    self._articulation_controller.apply_action(action)
                    await asyncio.sleep(1.0 / 40.0)

                # --- 7. Move down to place ---
                self._controller.reset()
                closed_fingers = [0.001, 0.001]
                self._controller._make_new_plan(place, orientation, finger_override=closed_fingers)
                while self._controller._action_sequence:
                    action = self._controller.forward(place, orientation)
                    self._articulation_controller.apply_action(action)
                    await asyncio.sleep(1.0 / 40.0)

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

    def _on_sim_step(self, step_size):
        if self._controller is None:
            kps, kds = self._franka_task.get_custom_gains()
            self._articulation_controller.set_gains(kps, kds)
        return

    def _pass_world_state_to_controller(self):
        self._controller.reset()
        for wall in self._franka_task.get_obstacles():
            self._controller.add_obstacle(wall)
        return

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
        self.get_world().get_current_tasks().values().__iter__().__next__().add_obstacle()
        return

    def _on_remove_wall_event(self):
        task = self.get_world().get_current_tasks().values().__iter__().__next__()
        task.remove_obstacle()
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
