from isaacsim.examples.interactive.base_sample import BaseSample
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from MPlib_extension.controller import FrankaMplibController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.objects import DynamicCuboid
from pxr import UsdShade, UsdPhysics, Sdf, Gf
from omni.isaac.franka import Franka
import numpy as np
import asyncio

class MassThrowTaskRunner(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._controller = None
        self._articulation_controller = None
        self._gripper_closed = np.array([0.0115, 0.0115])
        self._gripper_open = np.array([0.04, 0.04])
        self._ready_to_throw = False
        self._finger_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        self._finger_indices = None
        self._franka = None
        self._init_franka_q = None
        self._init_block_poses = [] 
        self.blocks = []
        self.block_masses = []

    def setup_scene(self):
        world = self.get_world()
        scene = world.scene
        stage = world.stage
        scene.add_default_ground_plane()

        # add franka arm
        self._franka = scene.add(Franka(prim_path="/World/Franka", name="Franka"))
        self._init_franka_q = np.array([0.0, -0.6, 0.0, -2.4, 0.0, 1.8, 0.8] + self._gripper_open.tolist())

        # add a target zone
        self.target_zone = scene.add(
            DynamicCuboid(
                prim_path="/World/target_zone",
                name="target_zone",
                position=np.array([0.8, 0.0, 0.01]),
                scale=np.array([0.25, 1, 0.01]),
                color=np.array([1.0, 1.0, 1.0]),
                mass=0.0
            )
        )

        # add blocks
        block_positions = [
            np.array([0.4, -0.3, 0.02]),
            np.array([0.4, 0.0, 0.02]),
            np.array([0.4, 0.3, 0.02]),
        ]
        block_colors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        block_masses = [0.05, 0.08, 0.1] # kg

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

        for i in range(len(block_masses)):
            mass = block_masses[i]
            block = scene.add(
                DynamicCuboid(
                    prim_path=f"/World/block_{i}",
                    name=f"block_{i}",
                    position=block_positions[i],
                    scale=np.array([0.04, 0.04, 0.04]),
                    color=block_colors[i],
                    mass=mass
                )
            )

            material_prim_path = f"/World/Materials/HighFrictionBlock_{i+1}"
            if not stage.GetPrimAtPath(material_prim_path):
                create_physics_material(
                    stage,
                    material_path=material_prim_path,
                    static_friction=2.5,
                    dynamic_friction=1.0,
                    restitution=0.0,
                    color=block_colors[i]
                )
    
            cube_prim = world.stage.GetPrimAtPath(block.prim_path)
            bind_material_to_prim(stage, cube_prim, material_prim_path)

            self.blocks.append(block) 
            self.block_masses.append(mass)    

        self._init_block_poses = [blk.get_world_pose() for blk in self.blocks]


    async def setup_pre_reset(self):
        self._controller.reset()
        return        

    async def setup_post_reset(self):
        world = self.get_world()
        await world.reset_async() 
        await asyncio.sleep(0.1)

        if not hasattr(self, "_init_franka_q"):
            self._init_franka_q = None
        if not hasattr(self, "_init_block_poses"):
            self._init_block_poses = []

        if self._controller:
            self._controller.reset()
            self._controller.set_fast_mode(False)

        # reset Franka to initial pose
        if self._franka and self._init_franka_q is not None:
            self._articulation_controller.apply_action(
                ArticulationAction(joint_positions=self._init_franka_q.copy())
            )
            await asyncio.sleep(0.05)
            self._franka.set_joint_velocities(np.zeros_like(self._franka.get_joint_velocities()))

        if self._init_franka_q is None and self._franka is not None:
            self._init_franka_q = np.array([0.0, -0.6, 0.0, -2.4, 0.0, 1.8, 0.8] + self._gripper_open.tolist())

        if not self._init_block_poses and self.blocks:
            self._init_block_poses = [blk.get_world_pose() for blk in self.blocks if blk is not None]

        # reset each block
        for block in self.blocks:
            block.set_linear_velocity(np.zeros(3))
            block.set_angular_velocity(np.zeros(3))

        await asyncio.sleep(0.1)

    def world_cleanup(self):
        self._controller = None
        return
    
    def check_if_block_in_target(self, block):
        block_pos, _ = block.get_world_pose()
        target_pos, _ = self.target_zone.get_world_pose()
        target_extent = self.target_zone.get_local_scale() * 0.5

        dx = abs(block_pos[0] - target_pos[0])
        dy = abs(block_pos[1] - target_pos[1])

        if dx < target_extent[0] and dy < target_extent[1]:
            print(f"[SUCCESS] {block.name} landed in target.")
            return True
        else:
            print(f"[MISS] {block.name} missed the target.")
            return False

    async def setup_post_load(self):
        self._articulation_controller = self._franka.get_articulation_controller()
        self._controller = FrankaMplibController("mplib_controller", self._franka)
        self._finger_indices = [self._franka.dof_names.index(j) for j in self._finger_joint_names]

        # refresh valid cube references to avoid errors
        scene = self.get_world().scene
        self.blocks = [
            b for b in (scene.get_object(f"block_{i}") for i in range(len(self.block_masses)))
            if b is not None
        ]

    async def _on_follow_target_event_async(self):
        await self.throw_mass()

    async def throw_mass(self):
        if not self.blocks:
            print("[ERROR] No blocks found to throw.")
            return

        for i, block in enumerate(self.blocks):
            if block is None:
                print(f"[WARN] Block {i} not found in scene.")
                continue

            mass = self.block_masses[i]
            print(f"\n[INFO] Throwing block {i} (mass={mass})")

            pos, _ = block.get_world_pose()
            height = block.get_local_scale()[2]

            grasp = pos + np.array([0.0, 0.0, 0.5 * height + 0.09])
            above = grasp + np.array([0.0, 0.0, 0.1])
            orientation = euler_angles_to_quat(np.array([np.pi, 0, 0]))

            pre_throw = above
            max_throw_direction = np.array([0.32, 0.0, 0.17]) 

            target_pos, _ = self.target_zone.get_world_pose()
            target_x = target_pos[0]  # how far target is from origin

            target_scale = (target_x / 1.0) ** 2

            k = 0.5 # scaling factor for mass influence on throw strength
            m_ref = 0.1 # reference mass
            mass_factor_raw = (m_ref / mass) ** 0.5 

            mass_factor = 1.0 + k * (mass_factor_raw - 1.0)

            # make caps to prevent extremely weak or strong throws
            mass_factor = np.clip(mass_factor, 1.0, 1.8)

            mass_scale = mass_factor * target_scale *  1.18

            throw_direction = mass_scale * max_throw_direction
            if np.linalg.norm(throw_direction) > np.linalg.norm(max_throw_direction):
                throw_direction = throw_direction / np.linalg.norm(throw_direction) * np.linalg.norm(max_throw_direction)
            release_pose = pre_throw + throw_direction

            # --- 1. grasp block ---
            await self._follow_plan(above, orientation, self._gripper_open)
            await self._follow_plan(grasp, orientation, self._gripper_open)

            current_joint_positions = self._franka.get_joint_positions()

            current_joint_positions[self._finger_indices[0]] = self._gripper_closed[0]
            current_joint_positions[self._finger_indices[1]] = self._gripper_closed[1]

            self._articulation_controller.apply_action(
                ArticulationAction(joint_positions=current_joint_positions)
            )

            await asyncio.sleep(0.8)
            await self._follow_plan(above, orientation, self._gripper_closed)
            await asyncio.sleep(0.2)

            # --- 2. plan a single motion from pre_throw to release_pose ---
            self._controller.reset()
            self._controller.set_fast_mode(True)
            self._controller._make_new_plan(release_pose, orientation, finger_override=self._gripper_closed)
            
            if not self._controller._action_sequence:
                print("[ERROR] Failed to generate throw motion.")
                return
            
            release_index = int(0.5 * len(self._controller._action_sequence))

            # --- 3. execute motion quickly and open gripper during action ---
            for i, action in enumerate(self._controller._action_sequence):
                q = action.joint_positions.copy()
                if i == release_index:
                    q[-2:] = self._gripper_open
                self._articulation_controller.apply_action(
                    ArticulationAction(joint_positions=q)
                )
                await asyncio.sleep(1.0 / 140.0)

        self._controller.set_fast_mode(False)
        await self._follow_plan(above, orientation, self._gripper_open)
        await asyncio.sleep(0.5)
        for block in self.blocks:
            self.check_if_block_in_target(block)
        await asyncio.sleep(1.0)

    async def _follow_plan(self, pos, orn, fingers):
        self._controller.reset()
        self._controller._make_new_plan(pos, orn, finger_override=fingers)
        while self._controller._action_sequence:
            action = self._controller.forward(pos, orn)
            self._articulation_controller.apply_action(action)
            await asyncio.sleep(1.0 / 60.0)
    
    def _on_logging_event(self, val):
        world = self.get_world()
        data_logger = world.get_data_logger()
        if not world.get_data_logger().is_started():
            robot_name = self._task_params["robot_name"]["value"]
            target_name = self._task_params["target_name"]["value"]

            def frame_logging_func(scene):
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