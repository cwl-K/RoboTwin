from envs._base_task import Base_Task
from envs.beat_block_hammer import beat_block_hammer
from envs.utils import *
import sapien

class gpt_beat_block_hammer(beat_block_hammer):
    def play_once(self):
        # Observation: initial scene state
        self.save_camera_images(task_name="beat_block_hammer", step_name="step0_initial_scene", generate_num_id="generate_num_0")
        
        # Get the block's position to decide which arm to use
        block_pose = self.block.get_pose()
        block_x = block_pose.p[0]  # x coordinate
        
        # Select arm based on block's x coordinate
        if block_x > 0:
            arm_tag = ArmTag("right")
        else:
            arm_tag = ArmTag("left")
        
        # Step 1: Grasp the hammer
        # The hammer is at a fixed position on the table
        self.move(
            self.grasp_actor(
                actor=self.hammer,
                arm_tag=arm_tag,
                contact_point_id=0,  # Grasp the hammer handle (contact point 0)
                pre_grasp_dis=0.1,
                grasp_dis=0
            )
        )
        
        # Observation: after grasping the hammer
        self.save_camera_images(task_name="beat_block_hammer", step_name="step1_grasp_hammer", generate_num_id="generate_num_0")
        
        # Lift the hammer slightly after grasping to avoid collision
        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=0.07,
                move_axis='world'
            )
        )
        
        # Observation: after lifting hammer
        self.save_camera_images(task_name="beat_block_hammer", step_name="step2_lift_hammer", generate_num_id="generate_num_0")
        
        # Step 2: Beat the block by placing the hammer's contact point on the block's functional point
        # Get the block's top functional point (id=1) as the target for beating
        target_pose = self.block.get_functional_point(1, "pose")
        
        # Place the hammer so that its head (functional point 0) aligns with the block's top
        self.move(
            self.place_actor(
                actor=self.hammer,
                arm_tag=arm_tag,
                target_pose=target_pose,
                functional_point_id=0,  # Align hammer head with block's top
                pre_dis=0.1,
                dis=0.02,
                is_open=False,  # Keep gripper closed to maintain hold of the hammer
                constrain="free",
                pre_dis_axis='fp'
            )
        )
        
        # Observation: after beating the block
        self.save_camera_images(task_name="beat_block_hammer", step_name="step3_beat_block", generate_num_id="generate_num_0")
        
        # Note: No need to lift the hammer after beating, no need to open gripper or return arm to origin

'''
Observation Point Analysis:
1. Initial scene state before any actions
2. Grasp the hammer (arm moves to hammer, gripper closes)
3. Lift hammer slightly after grasping
4. Beat the block (hammer head placed on block's functional point)
5. Final scene state after beating
'''
