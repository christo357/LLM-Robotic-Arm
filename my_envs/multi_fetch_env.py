# multi_fetch_env.py

import os
import numpy as np
import gymnasium as gym 
import gymnasium_robotics
from gymnasium_robotics.envs.fetch import MujocoFetchEnv
from gymnasium.utils.ezpickle import EzPickle


# GYMROB_FILE = os.path.dirname(gymnasium_robotics.__file__)

current_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(current_dir, '..', 'assets', 'xml')
MODEL_XML_PATH = os.path.join(
    assets_dir, 
    "multi_pick_and_place.xml"
)
if not os.path.exists(MODEL_XML_PATH):
    raise FileNotFoundError(f"❌ Error: Could not find XML at {MODEL_XML_PATH}")


def sample_non_overlapping_xy(
    center_xy, 
    obj_range, 
    n_objects,
    min_dist,
    rng, 
    min_gripper_dist=0.05,
    max_tries = 1000
):
    
    positions = []
    tries = 0
    
    while len(positions)<n_objects and tries < max_tries :
        candidate = center_xy + rng.uniform(-obj_range, obj_range, size=2)
        
        # keep away from gripper
        if np.linalg.norm(candidate - center_xy) < min_gripper_dist:
            tries += 1
            continue
        
        # keep away from other objects
        ok = True
        for p in positions:
            if np.linalg.norm(candidate - p) < min_dist:
                ok = False
                break
                
        
        if ok:
            positions.append(candidate)
        tries += 1
            
    if len(positions) < n_objects:
        raise RuntimeError("Could not sample non-overlapping object positions. Reduce min_dist or increase obj_range.")
    return positions
    
class MultiObjectFetchPickAndPlaceEnv(MujocoFetchEnv, EzPickle):
    def __init__(self, reward_type = 'sparse',n_objects=5,  **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object1:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object2:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object3:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object4:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
            
        }
        MujocoFetchEnv.__init__(
                self,
                model_path = MODEL_XML_PATH, 
                has_object=True, 
                block_gripper=False,
                n_substeps=20,
                gripper_extra_height=0.2,
                target_in_the_air=True,
                target_offset=0.0,
                obj_range=0.15,
                target_range=0.14,
                distance_threshold=0.05,
                initial_qpos=initial_qpos,
                reward_type=reward_type,
                **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, n_objects=n_objects, **kwargs)
        self.n_objects = n_objects
        
    def _reset_sim(self):
        # Reset buffers for joint states, actuators, warm-start, control buffers etc.
        self._mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None
            
        # Randomize start position of object.
        if self.has_object:
            
            center_xy = self.initial_gripper_xpos[:2].copy()
            xy_list = sample_non_overlapping_xy(
                center_xy=center_xy, 
                obj_range=self.obj_range, 
                n_objects=self.n_objects,
                min_dist=self.distance_threshold, 
                rng = self.np_random,
                max_tries=1000
            )
            
            # Table height from object0 default (computed in _env_setup)
            z = float(self.height_offset)
        
            for i, xy in enumerate(xy_list):
                x, y = xy
                joint_name = f"object{i}:joint"
                object_qpos = self._utils.get_joint_qpos(self.model, self.data, joint_name)
                assert object_qpos.shape == (7,)
                object_qpos[0] = x
                object_qpos[1] = y
                object_qpos[2] = z
                # keep orientation as-is (upright), or set quaternion explicitly if you want
                self._utils.set_joint_qpos(
                    self.model, self.data, joint_name, object_qpos 
                )
            for i in range(self.n_objects, 5):
                joint_name = f"object{i}:joint"
                try:
                    object_qpos = self._utils.get_joint_qpos(self.model, self.data, joint_name)
                    object_qpos[0] = 0.0
                    object_qpos[1] = 0.0
                    object_qpos[2] = -10.0   # far below the table
                    self._utils.set_joint_qpos(self.model, self.data, joint_name, object_qpos)
                except KeyError:
                        pass

        # 3) Forward physics
        self._mujoco.mj_forward(self.model, self.data)
        return True
    
    def _viewer_setup(self):
        # Fixed camera configuration
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

        # Fixed look-at point (table center)
        self.viewer.cam.lookat[:] = np.array([1.3, 0.75, 0.55])
            