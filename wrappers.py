import gymnasium as gym
from gymnasium_robotics.utils import rotations

import numpy as np     

class ActiveObjectWrapper(gym.Wrapper):
    """
    Wraps a Fetch-style GoalEnv with multiple objects named object0..object4.

    active_object_index:
        which object index to treat as the task object.
        0 means original object0.
    """
    def __init__(self, env, default_index: int = 0, n_objects: int = 5):
        super().__init__(env)
        self.base = env.unwrapped
        self.n_objects = n_objects
        self.active_object_index = default_index

    def set_active_object(self, idx: int):
        """Change which object is used as 'achieved_goal'."""
        if idx < 0 or idx >= self.n_objects:
            raise ValueError(f"Object index must be within range 0 to {self.n_objects-1}")
        self.active_object_index = idx
        
    def get_current_obs(self):
        """
        Returns the current observation (remapped for the active object).
        Useful for scripted access or debugging.
        """
        # 1. Get base observation (from the unwrapped env)
        raw_base_obs = self.base._get_obs()
        
        # 2. Apply our remapping logic
        # (This ensures we see Object 2, not Object 0)
        final_obs = self._remap_obs(raw_base_obs)
        
        return final_obs

    def _get_active_object_data(self):
        """Helper to get all physics data for the active object"""
        obj_name = f"object{self.active_object_index}"
        
        # 1. Get Real Physics Data
        # Position (x,y,z)
        pos = self.base._utils.get_site_xpos(self.base.model, self.base.data, obj_name)
        # Rotation Matrix (flattened) -> Euler depending on env version, usually 3 items for Fetch
        rot_mat = self.base._utils.get_site_xmat(self.base.model, self.base.data, obj_name)
        rot = rotations.mat2euler(rot_mat.reshape(3, 3))
        
        # Velocities
        dt = self.base.n_substeps * self.base.model.opt.timestep
        velp = self.base._utils.get_site_xvelp(self.base.model, self.base.data, obj_name) * dt
        velr = self.base._utils.get_site_xvelr(self.base.model, self.base.data, obj_name) * dt
        
        return pos, rot, velp, velr

    def _remap_obs(self, obs):
        """
        Surgically replace Object 0 data with Active Object data 
        in the 'observation' vector.
        """
        # 1. Get Active Object Data
        pos, rot, velp, velr = self._get_active_object_data()

        pos = np.asarray(pos, dtype=np.float64)
        rot = np.asarray(rot, dtype=np.float64)
        velp = np.asarray(velp, dtype=np.float64)
        velr = np.asarray(velr, dtype=np.float64)

        assert pos.shape == (3,), f'pos shape: {pos.shape}'
        assert rot.shape == (3,), f'rot shape: {rot.shape}'
        assert velp.shape == (3,)
        assert velr.shape == (3,)
        
        # copy observation vector before editing
        o = obs["observation"].copy()
        
        # 2. Get Gripper Data (Indices 0:3) to compute relative position
        grip_pos = obs['observation'][0:3]
        rel_pos = pos - grip_pos

        # 3. Update 'achieved_goal' (For the Referee/Reward)
        obs['achieved_goal'] = pos.copy()

        # 4. Update 'observation' vector (For the Neural Network)
        # FetchPickAndPlace Observation Structure:
        # [0:3]   Grip Pos
        # [3:6]   Object Pos      <-- REPLACE
        # [6:9]   Rel Pos         <-- REPLACE
        # [9:11]  Gripper State
        # [11:14] Object Rot      <-- REPLACE
        # [14:17] Object VelP     <-- REPLACE
        # [17:20] Object VelR     <-- REPLACE
        
        o[3:6] = pos
        o[6:9] = rel_pos
        o[11:14] = rot
        o[14:17] = velp
        o[17:20] = velr
        
        obs["observation"] = o
        return obs

    def reset(self, *, active_object_id=None, **kwargs):
        if active_object_id is not None:
            self.set_active_object(active_object_id)
            
        obs, info = self.env.reset(**kwargs)
        obs = self._remap_obs(obs)
        return obs, info
        
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        obs = self._remap_obs(obs)
        
        # Recompute reward based on NEW object position
        # (The base env computes reward based on object0, which is wrong now)
        a_g = obs['achieved_goal']
        t_g = obs['desired_goal']
        
        reward = self.base.compute_reward(a_g, t_g, info)
        info["is_success"] = self.base._is_success(a_g, t_g)
        
        return obs, reward, term, trunc, info
    
    


class ManualGoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def set_goal(self, goal):
        g = np.asarray(goal, dtype=np.float64).copy()
        u = self.env.unwrapped
        u.goal = g  # this is what compute_reward/_is_success uses

    def set_goal_relative_to_object(self, object_id: int, offset=(0.0, 0.0, 0.10)):
        u = self.env.unwrapped
        name = f"object{int(object_id)}"
        obj_pos = u._utils.get_site_xpos(u.model, u.data, name).copy()
        g = obj_pos + np.asarray(offset, dtype=np.float64)
        u.goal = g