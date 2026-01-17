""" 
export RESUME_ID=08vger36  
python ppo_dense.py
"""
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo

import time
import os
import glob
import numpy as np
from collections import deque


import wandb
from wandb.integration.sb3 import WandbCallback
from sb3_contrib import TQC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import  HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.results_plotter import load_results, ts2xy

gym.register_envs(gymnasium_robotics)


# Allow user to resume training by setting RESUME_ID in environment or command line
RESUME_ID = os.getenv("RESUME_ID", None)



NUM_ENV = 5
GLOBAL_SEED = 42
LOG_DIR = 'logs/'
CHECK_FREQ = 500
set_random_seed(GLOBAL_SEED)

if RESUME_ID is not None:
    RUN_LOGDIR = os.path.join(LOG_DIR, RESUME_ID)
    MODEL_PATH = os.path.join(RUN_LOGDIR, "best_model.zip")
    VECNORM_PATH = os.path.join(RUN_LOGDIR, "vecnormalize.pkl")


class SuccessRateCallback(BaseCallback):
    def __init__(self, window_size=100, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.success_buffer = deque(maxlen = self.window_size)
        
        
    def _on_step(self):
        infos = self.locals['infos']
        for info in infos:
            # FetchReach-v4 includes "is_success" in each env's info
            if "is_success" in info:
                self.success_buffer.append(info["is_success"])
    
            
        if len(self.success_buffer)>0:
            success_rate = np.mean(self.success_buffer)
            wandb.log(
                {
                    "rollout/success_rate": success_rate,
                    "time/steps": self.num_timesteps,
                }
            )
        return True
    
class SaveCheckPointCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=0, ):

        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            
            
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model at {} timesteps".format(x[-1]))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)
                    
                    self.model.get_vec_normalize_env().save(os.path.join(self.log_dir, "vecnormalize.pkl"))

        return True
    
# --- NEW CALLBACK FOR WANDB UPLOAD ---
class WandbCleanupVideoCallback(BaseCallback):
    def __init__(self, run_id, verbose=0):
        super().__init__(verbose)
        self.run_id = run_id
        self.video_dir = os.path.join(LOG_DIR, self.run_id, "videos")
        self.uploaded_videos = set()

    def _on_step(self):
        # Check for new videos every 500 steps to save performance
        if self.num_timesteps % 500 == 0 and os.path.exists(self.video_dir):
            # Find all mp4 files
            mp4_files = glob.glob(os.path.join(self.video_dir, "*.mp4"))
            
            for mp4_path in mp4_files:
                filename = os.path.basename(mp4_path)
                
                if filename not in self.uploaded_videos:
                    try:
                        # 1. Upload to WandB
                        wandb.log({"rollout/video": wandb.Video(mp4_path, fps=30, format="mp4")})
                        self.uploaded_videos.add(filename)
                        
                        if self.verbose > 0:
                            print(f"Uploaded video: {filename}")
                            
                        # 2. OPTIONAL: Delete from local disk to save space
                        # os.remove(mp4_path) 
                        
                    except Exception as e:
                        print(f"Failed to upload video {filename}: {e}")
                        
        return True
    
# class EarlyStopCallback(BaseCallback):
#     def __init__(self, threshold, window_size=100, verbose=0):
#         super().__init__(verbose)
#         self.threshold = threshold
#         self.window_size = window_size
#         self.buffer = deque(maxlen=self.window_size)
        
#     def _on_step(self,):
#         infos = self.locals['infos']
#         for info in infos:
#             if 'is_success' in info:
#                 self.buffer.append(info['is_success'])
        
#         buffer_mean = float(np.mean(self.buffer))
#         if self.verbose>0:
#             print(f'Current success rate over last {self.window_size} episodes : {buffer_mean:.2f} ')
#         if buffer_mean > self.threshold:
#             if self.verbose > 0:
#                 print(f"Stopping training. Target success {self.threshold} reached.")
#             return False # tell sb3 to stop
#         else: 
#             return True
# class EarlyStopOnReturn(BaseCallback):
#     def __init__(self, reward_threshold, window_size=20, verbose=0):
#         super().__init__(verbose)
#         self.reward_threshold = reward_threshold
#         self.window_size = window_size
#         self.ret_buffer = deque(maxlen=window_size)

#     def _on_rollout_end(self) -> bool:
#         # SB3 stores episode reward in infos via Monitor wrapper
#         ep_info = self.model.rollout_buffer.ep_info_buffer
#         if len(ep_info) > 0:
#             rews = [ep["r"] for ep in ep_info]
#             self.ret_buffer.extend(rews)

#         if len(self.ret_buffer) == self.window_size:
#             mean_return = np.mean(self.ret_buffer)
#             if self.verbose:
#                 print(f"[ES] Avg Return {mean_return:.2f} (target {self.reward_threshold})")
#             if mean_return >= self.reward_threshold:
#                 if self.verbose:
#                     print("Stopping: mean return threshold reached")
#                 return False
            
#         return True
    
#     def _on_step(self):
#         pass




def make_env(id, rank, seed = 0, run_id='0', logdir='logs/', check_freq=500):
    def _init():
        env = gym.make(id, render_mode="rgb_array")
        log_dir = f"{logdir}/{run_id}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, f"env_{rank}.monitor.csv"))  
        # 3. Add Video Recorder ONLY for rank 0 (saves disk space)
        if rank == 0:
            video_folder = os.path.join(log_dir, "videos")
            os.makedirs(log_dir, exist_ok=True)
            env = RecordVideo(
                env, 
                video_folder=video_folder,
                # Record episode 0, then every 500th episode (0, 500, 1000...)
                episode_trigger=lambda x: x % check_freq == 0,
                name_prefix=f"agent-video", 
                
            )
        
        # env.action_space.seed(seed + rank)
        # env.observation_space.seed(seed + rank)
        
        return env

    return _init



wandb_config = dict(
    # env_id = "FetchReach-v4", 
    env_id = 'FetchPickAndPlaceDense-v4', 
    algo = 'TQC-HER', 
    policy_type = 'MultiInputPolicy', 
    NUM_ENV = NUM_ENV, 
    LR = 3e-4, 
    N_STEPS = 512, 
    TOTAL_TIMESTEPS=1_000_000, 
    BATCH_SIZE=512, 
    POLICY_SIZE=256, 
    GAMMA=0.95, 
    
    TAU=0.05,
    learning_starts=1000,
    # REWARD_THRESHOLD = -2,
    
)

if RESUME_ID:
    run = wandb.init(
        project='RL_diary',
        id=RESUME_ID,
        resume="allow",
        name = f"{wandb_config['algo']}_{wandb_config['env_id']}_{wandb_config['LR']}_size{wandb_config['POLICY_SIZE']}",
        config=wandb_config,
        sync_tensorboard=True,
        save_code=True,
        monitor_gym=False,
        notes='Resuming training run',
    )
    ACTIVE_RUN_ID = RESUME_ID
else:
    run = wandb.init(
        project='RL_diary',
        config=wandb_config,
        name=f"{wandb_config['algo']}_{wandb_config['env_id']}_{wandb_config['LR']}_size{wandb_config['POLICY_SIZE']}",
        sync_tensorboard=True,
        save_code=True,
        monitor_gym=False,
        notes='Initial phase : Dense Reward',
    )
    ACTIVE_RUN_ID = run.id

train_env = DummyVecEnv([make_env(wandb_config['env_id'], rank=id, seed=GLOBAL_SEED, run_id=ACTIVE_RUN_ID, logdir=LOG_DIR, check_freq=CHECK_FREQ) for id in range(NUM_ENV)])
train_env = VecNormalize(train_env, norm_obs=True, )

if RESUME_ID:
    vec_path = os.path.join(LOG_DIR, ACTIVE_RUN_ID, "vecnormalize.pkl")
    if os.path.exists(vec_path):
        train_env = VecNormalize.load(vec_path, train_env)
        train_env.training = True

# max_ep_len = 50
# learning_starts = max_ep_len * NUM_ENV * 2
model_path = os.path.join(LOG_DIR, ACTIVE_RUN_ID, "best_model.zip")
if RESUME_ID and os.path.exists(model_path):
    model = TQC.load(model_path, env=train_env)
    model.set_env(train_env)
else:
    model = TQC(
        policy='MultiInputPolicy',
        env=train_env,
        learning_rate=wandb_config['LR'],
        gamma=wandb_config['GAMMA'], 
        tau=wandb_config['TAU'],
        replay_buffer_class=HerReplayBuffer, 
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,      # number of HER replays
            goal_selection_strategy="future"  # from SB3: future, episode, random
        ), 
        n_steps=wandb_config['N_STEPS'],
        
        tensorboard_log=f"runs/{ACTIVE_RUN_ID}",
        batch_size=wandb_config['BATCH_SIZE'],
        policy_kwargs=dict(
            net_arch=[512, 512, 512], 
            n_critics=2
        ),
        
        learning_starts=wandb_config['learning_starts'],
    )




RUN_LOGDIR = os.path.join(LOG_DIR, ACTIVE_RUN_ID)
wandb_cb = WandbCleanupVideoCallback(ACTIVE_RUN_ID)
    
save_checkpoint_cb = SaveCheckPointCallback(check_freq=CHECK_FREQ, log_dir = RUN_LOGDIR)
    
# earlystop_cb = EarlyStopOnReturn(reward_threshold=wandb_config['REWARD_THRESHOLD'], window_size=20 )
    
callbacklist = CallbackList([SuccessRateCallback(), save_checkpoint_cb, wandb_cb, ])

model.learn(
    total_timesteps=wandb_config['TOTAL_TIMESTEPS'], 
    callback=callbacklist, 
    progress_bar=True, 
    reset_num_timesteps=not bool(RESUME_ID)
)


run.finish()
train_env.close()
            


