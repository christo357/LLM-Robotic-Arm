from gymnasium.envs.registration import register

register(
    id="MultiObjectFetchPickAndPlace-v0",
    entry_point="my_envs.multi_fetch_env:MultiObjectFetchPickAndPlaceEnv",
    max_episode_steps=50,  # optional
)