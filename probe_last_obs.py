"""Throwaway: what does the base env actually expose after reset?

Builds the same MS-HAB env the smoke test does, with the graph off so nothing
touches sensor data, then prints the observation plumbing. Delete when done.
"""

from embodied.envs.maniskill import ManiSkill

env = ManiSkill(
    'PickSubtaskTrain-v0',
    num_envs=2,
    obs_mode='rgb+segmentation',
    image_size=112,
    control_mode='pd_joint_delta_pos',
    mshab_task='set_table',
    mshab_split='train',
    mshab_obj='024_bowl',
    mshab_num_build_configs=4,
    max_episode_steps=100,
    num_frames=1,
    frame_stack=1,
    graph=None,
    seed=0,
)

vec = env._env
base = vec.unwrapped
print('\n--- identity ---')
print('vector env   :', type(vec))
print('unwrapped    :', type(base))
print('is base_env  :', base is getattr(vec, 'base_env', None))
print('obs_mode     :', getattr(base, 'obs_mode', '<none>'),
      '|', getattr(base, '_obs_mode', '<none>'))
print('obs struct   :', getattr(base, 'obs_mode_struct', '<none>'))

for name in ('_last_obs', '_init_raw_obs'):
    value = getattr(base, name, None)
    print(f'\n--- base.{name} ---')
    print('type         :', type(value))
    if isinstance(value, dict):
        for key, item in value.items():
            shape = getattr(item, 'shape', None)
            print(f'  {key:14s} {type(item).__name__:12s} '
                  f'{tuple(shape) if shape is not None else list(item)[:6]}')

print('\n--- base.get_obs() ---')
obs = base.get_obs()
if isinstance(obs, dict):
    print('keys         :', list(obs))
    for cam, fields in obs.get('sensor_data', {}).items():
        print(f'  {cam:14s}',
              {k: (tuple(v.shape), str(v.dtype)) for k, v in fields.items()})
else:
    print('type         :', type(obs), getattr(obs, 'shape', ''))

env.close()
