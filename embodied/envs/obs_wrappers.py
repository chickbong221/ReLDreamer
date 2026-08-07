import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper


class NonPrivilegedObsWrapper(gym.ObservationWrapper):
    """Remove simulator-only privileged fields from obs['extra']."""

    PRIVILEGED_KEYS = {
        # PickCube-style keys
        'is_grasped', 'goal_pos', 'obj_pose',
        'tcp_to_obj_pos', 'obj_to_goal_pos',
        # MSHAB-style keys
        'obj_pose_wrt_base', 'goal_pos_wrt_base',
    }

    def __init__(self, env) -> None:
        super().__init__(env)
        self._base_env: BaseEnv = env.unwrapped
        init_raw_obs = common.to_tensor(self._base_env._init_raw_obs)
        self._base_env.update_obs_space(self.observation(init_raw_obs))

    def observation(self, obs):
        if 'extra' in obs:
            obs = dict(obs)
            obs['extra'] = {k: v for k, v in obs['extra'].items()
                            if k not in self.PRIVILEGED_KEYS}
        return obs


class NamedCameraRGBWrapper(FlattenRGBDObservationWrapper):
    """Flatten state exactly as upstream does, but keep cameras separable.

    Upstream concatenates every RGB sensor along the channel axis, which loses
    which camera a slice came from. Subclassing rather than reimplementing is
    deliberate: ``state`` comes straight back out of the upstream conversion,
    so its ordering and width cannot drift from the non-graph configurations.

    Sitting below ``ManiSkillVectorEnv`` means ``info['final_observation']`` is
    built from these outputs, so terminal transitions carry the true final
    frames as ordinary top-level tensors.
    """

    def __init__(self, env, camera_keys) -> None:
        # {obs key: camera name}, given explicitly so nothing depends on the
        # order sensor_data happens to iterate in.
        self._camera_keys = dict(camera_keys)
        super().__init__(env, rgb=True, depth=False, state=True)

    def observation(self, obs):
        sensors = obs.get('sensor_data', {})
        missing = [c for c in self._camera_keys.values() if c not in sensors]
        if missing:
            raise KeyError(
                f'cameras {missing} are not rendered; available: '
                f'{sorted(sensors)}')
        # Cloned, not referenced: the vector env stores this dict as
        # final_observation and then re-renders into the sensor buffers on
        # auto-reset, which would overwrite a view.
        frames = {
            key: sensors[cam]['rgb'].clone()
            for key, cam in self._camera_keys.items()
        }
        out = dict(super().observation(obs))
        out.pop('rgb', None)
        out.update(frames)
        return out
