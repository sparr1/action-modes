import numpy as np
import mujoco
from gymnasium.spaces import Box
from .ant_variable_legs import AntVariableLegsEnv


class AntLegAdaptationEnv(AntVariableLegsEnv):
    """
    Ant morphology adaptation environment.

    Starts with `start_legs` evenly-spaced legs. After
    `total_timesteps * switch_fraction` environment steps the model is
    rebuilt with `end_legs` evenly-spaced legs at the next episode reset.

    Observation and action spaces are fixed to `start_legs` dimensions
    throughout so the RL network never changes shape:
      - Actions: trailing dims beyond end_legs*2 are ignored post-switch
      - Observations: missing dims are zero-padded post-switch

    Parameters
    ----------
    total_timesteps : int
        Total training steps (used to compute the switch point).
    start_legs : int
        Number of legs during the first phase.
    end_legs : int
        Number of legs during the second phase.
    switch_fraction : float
        Fraction of total_timesteps at which to trigger the switch (default 0.5).
    xml_file : str
        Output XML filename written into assets/. Change this if running
        multiple adaptation envs simultaneously to avoid file collisions.
    """

    def __init__(
        self,
        total_timesteps,
        start_legs=4,
        end_legs=3,
        switch_fraction=0.5,
        xml_file="ant_adaptation.xml",
        **kwargs,
    ):
        self._total_timesteps = total_timesteps
        self._start_legs = start_legs
        self._end_legs = end_legs
        self._switch_step = int(total_timesteps * switch_fraction)
        self._steps_taken = 0
        self._switched = False
        self._pending_switch = False

        super().__init__(num_legs=start_legs, xml_file=xml_file, **kwargs)

        # Lock action space to start_legs dims — never changes
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(start_legs * 2,), dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Observation space size stays fixed to start_legs throughout
    # ------------------------------------------------------------------

    def _get_obs_dim(self):
        if self._exclude_current_positions_from_observation:
            return 27 + (self._start_legs - 4) * 4
        else:
            return 29 + (self._start_legs - 4) * 4

    # ------------------------------------------------------------------
    # Leg switch
    # ------------------------------------------------------------------

    def _do_switch(self):
        """Rebuild the MuJoCo model with end_legs at an episode boundary."""
        self.num_legs = self._end_legs
        self._set_num_legs(self._end_legs)

        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        # Sync init state so reset_model() uses the new morphology
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self._switched = True
        self._pending_switch = False

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def _pad_obs(self, obs):
        target = self._get_obs_dim()
        if len(obs) < target:
            obs = np.concatenate([obs, np.zeros(target - len(obs))])
        return obs

    def step(self, action):
        self._steps_taken += 1

        if (
            not self._switched
            and not self._pending_switch
            and self._steps_taken >= self._switch_step
        ):
            self._pending_switch = True

        # Trim to however many actuators the current model has
        effective_action = action[: self.num_legs * 2]
        obs, reward, terminated, truncated, info = super().step(effective_action)

        info["num_legs"] = self.num_legs
        info["switched"] = self._switched
        return self._pad_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self._pending_switch:
            self._do_switch()

        obs, info = super().reset(**kwargs)
        info["num_legs"] = self.num_legs
        info["switched"] = self._switched
        return self._pad_obs(obs), info
