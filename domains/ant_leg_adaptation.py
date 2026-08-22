import copy
from numbers import Integral, Real

import mujoco
import numpy as np
from gymnasium.spaces import Box

from .ant_variable_legs import AntVariableLegsEnv, DEFAULT_CAMERA_CONFIG


def _add_cleanup_note(primary_error, message):
    add_note = getattr(primary_error, "add_note", None)
    if callable(add_note):
        add_note(message)
    else:
        notes = list(getattr(primary_error, "__notes__", ()))
        notes.append(message)
        primary_error.__notes__ = notes


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
        Filename used for this instance's private generated XML.
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
        if (
            isinstance(total_timesteps, bool)
            or not isinstance(total_timesteps, Integral)
            or total_timesteps <= 0
        ):
            raise ValueError(
                "AntLegAdaptationEnv total_timesteps must be a positive integer."
            )
        for name, value in (("start_legs", start_legs), ("end_legs", end_legs)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0
            ):
                raise ValueError(
                    f"AntLegAdaptationEnv {name} must be a positive integer."
                )
        if end_legs > start_legs:
            raise ValueError(
                "AntLegAdaptationEnv end_legs cannot exceed start_legs because "
                "its observation and action spaces are fixed to start_legs."
            )
        if (
            isinstance(switch_fraction, bool)
            or not isinstance(switch_fraction, Real)
            or not np.isfinite(switch_fraction)
            or not 0.0 <= switch_fraction <= 1.0
        ):
            raise ValueError(
                "AntLegAdaptationEnv switch_fraction must be finite and in [0, 1]."
            )

        self._total_timesteps = int(total_timesteps)
        self._start_legs = int(start_legs)
        self._end_legs = int(end_legs)
        self._switch_fraction = float(switch_fraction)
        self._switch_step = int(self._total_timesteps * self._switch_fraction)
        self._steps_taken = 0
        self._switched = False
        # A zero-step first phase means the end morphology is authoritative at
        # the first reset, rather than after one accidental start-morphology step.
        self._pending_switch = self._switch_step == 0

        super().__init__(num_legs=self._start_legs, xml_file=xml_file, **kwargs)

        # Lock action space to start_legs dims — never changes
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self._start_legs * 2,), dtype=np.float64
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
        # Keep Gymnasium's optional rendering imports lazy, as MujocoEnv does.
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self._set_num_legs(self._end_legs)

        new_model = mujoco.MjModel.from_xml_path(self.xml_file)
        new_data = mujoco.MjData(new_model)
        mujoco.mj_resetData(new_model, new_data)
        new_renderer = MujocoRenderer(
            new_model,
            new_data,
            default_cam_config=DEFAULT_CAMERA_CONFIG,
        )

        old_renderer = self.mujoco_renderer
        try:
            old_renderer.close()
        except BaseException as exc:
            try:
                new_renderer.close()
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    exc,
                    "Additional replacement-renderer cleanup failure: "
                    f"{cleanup_error}",
                )
            raise

        self.num_legs = self._end_legs
        self.model = new_model
        self.data = new_data
        self.mujoco_renderer = new_renderer
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

    def training_resume_state(self):
        state = super().training_resume_state()
        state.update(
            adaptation_schema_version=1,
            steps_taken=int(self._steps_taken),
            switched=bool(self._switched),
            pending_switch=bool(self._pending_switch),
            start_legs=int(self._start_legs),
            end_legs=int(self._end_legs),
            switch_step=int(self._switch_step),
            np_random=copy.deepcopy(self.np_random.bit_generator.state),
        )
        return state

    def load_training_resume_state(self, state):
        self.validate_training_resume_state(state)
        saved_switched = state["switched"]
        if saved_switched and not self._switched:
            self._do_switch()
        self._steps_taken = state["steps_taken"]
        self._switched = saved_switched
        self._pending_switch = state["pending_switch"]
        self.np_random.bit_generator.state = copy.deepcopy(state["np_random"])

    def validate_training_resume_state(self, state):
        expected = {
            "schema_version",
            "num_legs",
            "np_random",
            "adaptation_schema_version",
            "steps_taken",
            "switched",
            "pending_switch",
            "start_legs",
            "end_legs",
            "switch_step",
        }
        if (
            not isinstance(state, dict)
            or set(state) != expected
            or state.get("schema_version") != 1
            or state.get("adaptation_schema_version") != 1
        ):
            raise ValueError("Unsupported leg-adaptation training-resume state.")
        immutable = {
            "start_legs": self._start_legs,
            "end_legs": self._end_legs,
            "switch_step": self._switch_step,
        }
        for key, value in immutable.items():
            if state.get(key) != int(value):
                raise ValueError(f"Leg-adaptation {key} changed across resume.")
        for key in ("switched", "pending_switch"):
            if not isinstance(state.get(key), bool):
                raise ValueError(f"Leg-adaptation {key} must be bool.")
        steps_taken = state.get("steps_taken")
        if isinstance(steps_taken, bool) or not isinstance(steps_taken, int) or steps_taken < 0:
            raise ValueError("Leg-adaptation steps_taken is invalid.")
        saved_switched = state["switched"]
        pending_switch = state["pending_switch"]
        if saved_switched:
            if pending_switch:
                raise ValueError(
                    "A switched morphology cannot also have a pending switch."
                )
            if steps_taken < self._switch_step:
                raise ValueError(
                    "A switched morphology cannot predate the switch step."
                )
        elif pending_switch:
            if steps_taken < self._switch_step:
                raise ValueError(
                    "A pending morphology switch cannot predate the switch step."
                )
        elif steps_taken >= self._switch_step:
            raise ValueError(
                "An unswitched morphology at or after the switch step must be pending."
            )
        expected_legs = self._end_legs if saved_switched else self._start_legs
        if state.get("num_legs") != int(expected_legs):
            raise ValueError("Leg-adaptation morphology state is inconsistent.")
        if saved_switched and not self._switched:
            # The actual model rebuild occurs only after every checkpoint
            # component has passed preflight.
            pass
        elif not saved_switched and self._switched:
            raise ValueError("Cannot restore a pre-switch state into a switched model.")
        probe = copy.deepcopy(self.np_random)
        probe.bit_generator.state = copy.deepcopy(state["np_random"])
