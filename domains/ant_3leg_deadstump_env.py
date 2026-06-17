import mujoco
from .ant_variable_legs import AntVariableLegsEnv


class Ant3LegDeadStumpEnv(AntVariableLegsEnv):
    """
    4-leg ant with the rear-right leg (right_back_leg / aux_4) rigidly welded:
    hip_4 and ankle_4 joints are removed; all bodies and geoms are kept.

    Properties vs. healthy 4-leg ant:
      nu : 8 → 6   (two fewer actuators)
      nq : 15 → 13 (two fewer joint DOF)
      nv : 14 → 12
      mass: identical (no geometry removed)

    Intended for transfer / fine-tuning experiments where a policy trained on
    the healthy ant must adapt to the crippled morphology.
    """

    def __init__(self, xml_file="ant_3leg_deadstump.xml", **kwargs):
        kwargs.setdefault("num_legs", 3)
        super().__init__(xml_file=xml_file, **kwargs)

        assert self.model.nu == 6, (
            f"Expected 6 actuators (8→6 after removing hip_4/ankle_4), got {self.model.nu}"
        )
        assert self.model.nq == 13, (
            f"Expected nq=13 (15→13 after removing 2 joints), got {self.model.nq}"
        )
        assert self.model.nv == 12, (
            f"Expected nv=12 (14→12 after removing 2 joints), got {self.model.nv}"
        )
        assert self.action_space.shape[0] == self.model.nu, (
            f"action_space dim {self.action_space.shape[0]} != model.nu {self.model.nu}"
        )

    def _set_num_legs(self, num_legs):
        # Static XML — skip the dynamic leg-generation logic in AntVariableLegsEnv.
        # The template copy that would overwrite ant_3leg_deadstump.xml is also blocked.
        pass
