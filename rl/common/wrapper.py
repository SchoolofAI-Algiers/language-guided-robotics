class KukaWrapper:
    """
    Shared base for all strategy wrappers (Alpha/Beta/Gamma).

    Holds only what's genuinely universal across all three:
    - inference_mode flag (disables training-only randomness like dropout)
    - action pass-through to the environment

    Note: set_instruction()/set_target_object() were originally planned for
    this base class, but investigation during #9 found the real, working
    versions already live in RewardShapingWrapper (rl/reward_shaping.py),
    which sits OUTSIDE these wrappers in the stack (RewardShapingWrapper ->
    BetaLanguageConditionedWrapper -> KukaEnv). Duplicating them here with a
    different signature would create two disconnected, conflicting versions.
    Not included in this base class for that reason.
    """

    def _init_common(self):
        # Call this from each subclass's __init__, after super().__init__(env)
        self._inference_mode = False

    def step(self, action):
        # Actions always pass straight through, unchanged, for every strategy.
        return self.env.step(action)