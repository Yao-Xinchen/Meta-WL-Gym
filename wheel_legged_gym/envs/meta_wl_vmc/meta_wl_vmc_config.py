from wheel_legged_gym.envs.meta_wl.meta_wl_config import (
    MetaWLCfg,
    MetaWLCfgPPO,
)

class MetaWLVMCCfg(MetaWLCfg):
    class env(MetaWLCfg.env):
        num_privileged_obs = (
            MetaWLCfg.env.num_observations + 7 * 11 + 3 + 6 * 7 + 3 + 3
        )

    class control(MetaWLCfg.control):
        action_scale_theta = 0.5

    class normalization(MetaWLCfg.normalization):
        class obs_scales(MetaWLCfg.normalization.obs_scales):
            l0 = 5.0
            l0_dot = 0.25

    class noise(MetaWLCfg.noise):
        class noise_scales(MetaWLCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1