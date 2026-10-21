"""RPO-Flat backpack variant with pitch-adaptive posture and fall resets."""

import gymnasium as gym


gym.register(
    id="RPO-Flat-Backpack-Adaptive",
    entry_point=f"{__name__}.env:BackpackRPOFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BackpackRPOFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "robolab.tasks.direct.base.agents.rpo_agent_cfg:RPOFlatAgentCfg",
    },
)

gym.register(
    id="RPO-Flat-Backpack-Bounded",
    entry_point=f"{__name__}.env:BackpackRPOFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BackpackRPOFlatBoundedEnvCfg",
        "rsl_rl_cfg_entry_point": "robolab.tasks.direct.base.agents.rpo_agent_cfg:RPOFlatAgentCfg",
    },
)

gym.register(
    id="RPO-Flat-Backpack-Compliant",
    entry_point=f"{__name__}.env:BackpackRPOFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BackpackRPOFlatCompliantEnvCfg",
        "rsl_rl_cfg_entry_point": "robolab.tasks.direct.base.agents.rpo_agent_cfg:RPOFlatAgentCfg",
    },
)
