"""Create native task subscenes on a shared Isaac Lab stage."""
from dataclasses import is_dataclass
import torch
from isaaclab.managers import EventManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage
from isaaclab.utils.noise import NoiseModel


def scoped_paths(value, scope):
    if isinstance(value, str) and value.startswith('/World/'):
        return scope + value[len('/World'):]
    if isinstance(value, list):
        return [scoped_paths(item, scope) for item in value]
    if isinstance(value, tuple):
        return tuple(scoped_paths(item, scope) for item in value)
    if isinstance(value, dict):
        return {key: scoped_paths(item, scope) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        # Isaac Lab attaches terrain and robot configs after scene creation.
        for name, item in vars(value).items():
            setattr(value, name, scoped_paths(item, scope))
    return value


class TaskScene(InteractiveScene):
    def __init__(self, cfg, scope):
        self.scope = scope
        super().__init__(cfg)

    @property
    def env_ns(self):
        return self.scope + '/envs'

    def filter_collisions(self, global_prim_paths=None):
        scope = self.scope + '/collisions'
        if self.stage.GetPrimAtPath(scope):
            return
        globals_ = list(dict.fromkeys([*self._global_prim_paths, self.cfg.terrain.prim_path]))
        self.cloner.filter_collisions(self.physics_scene_path, scope, self.env_prim_paths, globals_)


def add_subscene(sim, env_type, cfg, task):
    cfg.validate()
    if cfg.sim.dt != sim.get_physics_dt() or cfg.sim.device != sim.device:
        raise ValueError(f'{task} needs different simulation timing or device')
    scope = '/World/Mixed_' + task
    cfg.scene = scoped_paths(cfg.scene, scope)
    env = env_type.__new__(env_type)
    # This driver owns the simulator lifecycle, including failure during setup.
    env._is_closed = True
    env.sim, env.cfg, env.render_mode = sim, cfg, None
    env._window = None
    env.viewport_camera_controller = None
    env._sim_step_counter = env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=sim.device, dtype=torch.long)
    env.extras, env.obs_buf = {}, {}
    env.seed(cfg.seed)
    env.scene = TaskScene(cfg.scene, scope)
    if task != 'run':
        env._setup_scene()
    else:
        env.scene.filter_collisions()
    env.event_manager = EventManager(cfg.events, env)
    if 'prestartup' in env.event_manager.available_modes:
        env.event_manager.apply(mode='prestartup')
    return env


def initialize_task(env, task):
    env.scene.update(dt=env.physics_dt)
    if task == 'run':
        env.load_managers()
        return
    env._debug_vis_handle = None
    env.has_debug_vis_implementation = False
    env.reset_terminated = torch.zeros_like(env.episode_length_buf, dtype=torch.bool)
    env.reset_time_outs = torch.zeros_like(env.reset_terminated)
    env.reset_buf = torch.zeros_like(env.reset_terminated)
    env._configure_gym_env_spaces()
    for setting in ('action_noise_model', 'observation_noise_model'):
        cfg = getattr(env.cfg, setting)
        if cfg:
            setattr(env, '_' + setting, cfg.class_type(cfg, num_envs=env.num_envs, device=env.device))
    if 'startup' in env.event_manager.available_modes:
        env.event_manager.apply(mode='startup')
    env._initialize_task()


def create_subscenes(recipes, task_names):
    cfgs = [cfg for _, cfg in recipes]
    if any(cfg.decimation != cfgs[0].decimation for cfg in cfgs):
        raise ValueError('Task control decimation differs')
    sim_cfg = cfgs[0].sim.copy()
    configs = [cfg.sim.physx.to_dict() for cfg in cfgs]
    differing = {key: [cfg[key] for cfg in configs]
                 for key in configs[0] if any(cfg[key] != configs[0][key] for cfg in configs[1:])}
    # Rough reserves a larger collision stack. Capacity can be shared across
    # tasks, while solver behavior must match exactly.
    stack = differing.pop('gpu_collision_stack_size', None)
    if differing:
        raise ValueError(f'Task PhysX solver settings differ: {differing}')
    if stack:
        sim_cfg.physx.gpu_collision_stack_size = max(stack)
    sim = SimulationContext(sim_cfg)
    with use_stage(sim.get_initial_stage()):
        tasks = [add_subscene(sim, env_type, cfg, name)
                 for (env_type, cfg), name in zip(recipes, task_names, strict=True)]
        attach_stage_to_usd_context()
        print('MIXED_SIM_RESET', [env.num_envs for env in tasks], flush=True)
        sim.reset()
        for env, name in zip(tasks, task_names, strict=True):
            initialize_task(env, name)
    return sim, tasks
