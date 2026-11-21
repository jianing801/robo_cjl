"""Deployable student observations; teachers keep their native clean inputs.

Noise is drawn once for each new sensor frame, using a separate RNG so that
adding a student never changes the simulator's randomization stream.
"""
import math
import torch
from amp_adapter import StreamingHistory
from randomization import student_noise_spec
from evaluation import EVALUATION_VELOCITY_FRAME, FIXED_COURSE_VELOCITY_FRAME

STUDENT_ACTION_CLIP = 100.0
STUDENT_OBSERVATION_CLIP = 100.0


def forward_velocity(robot_data, frame):
    """Use the commanded horizontal axis when measuring forward progress."""
    if frame == EVALUATION_VELOCITY_FRAME:
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat
        return quat_apply_inverse(yaw_quat(robot_data.root_quat_w), robot_data.root_lin_vel_w)[:, 0]
    if frame == FIXED_COURSE_VELOCITY_FRAME:
        return robot_data.root_lin_vel_w[:, 0]
    if frame is None:
        return robot_data.root_lin_vel_b[:, 0]
    raise ValueError(f'Unknown forward velocity frame: {frame}')


def boundary_contract():
    # Numerical guards inherited from the direct tasks, NOT hardware limits.
    return dict(student_action_clip=STUDENT_ACTION_CLIP,
                student_observation_clip=STUDENT_OBSERVATION_CLIP,
                previous_action='executed_unscaled')


class StudentHistory:
    def __init__(self, num_envs, device, task, *, profile, strength, seed):
        self.history = StreamingHistory(num_envs, device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        spec = student_noise_spec(task, profile=profile, strength=strength)
        self.amplitudes = torch.tensor(spec['frame_amplitudes'], device=device)

    def append(self, clean_frame, reset=None):
        if clean_frame.shape != self.history.frames[:, -1].shape:
            raise ValueError('Student requires one physical 78-dimensional frame')
        if not torch.isfinite(clean_frame).all():
            raise ValueError('Nonfinite student sensor input')
        noise = torch.rand(clean_frame.shape, device=clean_frame.device,
                           dtype=clean_frame.dtype, generator=self.generator) * 2 - 1
        frame = (clean_frame + noise * self.amplitudes).clamp(
            -STUDENT_OBSERVATION_CLIP, STUDENT_OBSERVATION_CLIP)
        return self.history.append(frame, reset)


def speed_bins(command, task):
    """Three moving-speed bands (0, 1, 2); never collect standing labels."""
    low, high = {'stairs': (.2, .6), 'backpack': (.2, .8), 'run': (1.5, 2.5)}[task]
    vx = command[:, 0]
    if not torch.isfinite(command).all() or (vx <= .05).any():
        raise ValueError('Continuous-locomotion collection requires positive moving commands; standing commands are disabled')
    return (((vx - low) / (high - low) * 3).floor().long()).clamp(0, 2)


class EpisodeTracker:
    def __init__(self, num_envs, device):
        self.episode = torch.zeros(num_envs, dtype=torch.long, device=device)

    def advance(self, done):
        self.episode += done.to(dtype=torch.long, device=self.episode.device)


class TrackingError:
    """Initial-episode vx tracking and requested motion, never reset samples.

    ``valid`` selects real velocity measurements. ``command_valid`` can also
    include the last pre-reset command: pass the pre-step alive mask there and
    ``alive & ~done`` as valid when no terminal velocity is available.
    """
    # Retain zero-command exposure diagnostics to detect incompatible old data;
    # the current collection/evaluation protocol does not permit standing.
    COMMAND_ZERO_TOLERANCE_M_S = .05

    def __init__(self, num_envs, device):
        self.squared = torch.zeros(num_envs, device=device, dtype=torch.float64)
        self.count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.command_seconds = torch.zeros_like(self.squared)
        self.active_command_seconds = torch.zeros_like(self.squared)
        self.commanded_distance = torch.zeros_like(self.squared)
        self.command_squared_seconds = torch.zeros_like(self.squared)
        self.command_aligned_distance = torch.zeros_like(self.squared)
        self.command_count = torch.zeros_like(self.count)
        self.segments = {name: dict(command_count=torch.zeros_like(self.count),
            tracking_count=torch.zeros_like(self.count), squared_error=torch.zeros_like(self.squared),
            command_squared_seconds=torch.zeros_like(self.squared), actual_squared=torch.zeros_like(self.squared))
            for name in ('moving', 'standing')}
        self.step_dt = None

    def update(self, commanded_vx, actual_vx, valid, *, dt=None, command_valid=None):
        if any(value.shape != self.count.shape for value in (commanded_vx, actual_vx, valid)):
            raise ValueError('Tracking requires one command, velocity and validity per environment')
        if valid.dtype != torch.bool:
            raise ValueError('Tracking validity must be boolean')
        command_valid = valid if command_valid is None else command_valid
        if command_valid.shape != valid.shape or command_valid.dtype != torch.bool or (valid & ~command_valid).any():
            raise ValueError('Command exposure must include every valid tracking sample')
        if not torch.isfinite(commanded_vx[command_valid]).all() or not torch.isfinite(actual_vx[valid]).all():
            raise ValueError('Nonfinite valid command or velocity')
        if dt is not None:
            if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
                raise ValueError('Tracking dt must be positive finite seconds')
            if self.step_dt is not None and not math.isclose(self.step_dt, dt, rel_tol=0., abs_tol=1e-10):
                raise ValueError('Tracking requires a constant control dt')
            if self.step_dt is None and self.count.any():
                raise ValueError('Cannot start command exposure after un-timed tracking samples')
            self.step_dt = float(dt)
        elif self.step_dt is not None or command_valid is not valid:
            raise ValueError('Tracking dt is required for command exposure')
        command, actual = commanded_vx.to(torch.float64), actual_vx.to(torch.float64)
        self.squared[valid] += (actual[valid] - command[valid]).square()
        self.count[valid] += 1
        moving = command.abs() > self.COMMAND_ZERO_TOLERANCE_M_S
        for name, selected in (('moving', moving), ('standing', ~moving)):
            segment = self.segments[name]
            measured = valid & selected
            segment['tracking_count'][measured] += 1
            segment['squared_error'][measured] += (actual[measured] - command[measured]).square()
            segment['actual_squared'][measured] += actual[measured].square()
            if dt is not None:
                exposed = command_valid & selected
                segment['command_count'][exposed] += 1
                segment['command_squared_seconds'][exposed] += command[exposed].square() * dt
        if dt is not None:
            active = command_valid & moving
            self.command_count[command_valid] += 1
            self.command_seconds[command_valid] += dt
            self.active_command_seconds[active] += dt
            self.commanded_distance[active] += command[active].abs() * dt
            self.command_squared_seconds[command_valid] += command[command_valid].square() * dt
            measured_motion = valid & moving
            self.command_aligned_distance[measured_motion] += (
                actual[measured_motion] * command[measured_motion].sign() * dt)

    def values(self):
        """Tensor convenience API: unavailable measurements are NaN, never zero."""
        return torch.where(self.count > 0, (self.squared / self.count.clamp(min=1)).sqrt(),
                           torch.full_like(self.squared, float('nan')))

    def rows(self, *, horizontal=False):
        """JSON-safe metrics. Use this, not float(values()[i]), in reports."""
        if self.step_dt is None:
            raise ValueError('Timed tracking updates are required for an evaluation report')
        counts = self.count.cpu().tolist()
        errors = self.values().cpu().tolist()
        command_seconds = self.command_seconds.cpu().tolist()
        active_seconds = self.active_command_seconds.cpu().tolist()
        distance = self.commanded_distance.cpu().tolist()
        aligned = self.command_aligned_distance.cpu().tolist()
        squared = self.command_squared_seconds.cpu().tolist()
        command_counts = self.command_count.cpu().tolist()
        rows = [dict(tracking_rmse=errors[i] if counts[i] else None,
            tracking_samples=counts[i], tracking_seconds=counts[i] * self.step_dt,
            command_samples=command_counts[i], command_seconds=command_seconds[i], active_command_seconds=active_seconds[i],
            commanded_distance_m=distance[i], command_aligned_distance_m=aligned[i],
            commanded_vx_rms_m_s=math.sqrt(squared[i] / command_seconds[i]) if command_seconds[i] else None,
            step_dt_s=self.step_dt, command_zero_tolerance_m_s=self.COMMAND_ZERO_TOLERANCE_M_S)
            for i in range(len(counts))]
        for name, segment in self.segments.items():
            values = {key: tensor.cpu().tolist() for key, tensor in segment.items()}
            for i, row in enumerate(rows):
                nc, nv = values['command_count'][i], values['tracking_count'][i]
                row.update({f'{name}_command_samples': nc, f'{name}_command_seconds': nc * self.step_dt,
                    f'{name}_commanded_vx_rms_m_s': math.sqrt(values['command_squared_seconds'][i] / (nc * self.step_dt)) if nc else None,
                    f'{name}_tracking_samples': nv, f'{name}_tracking_seconds': nv * self.step_dt,
                    f'{name}_tracking_rmse': math.sqrt(values['squared_error'][i] / nv) if nv else None})
                if name == 'standing':
                    row['standing_vx_rms_m_s'] = math.sqrt(values['actual_squared'][i] / nv) if nv else None
        if horizontal:
            for row in rows:
                seconds = row['moving_tracking_seconds']
                row['mean_horizontal_forward_speed_m_s'] = (
                    row['command_aligned_distance_m'] / seconds if seconds else None)
                for key in ('tracking_rmse', 'moving_tracking_rmse', 'standing_tracking_rmse', 'standing_vx_rms_m_s'):
                    del row[key]
        return rows
