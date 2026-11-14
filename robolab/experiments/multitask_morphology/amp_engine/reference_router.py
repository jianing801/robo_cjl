"""Route reference queries by immutable physical environment, not task labels."""
import torch


class ReferenceRouter:
    def __init__(self, terms, env_library, device):
        if not terms:
            raise ValueError("empty reference bank")
        self.terms, self.device = terms, device
        self.env_library = torch.as_tensor(env_library, dtype=torch.long, device=device)
        if self.env_library.min() < 0 or self.env_library.max() >= len(terms):
            raise ValueError("environment/reference library mismatch")
        self.offsets = [0]
        for term in terms:
            self.offsets.append(self.offsets[-1]+term.get_num_motions())
        self.num_dofs, self.num_key_bodies = terms[0].num_dofs, terms[0].num_key_bodies
        if any((t.num_dofs, t.num_key_bodies) != (self.num_dofs, self.num_key_bodies) for t in terms):
            raise ValueError("inconsistent AMP reference semantics")

    def sample_motions_for_envs(self, env_ids):
        out = torch.empty(len(env_ids), device=self.device, dtype=torch.long)
        groups = self.env_library[env_ids]
        for i, term in enumerate(self.terms):
            ids = (groups == i).nonzero().flatten()
            if len(ids):
                out[ids] = term.sample_motions(len(ids))+self.offsets[i]
        return out

    def sample_motions(self, n):
        raise RuntimeError("AMP reference sampling must provide physical env_ids, not only a batch size")

    def _route(self, method, motion_ids, **kwargs):
        if len(motion_ids) == 0:
            return getattr(self.terms[0], method)(motion_ids, **kwargs)
        output = None
        for i, term in enumerate(self.terms):
            indices = ((motion_ids >= self.offsets[i]) & (motion_ids < self.offsets[i+1])).nonzero().flatten()
            if not len(indices):
                continue
            args = {k: v[indices] if torch.is_tensor(v) and v.ndim else v for k, v in kwargs.items()}
            value = getattr(term, method)(motion_ids[indices]-self.offsets[i], **args)
            if isinstance(value, dict):
                if output is None:
                    output = {k: torch.empty((len(motion_ids), *v.shape[1:]), dtype=v.dtype, device=v.device) for k, v in value.items()}
                for key in value:
                    output[key][indices] = value[key]
            else:
                if output is None:
                    output = torch.empty((len(motion_ids), *value.shape[1:]), dtype=value.dtype, device=value.device)
                output[indices] = value
        if output is None or (motion_ids < 0).any() or (motion_ids >= self.offsets[-1]).any():
            raise ValueError("invalid encoded AMP reference IDs")
        return output

    def get_motion_state(self, motion_ids, motion_times):
        return self._route("get_motion_state", motion_ids, motion_times=motion_times)

    def get_motion_durations(self, motion_ids):
        return self._route("get_motion_durations", motion_ids)

    def sample_times(self, motion_ids, **kwargs):
        return self._route("sample_times", motion_ids, **kwargs)

    def get_num_motions(self):
        return self.offsets[-1]

    def get_total_duration(self):
        return sum(t.get_total_duration() for t in self.terms)
