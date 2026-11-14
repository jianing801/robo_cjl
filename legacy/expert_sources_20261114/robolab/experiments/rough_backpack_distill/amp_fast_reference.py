"""Cache reference membership indices only; retain native interpolation/RNG order."""
import torch

def merge_native_tables(router):
    """Join immutable frames, then call the ORIGINAL term interpolation methods.

    Sampling remains per physical library and in original group order.
    Only deterministic state/duration lookup is eligible for vectorization.
    """
    first=router.terms[0]
    motion_fields=('motion_fps','motion_dt','motion_durations','motion_num_frames','motion_loop_modes')
    frame_fields=('root_pos_w','root_quat','root_vel_w','root_ang_vel_w','dof_pos','dof_vel','key_body_pos_w')
    if any(not hasattr(term,key) for term in router.terms for key in (*motion_fields,*frame_fields,'motion_start_indices')):
        return None
    merged=type(first).__new__(type(first));merged.__dict__=first.__dict__.copy()
    for key in (*motion_fields,*frame_fields):setattr(merged,key,torch.cat([getattr(term,key) for term in router.terms]))
    starts=[];offset=0
    for term in router.terms:
        starts.append(term.motion_start_indices+offset);offset+=len(term.root_pos_w)
    merged.motion_start_indices=torch.cat(starts)
    return merged

class CachedReferenceRouting:
    def __init__(self,router):
        self.router=router;self.native=router._route
        self.boundaries=torch.tensor(router.offsets[1:],device=router.device,dtype=torch.long)
        self.cache={};self.hits=0;self.misses=0
        self.merged=None;self.merged_verified=False;self.merge_fallback_reason=None

    def route(self,method,motion_ids,**kwargs):
        if motion_ids.ndim!=1 or not len(motion_ids):return self.native(method,motion_ids,**kwargs)
        if (motion_ids<0).any() or (motion_ids>=self.router.offsets[-1]).any():
            return self.native(method,motion_ids,**kwargs)
        if self.merged_verified and method in ('get_motion_state','get_motion_durations'):
            self.hits+=1
            return getattr(self.merged,method)(motion_ids,**kwargs)
        groups=torch.bucketize(motion_ids,self.boundaries,right=True)
        key=(method,len(motion_ids))
        cached=self.cache.get(key)
        if cached is None or not torch.equal(groups,cached[0]):
            indices=[(groups==i).nonzero().flatten() for i in range(len(self.router.terms))]
            if len(self.cache)>=8:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=(groups.clone(),indices);self.misses+=1
        else:indices=cached[1];self.hits+=1
        output=None
        for i,term in enumerate(self.router.terms):
            selected=indices[i]
            if not len(selected):continue
            args={k:v[selected] if torch.is_tensor(v) and v.ndim else v for k,v in kwargs.items()}
            value=getattr(term,method)(motion_ids[selected]-self.router.offsets[i],**args)
            if isinstance(value,dict):
                if output is None:output={k:torch.empty((len(motion_ids),*v.shape[1:]),dtype=v.dtype,device=v.device) for k,v in value.items()}
                for name in value:output[name][selected]=value[name]
            else:
                if output is None:output=torch.empty((len(motion_ids),*value.shape[1:]),dtype=value.dtype,device=value.device)
                output[selected]=value
        if output is None:raise ValueError('No valid reference IDs')
        return output

def install_verified(router,animation):
    """Compare native and cached deterministic reference outputs before enabling."""
    adapter=CachedReferenceRouting(router)
    ids=animation.motion_ids
    times=animation.motion_fetch_time[:,0]
    before=router.get_motion_state(ids,times)
    after=adapter.route('get_motion_state',ids,motion_times=times)
    if set(before)!=set(after) or any(not torch.equal(before[k],after[k]) for k in before):
        raise ValueError('Cached AMP reference interpolation differs from native')
    if not torch.equal(router.get_motion_durations(ids),adapter.route('get_motion_durations',ids)):
        raise ValueError('Cached AMP reference duration differs from native')
    merged=merge_native_tables(router)
    if merged is not None:
        # Exercise ALL clips and geometries, including wrap/clamp boundaries.
        all_ids=torch.arange(router.offsets[-1],device=router.device)
        durations=router.get_motion_durations(all_ids)
        exact=torch.equal(durations,merged.get_motion_durations(all_ids))
        for phase in (0.,.37,.93,1.2):
            native=router.get_motion_state(all_ids,durations*phase)
            packed=merged.get_motion_state(all_ids,durations*phase)
            exact=exact and set(native)==set(packed) and all(torch.equal(native[k],packed[k]) for k in native)
        if exact:adapter.merged=merged;adapter.merged_verified=True
        else:adapter.merge_fallback_reason='Packed lookup was not bitwise equal; using verified membership cache'
    router._route=adapter.route
    return adapter
