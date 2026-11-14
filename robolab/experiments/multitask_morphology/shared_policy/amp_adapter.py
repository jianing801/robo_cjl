"""Pure AMP history adaptation; never reinterpret its native normalizer input."""
import torch

TERM_WIDTHS=(3,3,3,23,23,23)

def decode_amp_frames(native):
    if native.ndim!=2 or native.shape[1]!=234:raise ValueError('Native AMP requires three term-major frames')
    return torch.cat([part.reshape(len(native),3,width)
        for part,width in zip(native.split([3*w for w in TERM_WIDTHS],dim=1),TERM_WIDTHS)],dim=-1)

class StreamingHistory:
    def __init__(self,num_envs,device):
        self.frames=torch.zeros(num_envs,10,78,device=device)
        self.initialized=torch.zeros(num_envs,dtype=torch.bool,device=device)

    def append(self,frame,reset=None):
        if frame.shape!=(len(self.frames),78):raise ValueError('One actual current frame per environment required')
        mask=~self.initialized
        if reset is not None:mask=mask|reset.bool()
        self.frames[:,:-1]=self.frames[:,1:].clone();self.frames[:,-1]=frame
        self.frames[mask]=frame[mask,None,:];self.initialized[:]=True
        return self.frames.flatten(1).clone()
