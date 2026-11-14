"""Record real simulator frames; stop focus clip at its first termination."""
import imageio.v2 as imageio
import numpy as np

def reuse_shader_cache(output):
    from pathlib import Path
    import shutil
    output=Path(output)
    source=output.parent/'jfv1_stairs/portable/cache/shadercache'
    target=output/'portable/cache/shadercache'
    if source.is_dir() and source.resolve()!=target.resolve() and not target.exists():
        shutil.copytree(source,target)

class Recorder:
    def __init__(self,raw,path):
        import omni.usd
        from pxr import UsdGeom
        stage=omni.usd.get_context().get_stage()
        for index in range(1,raw.num_envs):
            prim=stage.GetPrimAtPath(f'/World/envs/env_{index}/Robot')
            if prim.IsValid():UsdGeom.Imageable(prim).MakeInvisible()
        self.raw=raw;self.path=path;self.writer=imageio.get_writer(str(path),fps=25,codec='libx264',quality=8)
        self.ended=False;self.frames=0
    def capture(self,step,done):
        if self.ended:return
        if bool(done[0]):self.ended=True;return
        if step%2:return
        pos=self.raw.robot.data.root_pos_w[0].tolist()
        self.raw.sim.set_camera_view((pos[0]+2.5,pos[1]-3.,pos[2]+1.3),(pos[0],pos[1],pos[2]+.1))
        for attempt in range(30 if self.frames==0 else 1):
            self.raw.sim.render();frame=self.raw.render(recompute=True)
            if frame is not None and np.asarray(frame).size and np.asarray(frame).std()>1:break
        if frame is None or not np.asarray(frame).size or np.asarray(frame).std()<=1:raise ValueError('Simulator returned no usable camera image')
        if self.frames==0:imageio.imwrite(str(self.path.with_suffix('.png')),frame)
        self.writer.append_data(frame);self.frames+=1
    def close(self):
        self.writer.close()
        if self.frames<10:raise ValueError('Recording has too few real frames')
