"""One task+design MxT actor. Native teacher actions remain unsquashed.

Adapted from this repository's previous mxt_policy/morph_policy tokenization,
with distinct stairs/backpack task IDs and the intersection of exact motor categories.
No teacher optimizer, critic, reward or dynamics are trained or modified here.
"""
import copy
import torch
from torch import nn
from contract import input_width


class SharedActor(nn.Module):
    def __init__(self,schema,architecture=None):
        super().__init__()
        self.schema=copy.deepcopy(schema)
        if len(schema['tasks'])!=2:raise ValueError('Three-task distillation uses the shared MLP architecture')
        self.architecture=dict(d_model=256,num_heads=2,num_layers=3,feedforward_dim=512)
        self.architecture.update(architecture or {})
        a=self.architecture
        self.width=input_width(schema);self.context_width=self.width-1132
        self.register_buffer('observation_mean',torch.zeros(780))
        self.register_buffer('observation_std',torch.ones(780))
        node_width=141+self.context_width
        self.encoder=nn.Linear(node_width,a['d_model'])
        self.position=nn.Parameter(torch.empty(25,a['d_model']))
        nn.init.normal_(self.position,std=.02)
        layer=nn.TransformerEncoderLayer(a['d_model'],a['num_heads'],a['feedforward_dim'],
            dropout=0.,activation='relu',batch_first=True)
        self.transformer=nn.TransformerEncoder(layer,a['num_layers'],
            norm=nn.LayerNorm(a['d_model']),enable_nested_tensor=False)
        self.decoder=nn.Linear(a['d_model']+node_width,1)

    def nodes(self,x):
        if x.ndim!=2 or x.shape[1]!=self.width:raise ValueError('Canonical student input required')
        batch=len(x)
        frames=((x[:,:780]-self.observation_mean)/self.observation_std).reshape(batch,10,78)
        n=x.new_zeros(batch,25,141+self.context_width)
        n[:,0,:60]=frames[:,:,:6].reshape(batch,-1)
        j=torch.stack((frames[:,:,9:32],frames[:,:,32:55],frames[:,:,55:78]),dim=-1)
        n[:,1:24,60:90]=j.permute(0,2,1,3).reshape(batch,23,30)
        n[:,-1,90:122]=torch.cat((frames[:,:,6:9].reshape(batch,30),x[:,780:782]),-1)
        static=x[:,782:1132].reshape(batch,25,14).clone()
        static[:,:,11]/=10. # Known nominal link mass only, never sampled load.
        n[:,:,122:136]=static
        n[:,0,136]=1;n[:,1:24,137]=1;n[:,-1,138]=1
        n[:,1:24,139]=1;n[:,0,140]=1
        n[:,:,141:]=x[:,None,1132:]
        return n

    def forward(self,x):
        n=self.nodes(x)
        latent=self.transformer(self.encoder(n)+self.position)
        return self.decoder(torch.cat((latent,n),-1)).squeeze(-1)[:,1:24]


class SharedMLPActor(nn.Module):
    """One shared MLP and output head; teacher-native history plus task/context.

    The graph-token expansion is omitted for this explicit architectural
    ablation. Known leg lengths and physical motor descriptors stay in context.
    No teacher routing, separate task networks, or privileged input is used.
    """
    def __init__(self,schema,architecture=None):
        super().__init__()
        self.schema=copy.deepcopy(schema);self.width=input_width(schema);self.task_count=len(schema['tasks'])
        self.architecture=dict(type='mlp',hidden_dims=[1024,512,256])
        self.architecture.update(architecture or {})
        self.register_buffer('observation_mean',torch.zeros(780))
        self.register_buffer('observation_std',torch.ones(780))
        widths=[self.width-350,*self.architecture['hidden_dims'],23]
        layers=[]
        for i in range(len(widths)-1):
            layers.append(nn.Linear(widths[i],widths[i+1]))
            if i<len(widths)-2:layers.append(nn.ELU())
        self.network=nn.Sequential(*layers)

    def features(self,x):
        if x.ndim!=2 or x.shape[1]!=self.width:raise ValueError('Canonical student input required')
        return torch.cat(((x[:,:780]-self.observation_mean)/self.observation_std,
                          x[:,780:780+self.task_count],x[:,1130+self.task_count:]),-1)

    def forward(self,x):return self.network(self.features(x))


def save_student(path,model,**metadata):
    with open(path,'xb') as stream:
        torch.save(dict(kind='rough_backpack_shared_mxt_actor_v1',schema=model.schema,
            architecture=model.architecture,state_dict=model.state_dict(),
            critic_trained=False,PPO_updates=0,**metadata),stream)


def load_student(path,device='cpu'):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('kind')!='rough_backpack_shared_mxt_actor_v1':raise ValueError('Wrong student kind')
    cls=SharedMLPActor if data['architecture'].get('type')=='mlp' else SharedActor
    model=cls(data['schema'],data['architecture']).to(device)
    model.load_state_dict(data['state_dict'],strict=True)
    for value in model.state_dict().values():
        if not torch.isfinite(value).all():raise ValueError('Nonfinite student weights')
    model.eval()
    return model,data


def expand_mlp_tasks(checkpoint,schema,device='cpu'):
    """Add task columns to one shared MLP, preserving old task functions."""
    old,_=load_student(checkpoint,'cpu')
    if not isinstance(old,SharedMLPActor):raise ValueError('Task expansion requires the validated shared MLP')
    for key in ('knees','ankles','semantic_contract','ranges_m','shared_structure_keys'):
        if old.schema[key]!=schema[key]:raise ValueError('Task expansion cannot change the structure contract')
    if not set(old.schema['tasks'])<set(schema['tasks']):raise ValueError('Strict task expansion required')
    new=SharedMLPActor(schema,old.architecture)
    before=old.state_dict();after=new.state_dict()
    for name,value in before.items():
        if name!='network.0.weight':after[name].copy_(value)
    after['network.0.weight'].zero_()
    after['network.0.weight'][:,:780]=before['network.0.weight'][:,:780]
    for i,task in enumerate(old.schema['tasks']):
        after['network.0.weight'][:,780+schema['tasks'].index(task)]=before['network.0.weight'][:,780+i]
    after['network.0.weight'][:,780+len(schema['tasks']):]=before['network.0.weight'][:,780+old.task_count:]
    new.load_state_dict(after,strict=True)
    return new.to(device)
