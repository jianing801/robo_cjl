"""SW拟合驱动的RPO URDF生成器。新增文件，不依赖或修改旧generate_urdf.py。
大腿：质量一次、其余四次；小腿：质量一次、其余五次（排除0.30m训练点）。
需要Python>=3.9和NumPy。用户确认数据来自左腿且坐标对齐，默认左侧直接映射、右侧y镜像。
使用说明：docs/基于SW拟合参数的URDF更新.md。
"""

# 系数按归一化变量的升幂排列，输出已经恢复到kg、m、kg·m²。
MODELS = {'thigh': {'center': 0.325,
           'half_range': 0.075,
           'training_range': [0.25, 0.4],
           'prediction_range': [0.25, 0.4],
           'coefficients': {'mass_kg': [2.35598994, 0.1213446421323529],
                            'com_x_m': [8.505509765625e-05,
                                        -4.3829512979280805e-06,
                                        2.238388697457672e-07,
                                        -7.702726839712998e-09,
                                        1.3343718370562838e-09],
                            'com_y_m': [-0.015542036735026042,
                                        -0.00020576864041289643,
                                        1.0595937672099982e-05,
                                        -5.464331165148955e-07,
                                        2.7197637740204014e-08],
                            'com_z_m': [-0.2592954185872396,
                                        -0.05310133486891726,
                                        0.0008035395757825075,
                                        -4.152897267196897e-05,
                                        2.1393120247037594e-06],
                            'sw_com_xx_kg_m2': [0.028745309309895835,
                                                0.015994758327503585,
                                                0.0027395632074787356,
                                                8.651744649018983e-05,
                                                -1.5326437935594565e-06],
                            'sw_com_xy_kg_m2': [-8.488997395833333e-07,
                                                4.407088493292064e-08,
                                                5.121894721957636e-09,
                                                -2.5396671826627155e-09,
                                                -6.985829029233114e-09],
                            'sw_com_xz_kg_m2': [-1.3985979817708335e-05,
                                                -4.394323487239031e-06,
                                                -1.6484707575464124e-07,
                                                1.4163528518705405e-08,
                                                5.102009965163316e-10],
                            'sw_com_yy_kg_m2': [0.029776268824869796,
                                                0.01613650531209069,
                                                0.002739666552048307,
                                                8.650869719720504e-05,
                                                -1.5343313814664364e-06],
                            'sw_com_yz_kg_m2': [0.0020332672298177083,
                                                0.0005145604172248357,
                                                -7.560809330532423e-06,
                                                3.885830331612971e-07,
                                                -2.5627788517506285e-08],
                            'sw_com_zz_kg_m2': [0.0018869199381510418,
                                                0.00017113329258702488,
                                                -1.2231951152810283e-07,
                                                6.6003438314893746e-09,
                                                2.2095627772334943e-08]}},
 'calf': {'center': 0.38,
          'half_range': 0.07,
          'training_range': [0.31, 0.45],
          'prediction_range': [0.3, 0.45],
          'coefficients': {'mass_kg': [1.936134374, 0.24022469225000012],
                           'com_x_m': [0.00027189469116023295,
                                       -3.373109560212751e-05,
                                       4.179230534687179e-06,
                                       -5.178848684210406e-07,
                                       7.145369207677064e-08,
                                       -1.5726959791993914e-08],
                           'com_y_m': [-0.01013701820108684,
                                       0.0013759006762046517,
                                       -0.00017068273019099013,
                                       2.116616765400871e-05,
                                       -2.6975555642780284e-06,
                                       3.4287440759293804e-07],
                           'com_z_m': [-0.214158470369785,
                                       -0.04464515973888224,
                                       0.0011965716276151604,
                                       -0.00014846175505103543,
                                       1.8847898516615475e-05,
                                       -2.343233620898896e-06],
                           'sw_com_xx_kg_m2': [0.02011522591829223,
                                               0.010765838618492495,
                                               0.0021879986548068845,
                                               0.00012090550209376295,
                                               -2.8878316130514775e-06,
                                               3.414568131277625e-07],
                           'sw_com_xy_kg_m2': [2.0902237545736e-05,
                                               -7.25784528062237e-07,
                                               9.74049467044797e-08,
                                               -5.459355402081741e-09,
                                               -5.176548168032102e-09,
                                               -4.536302294195723e-09],
                           'sw_com_xz_kg_m2': [-8.079918811838315e-06,
                                               -1.334444372072716e-05,
                                               -6.360473047334537e-07,
                                               7.702830535046936e-08,
                                               -4.277691477758623e-09,
                                               -2.384894220837361e-09],
                           'sw_com_yy_kg_m2': [0.019925431393405357,
                                               0.010796561827069942,
                                               0.002191664462687719,
                                               0.00012040890666924262,
                                               -2.831246959951147e-06,
                                               3.742616168475846e-07],
                           'sw_com_yz_kg_m2': [0.0011872423765398688,
                                               0.0006247093959712629,
                                               2.570894412540552e-05,
                                               -3.1834661973079365e-06,
                                               3.866708207936031e-07,
                                               -5.581986683411302e-08],
                           'sw_com_zz_kg_m2': [0.0012928935753101388,
                                               0.0001872972352742714,
                                               -3.6787905805495808e-06,
                                               4.691591003823239e-07,
                                               -4.928550358312218e-08,
                                               -1.4175944679931892e-09]}}}

import argparse
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

# 原模板几何长度；不用于缩放拟合得到的质量或惯量。
DEFAULT_THIGH = 0.25
DEFAULT_CALF = 0.30
TEMPLATE_URDF = Path(__file__).resolve().parents[2] / 'data/robots/roboparty/rpo/urdf/rpo.urdf'
JOINTS = {'thigh': ('left_knee_joint', 'right_knee_joint'),
          'calf': ('left_ankle_pitch_joint', 'right_ankle_pitch_joint')}
LINKS = {'thigh': ('left_thigh_pitch_link', 'right_thigh_pitch_link'),
         'calf': ('left_knee_link', 'right_knee_link')}


def predict_properties(segment, length):
    """返回SW坐标下的mass_kg、com_m及标准质心惯性矩阵inertia_kg_m2。"""
    if segment not in MODELS:
        raise ValueError('segment must be thigh or calf')
    length = float(length)
    model = MODELS[segment]
    lo, hi = model['prediction_range']
    if not math.isfinite(length) or not lo <= length <= hi:
        raise ValueError(f'{segment} length must be in [{lo}, {hi}] m')
    x = (length - model['center']) / model['half_range']
    def value(key):
        result = 0.0
        for c in reversed(model['coefficients'][key]):
            result = result*x + c
        return result
    mass = value('mass_kg')
    com = np.array([value('com_'+a+'_m') for a in 'xyz'])
    v = {a: value('sw_com_'+a+'_kg_m2') for a in ('xx','xy','xz','yy','yz','zz')}
    # SW正惯性积的非对角项取负；不对质心惯量再做平行轴平移。
    inertia = np.array([[v['xx'], -v['xy'], -v['xz']],
                        [-v['xy'], v['yy'], -v['yz']],
                        [-v['xz'], -v['yz'], v['zz']]])
    _validate_physics(mass, com, inertia)
    return {'mass_kg': mass, 'com_m': com, 'inertia_kg_m2': inertia,
            'extrapolated': length < model['training_range'][0]}


def _validate_physics(mass, com, inertia):
    if not math.isfinite(mass) or mass <= 0 or not np.isfinite(com).all() or not np.isfinite(inertia).all():
        raise ValueError('Non-finite or non-positive predicted mass/property')
    eig = np.linalg.eigvalsh(inertia)
    if eig[0] <= 0 or eig[0]+eig[1] < eig[2]-1e-12:
        raise ValueError(f'Unphysical inertia principal moments: {eig}')


def aligned_frame_map(side):
    """显式假设：两段SW坐标与指定侧link完全重合，另一侧关于y=0镜像。"""
    if side not in ('left', 'right'):
        raise ValueError('aligned side must be left or right')
    return {link: {'matrix': (np.eye(3) if link.startswith(side+'_') else np.diag([1,-1,1])).tolist(),
                   'translation_m': [0,0,0]}
            for names in LINKS.values() for link in names}


def _frame(frame_map, link):
    if link not in frame_map:
        raise ValueError(f'Missing SW-to-link frame transform for {link}')
    f = frame_map[link]
    q = np.asarray(f['matrix'], dtype=float)
    t = np.asarray(f['translation_m'], dtype=float)
    if q.shape != (3,3) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError(f'Invalid frame shape or values: {link}')
    if not np.allclose(q @ q.T, np.eye(3), rtol=0, atol=1e-9):
        raise ValueError(f'Frame matrix must be orthogonal: {link}')
    # det=-1仅表示用户明确选择的左右结构镜像，不是刚体旋转。
    return q, t


def _vec(element, attribute, default=None):
    text = element.get(attribute, default)
    if text is None:
        raise ValueError(f'Missing {element.tag}.{attribute}')
    v = np.asarray([float(s) for s in text.split()])
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f'Invalid vector {text}')
    return v


def _fmt(v):
    return ' '.join(format(float(x), '.17g') for x in v)


def _required(root, tag, name):
    matches = [e for e in root.findall(tag) if e.get('name') == name]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {tag}: {name}')
    return matches[0]


def generate_urdf(thigh_length, calf_length, output_dir, template_path=None, *,
                  frame_map=None, aligned_side=None, mesh_dir=None):
    """兼容旧函数前四个参数；默认SW坐标与左侧link对齐，右侧沿y镜像。

    坐标关系：c_link=Q*c_SW+t；I_link=Q*I_SW*Q.T。
    输出 output_dir/urdf/rpo.urdf；拒绝覆盖输入模板或已有输出。
    若使用镜像，须确认左右部件组成/几何对称且link轴定义支持该镜像。
    """
    if frame_map is None and aligned_side is None:
        aligned_side = "left"
    elif frame_map is not None and aligned_side is not None:
        raise ValueError('Provide frame_map or aligned_side, not both')
    frames = aligned_frame_map(aligned_side) if aligned_side is not None else frame_map
    mapped = {link: _frame(frames, link) for names in LINKS.values() for link in names}
    properties = {seg: predict_properties(seg, length) for seg, length in [('thigh',thigh_length),('calf',calf_length)]}
    template = Path(template_path or TEMPLATE_URDF).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve() / 'urdf/rpo.urdf'
    if not template.is_file():
        raise FileNotFoundError(f'Template not found: {template}; provide --template')
    if output == template:
        raise ValueError('Output must not overwrite the input template')
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new output directory: {output}')
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(template, parser=parser)
    root = tree.getroot()
    report = {'template': str(template), 'thigh_length_m':float(thigh_length), 'calf_length_m':float(calf_length),
              'frame_assumption': 'explicit transforms' if aligned_side is None else f'SW aligned with {aligned_side}; opposite side mirrored in y',
              'calf_extrapolated':properties['calf']['extrapolated'], 'links':{}}
    for seg, length, base in [('thigh',float(thigh_length),DEFAULT_THIGH),('calf',float(calf_length),DEFAULT_CALF)]:
        p = properties[seg]
        for joint_name, link_name in zip(JOINTS[seg], LINKS[seg]):
            joint = _required(root, 'joint', joint_name)
            parent, origin = joint.find('parent'), joint.find('origin')
            if parent is None or parent.get('link') != link_name or origin is None:
                raise ValueError(f'Unexpected topology or missing origin: {joint_name}')
            old = _vec(origin, 'xyz')
            # 从原始几何模板生成，防止二次缩放已生成的URDF。
            if not math.isclose(old[2], -base, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f'{joint_name} must have baseline z={-base}; supply original template')
            old[2] = -length
            origin.set('xyz', _fmt(old))
            link = _required(root, 'link', link_name)
            inertial = link.find('inertial')
            if inertial is None:
                raise ValueError(f'Missing inertial: {link_name}')
            q,t = mapped[link_name]
            com = q @ p['com_m'] + t
            inertia = q @ p['inertia_kg_m2'] @ q.T
            _validate_physics(p['mass_kg'], com, inertia)
            for tag in ('origin','mass','inertia'):
                if inertial.find(tag) is None:
                    ET.SubElement(inertial,tag)
            inertial.find('origin').set('xyz',_fmt(com))
            # 张量已经变换到link坐标，故惯性坐标轴与link对齐。
            inertial.find('origin').set('rpy','0 0 0')
            inertial.find('mass').set('value',format(p['mass_kg'],'.17g'))
            for name,i,j in [('ixx',0,0),('ixy',0,1),('ixz',0,2),('iyy',1,1),('iyz',1,2),('izz',2,2)]:
                inertial.find('inertia').set(name,format(float(inertia[i,j]),'.17g'))
            # 沿用旧脚本的盒碰撞体近似；不按此比例再次缩放质量属性。
            for collision in link.findall('collision'):
                box = collision.find('geometry/box')
                if box is None:
                    raise ValueError(f'{link_name} has non-box collision; geometry adaptation is required')
                co = collision.find('origin')
                if co is not None:
                    if not np.allclose(_vec(co,'rpy','0 0 0'),0,rtol=0,atol=1e-10):
                        raise ValueError(f'{link_name} has rotated collision box')
                    xyz = _vec(co,'xyz','0 0 0');xyz[2] *= length/base;co.set('xyz',_fmt(xyz))
                size = _vec(box,'size');size[2] *= length/base;box.set('size',_fmt(size))
            report['links'][link_name] = {'mass_kg':p['mass_kg'],'com_m':com.tolist(),
                'inertia_kg_m2':inertia.tolist(),'matrix':q.tolist(),'translation_m':t.tolist()}
    # 相对路径改为实际绝对文件路径，输出目录改变后仍能加载原STL。
    # 不缩放visual mesh：保持旧脚本行为；它不是重新导出的参数化CAD网格。
    for mesh in root.findall('.//mesh'):
        name = mesh.get('filename','')
        if mesh_dir is not None:
            path = Path(mesh_dir).expanduser().resolve() / name.replace('\\','/').split('/')[-1]
        elif name.startswith('package://'):
            raise ValueError('package:// meshes require --mesh-dir')
        elif name.startswith('file://'):
            path = Path(name[7:])
        else:
            path = Path(name)
            if not path.is_absolute():
                path = template.parent/path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Mesh missing: {path}; provide --mesh-dir')
        mesh.set('filename',str(path))
    output.parent.mkdir(parents=True,exist_ok=True)
    # 独占创建，禁止无意覆盖既有URDF。
    with output.open('xb') as f:
        tree.write(f,encoding='utf-8',xml_declaration=True)
    report_path = output.with_suffix('.report.json')
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thigh',type=float)
    parser.add_argument('--calf',type=float)
    parser.add_argument('--output')
    parser.add_argument('--run',choices=['co_design_train','co_design_eval'],help='Run existing entry point with this generator injected in the current process')
    parser.add_argument('--template')
    parser.add_argument('--mesh-dir')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--frame-map',help='JSON: four link names -> matrix and translation_m')
    group.add_argument('--aligned-side',choices=['left','right'],default='left',help='SW frame aligned side; default left. Opposite side is mirrored in y')
    import sys
    argv = sys.argv[1:]
    divider = argv.index('--') if '--' in argv else len(argv)
    forwarded = argv[divider+1:]
    args = parser.parse_args(argv[:divider])
    frames = json.loads(Path(args.frame_map).read_text(encoding='utf-8')) if args.frame_map else None
    aligned_side = None if frames is not None else args.aligned_side
    if args.run:
        import runpy
        import types
        script = Path(__file__).resolve().with_name(args.run+'.py')
        if not script.is_file():
            parser.error('Place this file beside the existing co_design_train/eval.py scripts')
        if any(v is not None for v in (args.thigh,args.calf,args.output)):
            parser.error('For --run, pass training/evaluation arguments after --')
        def injected(thigh_length,calf_length,output_dir,template_path=None):
            return generate_urdf(thigh_length,calf_length,output_dir,template_path or args.template,frame_map=frames,aligned_side=aligned_side,mesh_dir=args.mesh_dir)
        module = types.ModuleType('robolab.scripts.tools.generate_urdf')
        module.generate_urdf = injected
        sys.modules[module.__name__] = module
        sys.modules['robolab.scripts.tools.generate_urdf_sw'] = module
        sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
        sys.argv = [str(script)] + forwarded
        runpy.run_path(str(script),run_name='__main__')
        return
    if forwarded or any(v is None for v in (args.thigh,args.calf,args.output)):
        parser.error('Generation requires --thigh, --calf and --output; -- is reserved for --run')
    path = generate_urdf(args.thigh,args.calf,args.output,args.template,frame_map=frames,aligned_side=aligned_side,mesh_dir=args.mesh_dir)
    print(path)


if __name__ == '__main__':
    main()
