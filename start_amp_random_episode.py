"""启动 AMP 轮换对照，每次换组采用普通 AMP 的首回合随机计时。"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

import start as expert


CONFIG = expert.ENTRY / 'configs' / 'amp_rotating_random_episode.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0, help='显卡编号，默认 0')
    roots = parser.add_mutually_exclusive_group()
    roots.add_argument('--root', type=Path, help='已有实验资源的根目录')
    roots.add_argument('--migrated', action='store_true', help='使用迁移后的 2500 结构库')
    parser.add_argument('--run-name', default='amp', help='独立实验目录名，默认 amp')
    parser.add_argument('--resume-paused', action='store_true', help='明确恢复本实验的暂停')
    parser.add_argument('--dry-run', action='store_true', help='预览启动命令')
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error('显卡编号不能为负数')
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.run_name) is None:
        parser.error('--run-name 仅接受字母、数字、下划线和横线，且以字母或数字开头')

    root = (args.root or (expert.MIGRATED_ROOT if args.migrated else expert.DEFAULT_ROOT)).expanduser().resolve()
    if (args.migrated or (root / 'migration_in_progress.json').is_file()) and not (root / 'migration_completed.json').is_file():
        parser.error('结构库迁移尚未完成')

    bank = expert.bank_path(root, 'amp')
    if not bank.is_file():
        raise FileNotFoundError(f'缺少 AMP 结构库：{bank}')
    output = root / 'experts_random_episode' / args.run_name
    if output.resolve() != output:
        raise ValueError(f'随机回合训练需使用独立的输出目录：{output}')
    plan_path = output / 'plan.json'
    resume = plan_path.is_file()
    if output.exists() and not resume:
        raise ValueError(f'目录已存在但缺少训练计划，请用 --run-name 另开目录：{output}')

    config = expert.read(CONFIG)
    if resume:
        plan = expert.read(plan_path)
        if (plan['configuration'] != config or Path(plan['bank']).resolve() != bank.resolve()
                or plan['execution_protocol'] != 'amp_persistent_v1'):
            raise ValueError(f'已有计划与本实验不匹配：{plan_path}')
    elif args.resume_paused:
        parser.error('当前目录没有可恢复的训练计划')

    command = expert.expert_command(root, 'amp', args.gpu, output, resume)
    command.extend(['--config', str(CONFIG), '--execution-mode', 'persistent'])
    library = Path(command[command.index('--reference-library') + 1])
    if not library.is_file():
        raise FileNotFoundError(f'缺少 AMP 参考动作库：{library}')
    if '--asset-cache' not in command:
        raise FileNotFoundError('缺少已登记的 AMP USD 缓存，请先准备 AMP 资产')
    if (output / 'USER_PAUSED').exists():
        if not args.resume_paused:
            raise RuntimeError(f'该实验已暂停，有已提交断点时请加 --resume-paused：{output}')
        if not any((output / 'commits').glob('stage_*.json')):
            raise RuntimeError('尚无已提交组断点，请用 --run-name 从首组另开实验')
        expert.resume_expert_pause(root, 'amp', output, preview=args.dry_run)

    print(f'AMP 换组随机首回合计时实验：{output}', flush=True)
    if args.dry_run:
        expert.run(command, preview=True)
        return
    expert.start_task_timer(root, 'amp-random-episode', output, config['rotation']['total_updates'],
                            expert.ENTRY_STARTED_WALL, expert.ENTRY_STARTED_MONOTONIC,
                            start_source=Path(__file__).name)
    os.chdir(expert.REPO)
    if sys.platform != 'win32':
        os.execv(sys.executable, command)
    raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f'启动未完成：{error}', file=sys.stderr)
        raise SystemExit(1)
