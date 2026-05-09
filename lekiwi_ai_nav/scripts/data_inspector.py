#!/usr/bin/env python3
"""Data inspector for recorded imitation-learning episodes.

Usage:
  python3 scripts/data_inspector.py episodes/
  python3 scripts/data_inspector.py episodes/ --episode 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2


def load_episode(ep_path: Path) -> dict:
    """Load one episode directory into a dict. Returns None on fatal error."""
    meta_file = ep_path / 'metadata.json'
    if not meta_file.exists():
        print(f'  [WARN] missing metadata.json in {ep_path}')
        return None

    with open(meta_file) as f:
        meta = json.load(f)

    actions_file = ep_path / 'actions.npy'
    poses_file = ep_path / 'poses.npy'
    timestamps_file = ep_path / 'timestamps.npy'

    missing = []
    for fpath in [actions_file, poses_file, timestamps_file]:
        if not fpath.exists():
            missing.append(fpath.name)
    if missing:
        print(f'  [WARN] missing files: {missing}')

    actions = np.load(actions_file) if actions_file.exists() else None
    poses = np.load(poses_file) if poses_file.exists() else None
    timestamps = np.load(timestamps_file) if timestamps_file.exists() else None

    rgb_files = sorted((ep_path / 'rgb').glob('*.jpg'))
    depth_files = sorted((ep_path / 'depth').glob('*.png'))

    return {
        'path': ep_path,
        'meta': meta,
        'actions': actions,
        'poses': poses,
        'timestamps': timestamps,
        'rgb_files': rgb_files,
        'depth_files': depth_files,
    }


def compute_stats(ep: dict) -> dict:
    actions = ep['actions']
    poses = ep['poses']

    if actions is None or len(actions) == 0:
        return {'error': 'no actions'}

    n = len(actions)
    stats = {
        'num_frames': n,
        'actions': {},
        'pose_range': {},
        'moving_ratio': 0.0,
    }

    for i, label in enumerate(['vx', 'vy', 'wz']):
        col = actions[:, i]
        stats['actions'][label] = {
            'min': float(col.min()),
            'max': float(col.max()),
            'mean': float(col.mean()),
            'std': float(col.std()),
        }

    if poses is not None and len(poses) > 0:
        stats['pose_range'] = {
            'x': [float(poses[:, 0].min()), float(poses[:, 0].max())],
            'y': [float(poses[:, 1].min()), float(poses[:, 1].max())],
            'theta': [float(poses[:, 2].min()), float(poses[:, 2].max())],
        }
        speed = np.linalg.norm(actions[:, :2], axis=1)
        stats['moving_ratio'] = float((speed > 0.01).mean())

    return stats


def print_episode_info(ep: dict):
    meta = ep['meta']
    stats = compute_stats(ep)
    ep_name = ep['path'].name

    print(f'\n{"="*60}')
    print(f'  Episode: {ep_name}')
    print(f'{"="*60}')
    print(f'  Path:       {ep["path"]}')
    print(f'  Frames:     {ep["meta"].get("num_frames", "?")}')
    print(f'  Duration:   {ep["meta"].get("duration", "?"):.1f}s' if isinstance(ep['meta'].get('duration'), (int, float)) else f'  Duration:   {ep["meta"].get("duration", "?")}')
    print(f'  RGB files:  {len(ep["rgb_files"])}')
    print(f'  Depth files:{len(ep["depth_files"])}')
    print(f'  Sync mean:  {ep["meta"].get("mean_sync_latency", 0)*1000:.1f}ms')
    print(f'  Sync max:   {ep["meta"].get("max_sync_latency", 0)*1000:.1f}ms')

    if 'error' in stats:
        print(f'  [ERROR] {stats["error"]}')
        return

    print(f'\n  Actions:')
    for label, s in stats['actions'].items():
        print(f'    {label}: min={s["min"]:+.3f} max={s["max"]:+.3f} '
              f'mean={s["mean"]:+.3f} std={s["std"]:.3f}')

    print(f'\n  Pose range:')
    pr = stats['pose_range']
    print(f'    x:     [{pr["x"][0]:.2f}, {pr["x"][1]:.2f}]')
    print(f'    y:     [{pr["y"][0]:.2f}, {pr["y"][1]:.2f}]')
    print(f'    theta: [{pr["theta"][0]:.2f}, {pr["theta"][1]:.2f}]')
    print(f'  Moving ratio: {stats["moving_ratio"]:.1%}')

    # Flag checks
    flags = []
    duration = ep['meta'].get('duration', 0)
    if duration < 3.0:
        flags.append(f'SHORT (< 3s, actual={duration:.1f}s)')

    actions = ep['actions']
    if actions is not None and len(actions) > 0:
        zero_ratio = float((np.abs(actions).sum(axis=1) < 0.001).mean())
        if zero_ratio > 0.5:
            flags.append(f'MANY ZERO ACTIONS ({zero_ratio:.0%})')

    rgb = len(ep['rgb_files'])
    depth = len(ep['depth_files'])
    nf = ep['meta'].get('num_frames', 0)
    if rgb != nf or depth != nf:
        flags.append(f'FILE COUNT MISMATCH (rgb={rgb}, depth={depth}, expected={nf})')

    if flags:
        print(f'\n  [FLAGS]')
        for f in flags:
            print(f'    ! {f}')
    else:
        print(f'\n  [OK] No issues detected.')


def plot_histograms(episodes: list):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n[SKIP] matplotlib not available for histograms.')
        return

    all_actions = []
    for ep in episodes:
        if ep['actions'] is not None:
            all_actions.append(ep['actions'])
    if not all_actions:
        return
    all_actions = np.concatenate(all_actions, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ['vx', 'vy', 'wz']
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.hist(all_actions[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{label} histogram')
        ax.set_xlabel(label)
        ax.set_ylabel('count')
    fig.tight_layout()
    out = Path.cwd() / 'action_histograms.png'
    fig.savefig(out, dpi=100)
    print(f'\nSaved histogram plot: {out}')


def plot_trajectories(episodes: list):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n[SKIP] matplotlib not available for trajectory plot.')
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    for ep in episodes:
        poses = ep['poses']
        if poses is not None and len(poses) > 1:
            ax.plot(poses[:, 0], poses[:, 1], linewidth=1.0, label=ep['path'].name)
            ax.scatter(poses[0, 0], poses[0, 1], marker='o', s=40)
            ax.scatter(poses[-1, 0], poses[-1, 1], marker='x', s=40)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('2D Trajectory')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out = Path.cwd() / 'trajectory_plot.png'
    fig.savefig(out, dpi=100)
    print(f'Saved trajectory plot: {out}')


def show_rgb_frames(episodes: list):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n[SKIP] matplotlib not available for frame preview.')
        return

    ep = episodes[0]
    rgb_files = ep['rgb_files']
    actions = ep['actions']
    if not rgb_files:
        print('\n[SKIP] No RGB frames to display.')
        return

    n = min(4, len(rgb_files))
    indices = np.linspace(0, len(rgb_files) - 1, n, dtype=int)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        img = cv2.imread(str(rgb_files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        if actions is not None and idx < len(actions):
            a = actions[idx]
            ax.set_title(f'Frame {idx}\nvx={a[0]:.2f} vy={a[1]:.2f} wz={a[2]:.2f}', fontsize=9)
        ax.axis('off')

    fig.tight_layout()
    out = Path.cwd() / 'frame_preview.png'
    fig.savefig(out, dpi=120)
    print(f'Saved frame preview: {out}')


def main():
    parser = argparse.ArgumentParser(description='Inspect recorded imitation-learning episodes')
    parser.add_argument('directory', help='Path to episodes directory')
    parser.add_argument('--episode', '-e', type=int, default=None, help='Inspect a specific episode ID')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    base = Path(args.directory)
    if not base.is_dir():
        print(f'[ERROR] Directory not found: {base}')
        sys.exit(1)

    episode_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith('episode_')])

    if args.episode is not None:
        target = base / f'episode_{args.episode:04d}'
        if not target.is_dir():
            print(f'[ERROR] Episode not found: {target}')
            sys.exit(1)
        episode_dirs = [target]

    print(f'Found {len(episode_dirs)} episode(s) in {base}')

    episodes = []
    for ep_dir in episode_dirs:
        ep = load_episode(ep_dir)
        if ep:
            episodes.append(ep)
            print_episode_info(ep)

    if not episodes:
        print('No valid episodes found.')
        sys.exit(1)

    if not args.no_plots and len(episodes) > 0:
        plot_histograms(episodes)
        plot_trajectories(episodes)
        show_rgb_frames(episodes)

    print('\nDone.')


if __name__ == '__main__':
    main()
