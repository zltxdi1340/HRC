#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_summary(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_edge(edge) -> Tuple[str, str]:
    if isinstance(edge, (list, tuple)) and len(edge) == 2:
        return str(edge[0]), str(edge[1])
    raise ValueError(f'Bad edge format: {edge!r}')


def collect(paths: List[str]):
    edge_to_tasks: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    task_to_edges: Dict[str, List[Tuple[str, str]]] = {}
    task_meta: Dict[str, dict] = {}

    for path in sorted(paths):
        data = load_summary(path)
        env_id = data.get('env_id', os.path.splitext(os.path.basename(path))[0])
        task_name = os.path.splitext(os.path.basename(path))[0]
        task_meta[task_name] = {
            'env_id': env_id,
            'mission': data.get('mission', ''),
            'path': path,
        }
        edges = [normalize_edge(e) for e in data.get('H_edges', [])]
        # de-duplicate inside a task
        edges = sorted(set(edges))
        task_to_edges[task_name] = edges
        for e in edges:
            edge_to_tasks[e].append(task_name)

    return edge_to_tasks, task_to_edges, task_meta


def make_report(edge_to_tasks, task_to_edges, task_meta):
    edge_rows = []
    for (u, v), tasks in sorted(edge_to_tasks.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1])):
        env_ids = [task_meta[t]['env_id'] for t in tasks]
        edge_rows.append({
            'edge': f'{u} -> {v}',
            'src': u,
            'dst': v,
            'count': len(tasks),
            'tasks': tasks,
            'env_ids': env_ids,
        })

    unique_patterns: Dict[Tuple[Tuple[str, str], ...], List[str]] = defaultdict(list)
    for task_name, edges in task_to_edges.items():
        unique_patterns[tuple(edges)].append(task_name)

    pattern_rows = []
    for edges, tasks in sorted(unique_patterns.items(), key=lambda kv: (-len(kv[1]), len(kv[0]), kv[1])):
        pattern_rows.append({
            'count': len(tasks),
            'tasks': tasks,
            'env_ids': [task_meta[t]['env_id'] for t in tasks],
            'edges': [f'{u} -> {v}' for (u, v) in edges],
        })

    return {
        'num_tasks': len(task_to_edges),
        'num_unique_edges': len(edge_rows),
        'edges': edge_rows,
        'edge_pattern_groups': pattern_rows,
    }


def write_json(path: str, obj: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, report: dict) -> None:
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['edge', 'count', 'tasks', 'env_ids'])
        for row in report['edges']:
            writer.writerow([
                row['edge'],
                row['count'],
                '; '.join(row['tasks']),
                '; '.join(row['env_ids']),
            ])


def plot_bar(path: str, report: dict, top_k: int = 20):
    rows = report['edges'][:top_k]
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, 'No edges found', ha='center', va='center', color='black')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return

    labels = [r['edge'] for r in rows][::-1]
    counts = [r['count'] for r in rows][::-1]

    fig_h = max(4, 0.5 * len(rows) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(range(len(rows)), counts)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, color='black')
    ax.set_xlabel('Frequency across tasks', color='black')
    ax.set_title('Repeated direct edges', color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    for idx, c in enumerate(counts):
        ax.text(c + 0.03, idx, str(c), va='center', color='black', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_presence_matrix(path: str, report: dict, task_meta: dict):
    edges = report['edges']
    if not edges:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, 'No edges found', ha='center', va='center', color='black')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return

    tasks = sorted(task_meta.keys())
    edge_labels = [row['edge'] for row in edges]
    matrix = []
    for edge_row in edges:
        present = set(edge_row['tasks'])
        matrix.append([1 if t in present else 0 for t in tasks])

    # edges as rows, tasks as columns
    fig_w = max(8, 0.75 * len(tasks) + 3)
    fig_h = max(4, 0.45 * len(edge_labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect='auto')
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha='right', color='black')
    ax.set_yticks(range(len(edge_labels)))
    ax.set_yticklabels(edge_labels, color='black')
    ax.set_xlabel('Tasks', color='black')
    ax.set_ylabel('Edges', color='black')
    ax.set_title('Edge presence matrix (1 = appears in task)', color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    for i in range(len(edge_labels)):
        for j in range(len(tasks)):
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center', color='black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def infer_output_prefix(paths: List[str], out_dir: str | None, prefix: str | None) -> str:
    if prefix:
        if out_dir:
            return os.path.join(out_dir, prefix)
        return prefix
    base = 'edge_frequency'
    if out_dir:
        return os.path.join(out_dir, base)
    if paths:
        return os.path.join(os.path.dirname(paths[0]) or '.', base)
    return base


def main():
    parser = argparse.ArgumentParser(description='Count repeated direct edges across multiple summary.json files.')
    parser.add_argument('--summary_glob', type=str, default=None, help='Example: outputs/stage1/*_summary.json')
    parser.add_argument('--summaries', nargs='*', default=None, help='Explicit list of summary JSON paths')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()

    paths: List[str] = []
    if args.summary_glob:
        paths.extend(sorted(glob.glob(args.summary_glob)))
    if args.summaries:
        paths.extend(args.summaries)
    # unique, preserve order
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]

    if not paths:
        raise SystemExit('No summary files found. Use --summary_glob or --summaries.')

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    edge_to_tasks, task_to_edges, task_meta = collect(paths)
    report = make_report(edge_to_tasks, task_to_edges, task_meta)
    prefix = infer_output_prefix(paths, args.out_dir, args.prefix)

    json_path = f'{prefix}.json'
    csv_path = f'{prefix}.csv'
    bar_path = f'{prefix}_bar.png'
    matrix_path = f'{prefix}_matrix.png'

    write_json(json_path, report)
    write_csv(csv_path, report)
    plot_bar(bar_path, report, top_k=args.top_k)
    plot_presence_matrix(matrix_path, report, task_meta)

    print(f'[OK] wrote {json_path}')
    print(f'[OK] wrote {csv_path}')
    print(f'[OK] wrote {bar_path}')
    print(f'[OK] wrote {matrix_path}')


if __name__ == '__main__':
    main()
