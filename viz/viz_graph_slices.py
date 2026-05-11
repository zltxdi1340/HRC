import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

import matplotlib.pyplot as plt


def _normalize_edges(raw_edges) -> List[Tuple[str, str]]:
    out = []
    for e in raw_edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.append((str(e[0]), str(e[1])))
    return out


def _summary_prefix(summary_path: Path) -> str:
    stem = summary_path.stem
    if stem.endswith('_summary'):
        return stem[:-8]
    return stem


def _order_within_level(nodes: List[str]) -> List[str]:
    ns = sorted(nodes)
    if 'success' in ns:
        ns = [x for x in ns if x != 'success'] + ['success']
    return ns


def _layout_by_levels(nodes: List[str], levels: Dict[str, int]) -> Dict[str, Tuple[float, float]]:
    by: Dict[int, List[str]] = {}
    for n in nodes:
        by.setdefault(int(levels.get(n, 0)), []).append(n)

    pos: Dict[str, Tuple[float, float]] = {}
    for lv, ns in sorted(by.items()):
        ns = _order_within_level(ns)
        k = len(ns)
        for i, n in enumerate(ns):
            y = (i - (k - 1) / 2.0) * 1.35
            x = float(lv) * 2.2
            pos[n] = (x, y)
    return pos


def _build_adj(edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, [])
    for k in adj:
        adj[k] = sorted(set(adj[k]))
    return adj


def _enumerate_paths(edges: List[Tuple[str, str]], nodes: List[str], max_nodes: int) -> List[List[str]]:
    adj = _build_adj(edges)
    found: Set[Tuple[str, ...]] = set()

    def dfs(path: List[str]):
        if len(path) >= 2:
            found.add(tuple(path))
        if len(path) >= max_nodes:
            return
        last = path[-1]
        for nxt in adj.get(last, []):
            if nxt in path:
                continue
            dfs(path + [nxt])

    for n in nodes:
        dfs([n])

    # keep only 2-node and 3-node path slices by default semantics
    keep = [list(p) for p in sorted(found, key=lambda x: (len(x), x)) if 2 <= len(p) <= max_nodes]
    return keep


def extract_slices(summary: Dict, max_nodes: int = 3) -> Dict:
    active = list(summary.get('ACTIVE', []))
    levels = dict(summary.get('levels', {}))
    edges = _normalize_edges(summary.get('H_edges', []))

    indeg = {n: 0 for n in active}
    outdeg = {n: 0 for n in active}
    for u, v in edges:
        indeg[v] = indeg.get(v, 0) + 1
        outdeg[u] = outdeg.get(u, 0) + 1
        indeg.setdefault(u, 0)
        outdeg.setdefault(v, 0)

    roots = sorted([n for n in active if indeg.get(n, 0) == 0])
    leaves = sorted([n for n in active if outdeg.get(n, 0) == 0])

    node_slices = [
        {
            'slice_id': f'node:{n}',
            'type': 'single_node',
            'nodes': [n],
            'edges': [],
        }
        for n in active
    ]

    edge_slices = [
        {
            'slice_id': f'edge:{u}->{v}',
            'type': 'direct_edge',
            'nodes': [u, v],
            'edges': [[u, v]],
        }
        for (u, v) in edges
    ]

    path_slices = []
    for path in _enumerate_paths(edges, active, max_nodes=max_nodes):
        if len(path) < 3:
            continue
        path_edges = [[path[i], path[i + 1]] for i in range(len(path) - 1)]
        path_slices.append(
            {
                'slice_id': 'path:' + '->'.join(path),
                'type': f'path_{len(path)}_nodes',
                'nodes': path,
                'edges': path_edges,
            }
        )

    all_slices = node_slices + edge_slices + path_slices

    return {
        'env_id': summary.get('env_id'),
        'mission': summary.get('mission'),
        'ACTIVE': active,
        'levels': levels,
        'roots': roots,
        'leaves': leaves,
        'num_edges': len(edges),
        'node_slices': node_slices,
        'edge_slices': edge_slices,
        'path_slices': path_slices,
        'all_slices': all_slices,
    }


def _draw_arrow(ax, x1, y1, x2, y2, color='black', lw=1.8, alpha=1.0):
    ax.annotate(
        '',
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', lw=lw, color=color, alpha=alpha),
    )


def draw_slice_overview(summary: Dict, slices: Dict, out_png: Path, max_cols: int = 3):
    active = list(summary.get('ACTIVE', []))
    levels = dict(summary.get('levels', {}))
    edges = _normalize_edges(summary.get('H_edges', []))
    pos = _layout_by_levels(active, levels)

    all_slices = slices.get('all_slices', [])
    if not all_slices:
        return

    n = len(all_slices)
    cols = min(max_cols, max(1, n))
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    base_nodes = set(active)
    base_edges = set(edges)

    for idx, sl in enumerate(all_slices):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        highlight_nodes = set(sl['nodes'])
        highlight_edges = {tuple(e) for e in sl['edges']}

        for (u, v) in sorted(base_edges):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            if (u, v) in highlight_edges:
                _draw_arrow(ax, x1, y1, x2, y2, color='tab:blue', lw=2.3, alpha=1.0)
            else:
                _draw_arrow(ax, x1, y1, x2, y2, color='lightgray', lw=1.2, alpha=0.8)

        for nname in active:
            x, y = pos[nname]
            if nname in highlight_nodes:
                face = 'tab:blue'
                text_color = 'white'
                size = 980
                alpha = 1.0
            else:
                face = 'lightgray'
                text_color = 'black'
                size = 760
                alpha = 0.7
            ax.scatter([x], [y], s=size, color=face, alpha=alpha, zorder=3)
            ax.text(
                x, y, nname, ha='center', va='center', fontsize=9, zorder=4, color=text_color,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.2)
            )

        ax.set_title(sl['slice_id'], fontsize=10)
        ax.set_axis_off()

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_axis_off()

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', type=str, required=True, help='Path to *_summary.json from hrc_stage1.py')
    ap.add_argument('--max_nodes', type=int, default=3, help='Max nodes in a path slice; 3 means single node / direct edge / 3-node chain')
    ap.add_argument('--max_cols', type=int, default=3)
    args = ap.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    slices = extract_slices(summary, max_nodes=args.max_nodes)

    prefix = _summary_prefix(summary_path)
    out_json = summary_path.with_name(prefix + '_slices.json')
    out_png = summary_path.with_name(prefix + '_slices_overview.png')

    out_json.write_text(json.dumps(slices, ensure_ascii=False, indent=2), encoding='utf-8')
    draw_slice_overview(summary, slices, out_png, max_cols=args.max_cols)

    print(f'[SAVE] slices json -> {out_json}')
    print(f'[SAVE] slices png  -> {out_png}')


if __name__ == '__main__':
    main()
