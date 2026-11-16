"""The dataset card of ``fancaps_index``: scale, columns and a few distribution plots.

Everything on the card is computed from the table itself (per-episode image counts and the episode
descriptions the job already holds), so the numbers cannot drift from the data they describe.

The plots follow one rule set: one series, one colour, thin marks, solid hairline grid, no value on every
mark and no legend (the title names the series). The images are opaque and drawn on a light surface so
they read the same on the light and the dark theme of the hub.
"""
import os
from collections import Counter
from typing import Dict, List

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from hfutils.utils import number_to_tag  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .client import CDN_URL  # noqa: E402
from .table import EpisodeInfo  # noqa: E402

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e8e7e3'
AXIS = '#cfcdc7'
SERIES = '#2a78d6'

FIGURES = [
    ('images_per_episode.png', 'Images per episode'),
    ('episodes_per_bangumi.png', 'Episodes per bangumi'),
    ('top_bangumis.png', 'The 20 largest bangumis by images'),
    ('growth.png', 'Cumulative images along the episode id'),
]


def compute_stats(episode_counts: Dict[int, int], episodes: Dict[int, EpisodeInfo], min_id: int, max_id: int) -> dict:
    """Numbers and arrays the card and the plots need."""
    episode_ids = sorted(episode_counts)
    counts = np.array([episode_counts[e] for e in episode_ids], dtype=np.int64)
    images_by_bangumi: Counter = Counter()
    episodes_by_bangumi: Counter = Counter()
    names: Dict[int, str] = {}
    for e in episode_ids:
        info = episodes[e]
        images_by_bangumi[info.bangumi_id] += episode_counts[e]
        episodes_by_bangumi[info.bangumi_id] += 1
        names[info.bangumi_id] = info.bangumi_name
    return {
        'images': int(counts.sum()),
        'episodes': len(episode_ids),
        'bangumis': len(episodes_by_bangumi),
        'min_id': int(min_id),
        'max_id': int(max_id),
        'episode_ids': np.array(episode_ids, dtype=np.int64),
        'counts': counts,
        'images_by_bangumi': images_by_bangumi,
        'episodes_by_bangumi': episodes_by_bangumi,
        'names': names,
    }


def _style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK_2, 'axes.linewidth': 0.8,
        'xtick.color': INK_2, 'ytick.color': INK_2, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
        'axes.axisbelow': True, 'font.size': 9,
    })


def _figure(title: str):
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.set_title(title, loc='left', fontsize=10.5, color=INK, pad=12)
    return fig, ax


def _short(value, _):
    """``1.2k`` / ``30M`` tick labels that stay distinct from each other."""
    if value >= 1e6:
        return f'{value / 1e6:g}M'
    return f'{value / 1e3:g}k' if value >= 1e3 else f'{value:g}'


def _hist(path: str, title: str, values: np.ndarray, xlabel: str, unit: str, upper: int, width: int):
    """Linear histogram of the bulk up to ``upper``; what lies beyond is named, not drawn."""
    fig, ax = _figure(title)
    # integer-valued data (width 1): centre every bar on its integer
    bins = np.arange(0, upper + width, width) - (0.5 if width == 1 else 0)
    ax.hist(values[values <= upper], bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')
    ax.grid(axis='x', visible=False)
    ax.yaxis.set_major_formatter(FuncFormatter(_short))
    median = float(np.median(values))
    ax.axvline(median, color=INK_2, linewidth=1.0)
    ax.annotate(f'median {median:,.0f} {unit}', xy=(median, 1), xycoords=('data', 'axes fraction'),
                xytext=(6, -12), textcoords='offset points', color=INK_2, fontsize=8.5)
    over = values[values > upper]
    if len(over):
        ax.annotate(f'{len(over):,} beyond {upper:,} (largest {int(values.max()):,})', xy=(1, 0.86),
                    xycoords='axes fraction', ha='right', color=INK_2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _top(path: str, title: str, stats: dict):
    fig, ax = _figure(title)
    ranked = stats['images_by_bangumi'].most_common(20)[::-1]
    labels = [(stats['names'][b][:38] + '…') if len(stats['names'][b]) > 38 else stats['names'][b] for b, _ in ranked]
    values = [v for _, v in ranked]
    ax.barh(range(len(values)), values, color=SERIES, height=0.62, edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.grid(axis='y', visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(_short))
    ax.set_xlabel('images')
    ax.annotate(f'{values[-1]:,}', xy=(values[-1], len(values) - 1), xytext=(5, 0), textcoords='offset points',
                va='center', color=INK_2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _growth(path: str, title: str, stats: dict):
    fig, ax = _figure(title)
    x = stats['episode_ids']
    y = np.cumsum(stats['counts'])
    ax.plot(x, y, color=SERIES, linewidth=2.0)
    ax.set_xlabel('episode id')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_ylabel('images')
    ax.yaxis.set_major_formatter(FuncFormatter(_short))
    ax.annotate(f'{int(y[-1]):,} images in {len(x):,} episodes', xy=(x[-1], y[-1]), xytext=(-8, -16),
                textcoords='offset points', ha='right', color=INK_2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_figures(stats: dict, directory: str) -> List[str]:
    """Write the plots into ``directory``; returns the file names."""
    _style()
    os.makedirs(directory, exist_ok=True)
    _hist(os.path.join(directory, FIGURES[0][0]), FIGURES[0][1], stats['counts'], 'images in the episode', 'images',
          upper=1200, width=20)
    per_bangumi = np.array(sorted(stats['episodes_by_bangumi'].values()), dtype=np.int64)
    _hist(os.path.join(directory, FIGURES[1][0]), FIGURES[1][1], per_bangumi, 'episodes in the bangumi', 'episodes',
          upper=60, width=1)
    _top(os.path.join(directory, FIGURES[2][0]), FIGURES[2][1], stats)
    _growth(os.path.join(directory, FIGURES[3][0]), FIGURES[3][1], stats)
    return [name for name, _ in FIGURES]


def _quantiles(values: np.ndarray) -> Dict[str, int]:
    return {k: int(np.percentile(values, q)) for k, q in (('p10', 10), ('median', 50), ('p90', 90), ('p99', 99))}


def render_readme(stats: dict, figures_dir: str, samples: List[dict]) -> str:
    """The dataset card text."""
    per_episode = _quantiles(stats['counts'])
    per_bangumi = _quantiles(np.array(list(stats['episodes_by_bangumi'].values()), dtype=np.int64))
    lines = [
        '---', 'license: other', 'task_categories:', '- image-classification',
        '- zero-shot-image-classification', '- text-to-image', 'language:', '- en', '- ja', '- ru', 'tags:', '- art',
        '- anime', '- not-for-all-audiences', 'size_categories:', f'- {number_to_tag(stats["images"])}',
        'annotations_creators:', '- no-annotation', 'source_datasets:', '- fancaps', '---', '',
        '# Fancaps anime screenshots: index', '',
        'One row per screenshot of the anime section of [Fancaps](https://fancaps.net/): the screenshot id, where its '
        'bytes live, and which bangumi and episode it belongs to. No image bytes are stored here.', '',
        '## Scale', '',
        '| | |', '|:--|--:|',
        f'| screenshots | {stats["images"]:,} |',
        f'| bangumis | {stats["bangumis"]:,} |',
        f'| episodes | {stats["episodes"]:,} |',
        f'| screenshot id range | {stats["min_id"]:,} to {stats["max_id"]:,} |',
        f'| screenshots per episode, median (p10 to p90) | {per_episode["median"]:,} '
        f'({per_episode["p10"]:,} to {per_episode["p90"]:,}) |',
        f'| episodes per bangumi, median (p10 to p90) | {per_bangumi["median"]:,} '
        f'({per_bangumi["p10"]:,} to {per_bangumi["p90"]:,}) |',
        '', 'The tail is long: one episode in a hundred holds more than '
        f'{per_episode["p99"]:,} screenshots, and the largest bangumi alone is '
        f'{max(stats["images_by_bangumi"].values()):,} of them.', '',
        f'![images per episode](figures/{FIGURES[0][0]})', '',
        f'![episodes per bangumi](figures/{FIGURES[1][0]})', '',
        f'![largest bangumis](figures/{FIGURES[2][0]})', '',
        f'![growth](figures/{FIGURES[3][0]})', '',
        '## Columns', '',
        '| column | type | meaning |', '|:--|:--|:--|',
        '| `id` | int64 | screenshot id, unique; the table is sorted by it, largest first |',
        '| `url` | string | full-size image, always `' + CDN_URL.replace('{id}', '{id}') + '` |',
        '| `thumbnail_url` | string | thumbnail address exactly as the site page gave it |',
        '| `bangumi_id` | int64 | show id on the site |',
        '| `bangumi_name` | string | show title |',
        '| `bangumi_url` | string | the show\'s page |',
        '| `episode_id` | int64 | episode id on the site |',
        '| `episode_name` | string | episode title (`Episode 5`, `Special`, ...) |',
        '| `episode_url` | string | the episode\'s page |', '',
        'Each `episode_id` belongs to exactly one bangumi and carries one title and one page address, so the '
        'episode and bangumi tables can be recovered with a `groupby`. `meta.json` lists the episode ids and bangumi '
        'ids that have rows, and `bangumi.json` is that hierarchy ready made.', '',
        '## Sample records', '',
        'One screenshot from each of the bangumis that hold the newest screenshots.', '',
        '| id | bangumi_name | bangumi_id | episode_name | episode_id | url |', '|--:|:--|--:|:--|--:|:--|',
    ]
    for s in samples:
        lines.append(f'| {s["id"]} | {s["bangumi_name"]} | {s["bangumi_id"]} | {s["episode_name"]} | '
                     f'{s["episode_id"]} | {s["url"]} |')
    lines.append('')
    return '\n'.join(lines)
