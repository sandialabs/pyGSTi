#!/usr/bin/env python

import json
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg') # For non-interactive like on remote cluster
import matplotlib.pyplot as plt


def extract_runtimes(path):
    """Extract total runtime from output files.
    
    Parameters
    ----------
    path: str
        Path to directory with output files

    Returns
    -------
    timings: dict
        Dictionary of timings with keys as <num_procs>_<num_host_procs>
    """
    wd = Path(path)
    datafiles = list(wd.glob('*_*.out'))

    timings = {}
    for df in datafiles:
        key = df.name[:-4] # Remove .out

        with df.open() as fin:
            lines = fin.readlines()

        time = None
        for l in lines:
            if 'Total Time' in l:
                time = float(l.split()[-1][:-2])

        if time:
            if key not in timings.keys():
                timings[key] = {}
            timings[key] = time
    
    return timings

def plot_speedup_boxplot(timings, title):
    """Plot 2D grid of speedups (total num procs vs num procs per 'host').
    
    Parameters
    ----------
    timings: dict
        Timing dict with keys as <num_procs>_<num_host_procs>
    title: str
        Plot title
    """
    procs = [1, 2, 4, 8, 16, 32, 64]
    xs = range(len(procs))

    serial = timings["1_1"]

    fig, ax = plt.subplots(1)

    # Colormap and normalizer
    Scmap = mpl.cm.get_cmap('viridis_r')
    Snorm = mpl.colors.Normalize(vmin=0.0, vmax=64.0)

    for i in range(len(xs)):
        for j in range(0, i+1):
            try:
                S = serial / timings[f'{procs[i]}_{procs[j]}']
            except KeyError:
                S = 0.0
            color = Scmap(Snorm(S))

            rect = mpl.patches.Rectangle((i, j), 1, 1, ec='k', fc=color, zorder=1)
            ax.add_patch(rect)

            tc = 'w' if Snorm(S) > 0.5 else 'k'
            plt.text(i+0.5, j+0.5, f'{S:2.1f}x', c=tc, ha='center', va='center', fontsize=12)

    plt.xlim([0,len(procs)])
    plt.xticks(xs, [])
    plt.xlabel('Number of Processors')

    minor = [xi + 0.5 for xi in xs]
    ax.tick_params(axis='x', which='minor', size=0)
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(minor))
    ax.xaxis.set_minor_formatter(mpl.ticker.FixedFormatter(procs))

    plt.ylim([0,len(procs)])
    plt.yticks(xs, [])
    plt.ylabel('Processors Per Shared Memory Group')

    ax.tick_params(axis='y', which='minor', size=0)
    ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(minor))
    ax.yaxis.set_minor_formatter(mpl.ticker.FixedFormatter(procs))

    ax.set_aspect(1)

    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=Snorm, cmap=Scmap), ax=ax)
    cbar.set_label('Speedup', rotation=90)
    cbar.set_ticks([0, 4, 16, 32, 64])
    
    plt.title(title, fontsize=16)

    plt.tight_layout()
    plt.savefig('2Dspeedup.pdf')


if __name__ == '__main__':
    timings = extract_runtimes('.') 
    
    with open('timings.json', 'w') as fout:
        json.dump(timings, fout, indent=2, sort_keys=True)

    plot_speedup_boxplot(timings, '2D Speedup')
