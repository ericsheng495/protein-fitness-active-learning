"""Generate README figures. Optional dependency: matplotlib."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets'
INK, MUTED, BLUE, GREEN, GOLD = '#15263d', '#5c6b80', '#3b69df', '#128477', '#c47b1b'
BG = '#f6f8fc'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK})


def canvas(title, subtitle, height=5.6):
    fig, ax = plt.subplots(figsize=(14, height), facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 14), ylim=(0, height))
    ax.axis('off')
    ax.text(.5, height-.55, title, fontsize=23, weight='bold')
    ax.text(.5, height-.91, subtitle, fontsize=11, color=MUTED)
    return fig, ax


def box(ax, x, y, w, h, title, body='', color=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=.12',
                               facecolor='white', edgecolor='#dce3ef', linewidth=1.2))
    ax.plot([x+.18, x+.18], [y+.2, y+h-.2], color=color, linewidth=3)
    ax.text(x+.35, y+h-.38, title, fontsize=12, weight='bold', color=color)
    if body:
        ax.text(x+.35, y+h-.69, body, fontsize=10.5, color=INK, va='top', linespacing=1.65)


def arrow(ax, start, end, color=MUTED, rad=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
                               linewidth=1.6, color=color, connectionstyle=f'arc3,rad={rad}'))


def save(fig, name):
    OUT.mkdir(exist_ok=True, parents=True)
    fig.savefig(OUT / name, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def overview():
    fig, ax = canvas('One amino acid changes. The model ranks its effect.',
                     'Actual archived example: W560F  ·  position 560 is zero-based', 5.8)
    sequence = ''.join(line.strip() for line in (ROOT/'data/sequence.fasta').read_text().splitlines()
                       if not line.startswith('>'))
    assert sequence[560] == 'W'
    labels = ['Wild type', 'Candidate']
    for row, label in enumerate(labels):
        y = 3.86-row*.77
        ax.text(.55, y, label, fontsize=11, color=MUTED, va='center')
        for index, aa in enumerate(sequence[557:564]):
            x = 2.05+index*.52
            selected = index == 3
            if row == 1 and selected: aa = 'F'
            color = (GREEN if row else GOLD) if selected else '#dfe6f2'
            ax.add_patch(Circle((x, y), .205, facecolor=color, edgecolor='none'))
            ax.text(x, y, aa, ha='center', va='center', fontsize=11,
                    color='white' if selected else INK, weight='bold')
    ax.text(3.61, 2.55, '560: W → F', ha='center', color=GREEN, fontsize=11, weight='bold')
    box(ax, 6.0, 2.8, 3.25, 1.65, 'Frozen protein model',
        'ESM-2 sequence context\n+ biochemical descriptors\n+ zero-shot mutation score')
    box(ax, 10.0, 2.8, 3.4, 1.65, 'Trained predictor',
        'Six base regressors\n→ Ridge stacking model\n→ predicted DMS fitness', GREEN)
    arrow(ax, (5.5,3.5),(5.95,3.5)); arrow(ax,(9.3,3.5),(9.95,3.5))
    scores = {r['mutant']: float(r['DMS_score_predicted']) for r in
              csv.DictReader((ROOT/'results/submissions/predictions.csv').open())}
    box(ax, .55, .6, 5.1, 1.35, 'Representation stays fixed',
        'No ESM fine-tuning. Fit the downstream regressors.', BLUE)
    box(ax, 6.0, .6, 7.4, 1.35, 'Archived output',
        f"W560F  ·  predicted DMS {scores['W560F']:.3f}  ·  first in submitted top 10", GREEN)
    ax.text(.55, .2, 'Predicted fitness is a model output, not independent wet-lab validation.',
            fontsize=9, color=MUTED)
    save(fig, 'prediction-walkthrough.png')


def acquisition():
    fig, ax = canvas('Which 100 measurements should come next?',
                     'Active learning balances promising mutants, model disagreement, and unexplored regions.', 5.7)
    box(ax,.55,2.7,3.55,1.7,'1  Fit and score','Start with 1,140 labels\nPredict the candidate pool\nEstimate base-model disagreement')
    box(ax,5.05,2.7,3.9,1.7,'2  Select a diverse batch','Exploit: high predicted fitness\nExplore: uncertainty + distance\nExclude previous queries',GREEN)
    box(ax,9.9,2.7,3.55,1.7,'3  Measure and retrain','Request up to 100 labels\nMerge returned DMS scores\nRefit downstream models',GOLD)
    arrow(ax,(4.15,3.55),(5.,3.55));arrow(ax,(9.,3.55),(9.85,3.55))
    arrow(ax,(11.6,2.6),(2.3,2.6),color=GREEN,rad=-.17)
    ax.text(7,1.94,'Repeat for three rounds · 300 new labels total',ha='center',color=GREEN,weight='bold')
    for x, title, body in [(0.55,'Round 1','Hybrid exploration'),(5.05,'Round 2','Multi-strategy portfolio'),(9.9,'Round 3','Focus on the top tail')]:
        box(ax,x,.45,3.55,1.12,title,body)
    save(fig,'active-learning-loop.png')


def results():
    fig, ax = plt.subplots(figsize=(12,4.3),facecolor=BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=.1,right=.96,bottom=.27,top=.77)
    x=[0,1,2]; y=[.05,.19,.39]
    ax.plot(x,y,color=BLUE,linewidth=3,zorder=2)
    ax.scatter(x,y,color=BLUE,s=100,zorder=3,edgecolors='white',linewidths=2)
    for xx,yy in zip(x,y):ax.annotate(f'~{yy:.2f}',(xx,yy),xytext=(0,13),textcoords='offset points',ha='center',weight='bold',fontsize=14)
    ax.set_xticks(x,['Baseline','PLM + initial queries','Stacking + PCA + portfolio'])
    ax.set(xlim=(-.18,2.18),ylim=(0,.46),ylabel='Spearman correlation')
    ax.set_yticks([0,.1,.2,.3,.4])
    ax.grid(axis='y',color='#dce3ef',linewidth=.8)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color('#c5cedd')
    ax.tick_params(colors=MUTED,length=0,pad=9)
    fig.text(.1,.91,'Reported ranking performance',fontsize=21,weight='bold',color=INK)
    fig.text(.1,.075,'Source: team report. Approximate external scores; stages combine multiple changes.',fontsize=10,color=MUTED)
    fig.text(.1,.025,'No separate post-round-3 score is provided. Local holdout results use a different protocol.',fontsize=10,color=MUTED)
    save(fig,'reported-performance.png')


if __name__ == '__main__':
    overview()
    acquisition()
    results()
    print(f'Generated three figures in {OUT}')
