from pathlib import Path
from html import escape
P=Path(__file__).resolve().parent
W,H=1600,1100
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1100" viewBox="0 0 1600 1100" role="img" aria-labelledby="title desc">', '<title id="title">Align tasks across ranks before sharding</title>', '<desc id="desc">Illustrative two-rank scheduling comparison. Interleaved A0 B0 A1 B1 sends task A to rank 0 and B to rank 1. Synchronized rank slots A0 A1 B0 B1 align task A on both ranks, then task B. Slots contain distinct batches, not replicated samples. Task A can use eight examples per rank and task B two. Eighteen real CPU Gloo two-process Trainer jobs cover six configurations times full, prefix and resume. No performance or GPU claim.</desc>']
colors={'A':('#e5f1fd','#17619b'),'B':('#fff0d8','#925c13')}
def rect(x,y,w,h,fill,stroke='none',r=12):s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
def text(x,y,t,size=22,fill='#18283b',weight=400,anchor='start'):
 s.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(t)}</text>')
def line(x1,y1,x2,y2,c='#94a3b8',width=2):s.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{c}" stroke-width="{width}" fill="none"/>')
def batch(x,y,label,w=130,h=55):
 bg,fg=colors[label[0]];rect(x,y,w,h,bg,fg,8);text(x+w/2,y+36,label,25,fg,600,'middle')
rect(0,0,W,H,'#ffffff',r=0)
text(56,63,'Align tasks across ranks before sharding',37,weight=700)
text(56,103,'Sentence Transformers · two-rank scheduling schematic (W = 2) · not a timing diagram',22,'#566579')
rect(56,129,1488,48,'#f2f5f9',r=8)
text(78,160,'Task A: retrieval / all_gather',21,colors['A'][1],600)
text(530,160,'Task B: similarity / no all_gather',21,colors['B'][1],600)
text(1120,160,'A₀, A₁: distinct batches',21,'#566579')
for x in [56,816]:rect(x,200,728,579,'#fbfcfe','#d6dfe9',14)
text(82,243,'BEFORE · interleaved task batches',27,weight=700)
text(842,243,'PROPOSED · same-task rank slots',27,weight=700)
text(82,283,'Batch stream before rank sharding',21,'#566579')
text(842,283,'Each group has W distinct batches of one task',21,'#566579')
for i,label in enumerate(['A₀','B₀','A₁','B₁']):batch(82+i*167,305,label)
for i,label in enumerate(['A₀','A₁','B₀','B₁']):batch(842+i*167,305,label)
# Group boundaries, not sample duplication.
for x,t in [(842,'A group'),(1176,'B group')]:
 line(x,375,x+297,375);text(x+148.5,401,t,18,'#566579',400,'middle')
text(82,408,'Accelerate selects alternating rank slots',20,'#566579')
text(842,435,'Accelerate alone selects each rank’s shard',20,'#566579')
for x in [56,816]:
 text(x+268,476,'Step 1',21,'#566579',600,'middle');text(x+528,476,'Step 2',21,'#566579',600,'middle')
 text(x+26,531,'Rank 0',21,weight=600);text(x+26,608,'Rank 1',21,weight=600)
for x,rows in [(56,[['A₀','A₁'],['B₀','B₁']]),(816,[['A₀','B₀'],['A₁','B₁']])]:
 for row,labels in enumerate(rows):
  for col,label in enumerate(labels):batch(x+183+col*260,495+row*77,label,170,55)
# Each vertical pair = same local training step.
for x in [56,816]:
 for c in range(2):line(x+268+c*260,552,x+268+c*260,570,'#aab6c5',2)
rect(82,652,676,98,'#fff1ef',r=8)
text(101,685,'A ≠ B at the same step',23,'#a3312c',700)
text(101,718,'Task-specific collectives can be misaligned.',22,'#a3312c')
rect(842,652,676,98,'#eaf6ee',r=8)
text(861,685,'A = A, then B = B',23,'#26623d',700)
text(861,718,'Same task on both ranks at every step.',22,'#26623d')
text(56,823,'Heterogeneous sizes remain per rank:',24,weight=700)
text(594,823,'A = 8 examples/rank · B = 2 examples/rank',24,'#18283b')
text(56,859,'Distinct sample shards; task-dependent batch sizes.',21,'#566579')
line(56,886,1544,886,'#d6dfe9')
text(56,925,'Validated: 18 real CPU/Gloo two-process Trainer runs',25,weight=700)
text(56,961,'6 configurations × full / prefix / resume; RR & proportional; G = 1/2; epoch/resume; nonzero tails.',21,'#566579')
text(56,996,'G = gradient accumulation steps; deterministic CPU fixtures.',20,'#566579')
text(56,1047,'Implementation validated; upstream review pending',22,'#26623d',600)
url='https://github.com/Aurantium-47/sentence-transformers/tree/b71dd46c9021a409b701cc3b5e28286c8605ed13/research/synchronized-dataset-batches'
s.append(f'<a href="{url}">');text(1544,1047,'Pinned evidence: b71dd46 · 2026-11-02',20,'#17619b',400,'end');s.append('</a>')
s.append('</svg>');(P/'synchronized-task-scheduling.svg').write_text('\n'.join(s))
