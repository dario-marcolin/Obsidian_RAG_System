"""Counts wikilinks per note in the test vault and prints the distribution."""

import re, glob
from collections import Counter

files = glob.glob('Test_Md_Docs/**/*.md', recursive=True)
link_pattern = re.compile(r'\[\[([^\]|#]+)')

counts = []
for f in files:
    with open(f, encoding='utf-8', errors='ignore') as fh:
        content = fh.read()
    counts.append(len(link_pattern.findall(content)))

dist = Counter(counts)
print('Distribuzione (n_link : n_note):')
for k in sorted(dist):
    print(f'  {k} link -> {dist[k]} note')
print(f'Max link in una singola nota: {max(counts)}')