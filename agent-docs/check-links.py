#! /usr/bin/env python3
"""Verify every relative link and heading anchor in AGENTS.md and agent-docs/.

Checks two things, both of which break silently and neither of which any test
covers:

  1. Every relative markdown link resolves to a file that exists.
  2. Every ``#anchor`` on a link to a ``.md`` file matches a heading in that file.

Run it after any edit to the agent docs -- especially after moving a file,
renaming a heading, or renumbering ``known-debt.md``. See META.md.

Usage:
    python3 agent-docs/check-links.py         # exits 1 if anything is broken

External links (http/https/mailto) are not checked.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# [text](target) -- target runs to the first whitespace or closing paren.
_LINK = re.compile(r'\[(?:[^\]]*)\]\(([^)\s]+)\)')
_HEADING = re.compile(r'^(#{1,6})\s+(.*)$')
_FENCE = re.compile(r'^\s*(```|~~~)')


def slugify(heading):
    """Convert heading text to a GitHub anchor slug.

    Mirrors github-slugger: render the markdown to plain text, lowercase,
    drop punctuation, then replace spaces with hyphens.

    Note that underscores are *kept*. GitHub would treat ``_foo_`` as emphasis
    and drop them, but every underscore in these docs is part of an identifier
    (``gaugeopt_suite``, ``other_unconstrained``), so keeping them is correct
    here. Stripping them silently mismatches every such anchor.
    """
    s = heading.strip()
    s = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', s)  # unwrap [text](url)
    s = s.replace('`', '')                          # drop code ticks
    s = re.sub(r'[*~]', '', s)                      # drop bold/italic/strike
    s = s.lower()
    s = re.sub(r'[^\w\- ]', '', s, flags=re.UNICODE)
    return s.replace(' ', '-')


def anchors_of(path):
    """Return the set of anchor slugs defined by headings in a markdown file."""
    seen = {}
    out = set()
    in_fence = False
    for line in path.read_text(encoding='utf-8').splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING.match(line)
        if not m:
            continue
        base = slugify(m.group(2))
        # GitHub disambiguates repeated headings with -1, -2, ...
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.add(base if n == 0 else '{}-{}'.format(base, n))
    return out


def main():
    docs = [REPO / 'AGENTS.md'] + sorted((REPO / 'agent-docs').glob('**/*.md'))
    anchor_cache = {}
    broken = []
    n_links = 0

    for doc in docs:
        if not doc.exists():
            broken.append((doc, 0, str(doc), 'MISSING DOC'))
            continue
        for lineno, line in enumerate(doc.read_text(encoding='utf-8').splitlines(), 1):
            for target in _LINK.findall(line):
                if target.startswith(('http://', 'https://', 'mailto:', '#!')):
                    continue
                n_links += 1
                path_part, _, anchor = target.partition('#')
                dest = doc if not path_part else (doc.parent / path_part).resolve()

                if not dest.exists():
                    broken.append((doc, lineno, target, 'DEAD PATH'))
                    continue
                if anchor and dest.suffix == '.md':
                    if dest not in anchor_cache:
                        anchor_cache[dest] = anchors_of(dest)
                    if anchor not in anchor_cache[dest]:
                        broken.append((doc, lineno, target, 'DEAD ANCHOR'))

    print('checked {} relative links across {} docs'.format(n_links, len(docs)))
    if not broken:
        print('OK -- all links and anchors resolve')
        return 0

    print('\n{} broken:'.format(len(broken)))
    for doc, lineno, target, kind in broken:
        rel = doc.relative_to(REPO) if doc.is_relative_to(REPO) else doc
        print('  {:12s} {}:{}  ->  {}'.format(kind, rel, lineno, target))
    return 1


if __name__ == '__main__':
    sys.exit(main())
