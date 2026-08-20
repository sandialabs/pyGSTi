# META — maintaining the agent docs

**Read this before editing [AGENTS.md](../AGENTS.md) or anything in `agent-docs/`.**

This file is about the docs, not about pyGSTi. Everything below exists because these
docs have a specific and unusual correctness contract, and because a handful of edits
that look local actually break things several files away.

---

## 0. Don't edit these docs unless that is the task

AGENTS.md instructs every agent working in this repo: **when the docs contradict the
code, the code is authoritative — flag the discrepancy, don't silently fix it.** That
instruction is deliberate, and it applies to you too. Editing docs is a job you were
given, not a side effect of finishing some other job.

If you were not asked to update docs, you are at [Level 1](#level-1--you-noticed-something-wrong-while-doing-other-work).

## 1. The bar is "not overtly wrong"

These docs are a **hint system to reduce trial-and-error, not a specification.** They
are maintained at a "not overtly wrong" bar, never at a "perfectly accurate" one. This
is the single most important thing to internalize, because it decides what is a defect
and what is just texture.

| Fix it — overtly wrong | Leave it — within the bar |
|---|---|
| A link to a file that no longer exists | A line-number anchor that drifted within the right file |
| A class, function, or kwarg named that doesn't exist (`FreeDataSet` → `FreeformDataSet`) | A description that is vague, terse, or incomplete |
| A claim that contradicts what the code does ("templates are *not* Jinja2" — they are) | A section that could be organized better |
| A sentence that is truncated or self-contradictory | An example that is simplified but not misleading |
| Debt described as "in flight" that has actually landed | Debt whose tracker issue is stale but whose substance holds |
| A whole subpackage no doc mentions or assigns | An individual new module that fits an existing description |
| "Do X" advice that would now break | Prose whose emphasis you'd have chosen differently |

When uncertain, ask: **would an agent following this end up somewhere wrong, or just
somewhere incomplete?** Only the first is a defect. Incompleteness is the normal state
of this doc set and is not worth your tokens.

## 2. Levels

### Level 1 — you noticed something wrong while doing other work

**Report, don't fix.** Tell the user, and be specific enough to act on: name the file,
the line, the claim, and the code path that contradicts it. Then carry on with your
actual task. Do not detour into a doc edit, and do not work around the doc silently.

### Level 2 — you were asked to fix a specific claim

1. **Read the contradicting code first.** Never edit a doc from memory or from a
   plausible-sounding inference. If you cannot cite `file:line` for why the doc is
   wrong, you are not ready to edit.
2. **Make the minimal diff.** Correct the wrong clause; don't rewrite the paragraph
   around it, don't "improve" neighbouring prose, don't restructure the section.
3. **Prefer deleting stale content over annotating it.** A removed sentence costs a
   reader nothing. A sentence hedged with "(this may have changed)" costs them a
   detour.
4. **Run the checker** (§4) before you report done.

### Level 3 — you were asked to audit or refresh the docs

This is the level with a cost trap. **The entire doc set is ~1,800 lines.** A refresh
is a small job. Do not fan out a large fleet of subagents across it — the output is
overwhelmingly line-number rot that you will then correctly throw away, and you will
have spent more effort on the audit than the docs are worth.

Work in this order, cheapest signal first:

1. **Run the checker.** It finds the entire mechanical class — dead paths, dead
   anchors — in under a second, and dead paths are the highest-yield drift signal
   there is. A dead path usually means a module moved, and a module that moved usually
   means a paragraph is now wrong.

2. **Diff the file inventory.** One command finds subsystem-level drift that reading
   prose never will:

   ```bash
   # what appeared and disappeared since the docs were written
   git log --oneline --reverse --diff-filter=A -- agent-docs/    # find the baseline commit
   git diff --name-status --diff-filter=AD <baseline>..HEAD -- pygsti/
   ```

   A deleted file that the docs still reference, or a new top-level subpackage no doc
   lists, is a real finding. This is how `tools/leakage.py` → `pygsti/leakage/` and the
   new `extras/ml/` were caught.

3. **Scope by churn.** `git log --oneline <baseline>..HEAD -- pygsti/<subpkg>/` tells
   you which docs are even worth re-reading. Untouched subsystems have not drifted.

4. **Only then read prose**, and only for the subsystems step 3 flagged.

Triage every candidate against the §1 table before you touch it. **Do not do a
line-number sweep** — see §3.

### Level 4 — you are changing the structure

Adding a doc, removing one, renaming a heading, moving a file, or renumbering
anything. Read §3 in full first; this is where the silent breakage lives.

- **Adding or removing a doc?** Update the *Layout of `agent-docs/`* table in
  [AGENTS.md](../AGENTS.md) in the same commit. That table is how agents find these
  files at all.
- **Moving a file?** Fix links in *both* directions. Outbound links from the moved file
  and inbound links to it live at different relative depths (§3), and only the checker
  will tell you which you missed.

## 3. Invariants that break silently

### Never renumber `known-debt.md`

Entries are `## 1.`, `## 2.`, … and roughly twenty links across eight files anchor on
those ordinals (`known-debt.md#12-gaugeopt_suite-representation-duality`). Deleting or
reordering an entry silently breaks every inbound link to everything below it.

**When an item is resolved, mark it resolved in place and keep its number:**

```markdown
## 2. RESOLVED — `tools/leakage.py` → `pygsti.leakage` move

**Status.** Done. <what landed, and where the code lives now>

*(Heading number retained deliberately — other agent-docs deep-link this anchor.)*
```

New entries append at the end. Numbers are permanent identifiers, not an ordering.

> If you ever *do* want to reclaim the numbering, the durable fix is to drop the
> ordinals from the headings entirely so anchors key on the name
> (`#gaugeopt_suite-representation-duality`). That is a ~20-link change and needs to be
> a deliberate, checker-verified commit of its own — not a side effect.

### Line numbers in anchors rot, and that's tolerable

Links like `gst.py#L857` drift constantly as code moves. **Do not sweep them.** A reader
who lands 40 lines from the class still lands in the right file and finds it in
seconds; the churn of re-verifying dozens of anchors is not repaid.

The one exception worth fixing: an anchor that lands on a **different class or function
than the one named**, which actively misleads. Fix those; leave the merely-drifted ones.

Corollary for new content: prefer linking a file or a symbol over a line number.

### Relative link depth differs by directory

| Link from | To pyGSTi source | To root AGENTS.md | To a top-level agent-doc |
|---|---|---|---|
| `AGENTS.md` (repo root) | `pygsti/...` | — | `agent-docs/01-...md` |
| `agent-docs/*.md` | `../pygsti/...` | `../AGENTS.md` | `01-...md` |
| `agent-docs/04-orchestration/*.md` | `../../pygsti/...` | `../../AGENTS.md` | `../01-...md` |

Getting this wrong produces a link that renders fine and resolves to nothing. The
checker is the only thing that catches it.

### AGENTS.md should account for every top-level subpackage

Its *Layout* table maps `pygsti/` subpackages to docs. When a new top-level subpackage
appears, it needs a home in that table — otherwise an agent working there gets no
orientation at all and never learns these docs exist.

## 4. The checker

```bash
python3 agent-docs/check-links.py     # exits 1 if anything is broken
```

Verifies that every relative link resolves to a real file and every `#anchor` on a
`.md` link matches a real heading. It is fast, has no dependencies, and catches the
entire class of breakage that code review reliably misses. **Run it before reporting
any doc work complete.** External (http) links are not checked.

## 5. House style

- **Mermaid** for diagrams (rendered by GitHub; see the note at the top of AGENTS.md).
- Link the first mention of a class or module; don't re-link every occurrence.
- Each subsystem doc follows roughly: *Covers* → *What lives here* → *Mental model* →
  *Key abstractions* → *Cross-subpackage relationships* → *Pitfalls and gotchas* →
  *Architectural debt* → *Canonical examples*. Match it when adding a doc; don't
  retrofit docs that already deviate.
- Numbered filenames (`01-`…`08-`) are **stable identifiers for referencing, not a
  reading order**, and not a ranking. Don't renumber them for the same reason you don't
  renumber known-debt entries.
- Write for someone who will read exactly one of these files and then start editing
  code. Front-load the thing that will bite them.

## 6. Before you say you're done

- [ ] Every claim I changed, I verified against code I actually read.
- [ ] `python3 agent-docs/check-links.py` passes.
- [ ] I didn't renumber `known-debt.md`.
- [ ] I didn't sweep line-number anchors.
- [ ] If I added or removed a doc, the AGENTS.md *Layout* table reflects it.
- [ ] Anything I found but deliberately did **not** change, I reported to the user.
