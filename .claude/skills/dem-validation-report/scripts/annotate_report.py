"""
Re-render a DEM validation report with an analyst's commentary.

`generate_report` pickles everything the renderer needs to `*_state.pkl` and
writes the numbers out as `*_brief.md`. Hand the brief to an analyst (the
skill uses a Fable subagent), get back a JSON object following
`dem_report.COMMENTARY_SCHEMA`, and run this. Nothing is recomputed: the
battery, the Monte Carlo and the figures are all reused, so annotating a
report that took ten minutes to produce takes under a second.

    python annotate_report.py \
        --state   OUT/report_state.pkl \
        --commentary OUT/commentary.json \
        --output  OUT/report.html

Repeatable: run it again with a revised commentary and the report is
rewritten from the same state.
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dem_report import _render_html  # noqa: E402


def load_commentary(path: Path) -> dict:
    """
    Parse the analyst's JSON, tolerating a ```json fence around it.

    A model asked for "a single JSON object" often wraps it anyway; failing
    the run over that would be silly.
    """
    text = path.read_text().strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    commentary = json.loads(text)
    if not isinstance(commentary, dict):
        raise ValueError(f"{path}: expected a JSON object, got "
                         f"{type(commentary).__name__}")
    unknown = set(commentary) - {"summary", "sections", "reading", "caveats"}
    if unknown:
        print(f"warning: ignoring unknown commentary keys {sorted(unknown)}")
    return commentary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, type=Path,
                    help="the *_state.pkl written by generate_report")
    ap.add_argument("--commentary", required=True, type=Path,
                    help="JSON following dem_report.COMMENTARY_SCHEMA")
    ap.add_argument("--output", type=Path,
                    help="where to write the HTML (default: the path the "
                         "original report was written to)")
    ap.add_argument("--title", help="override the report title")
    ap.add_argument("--subtitle", help="override the report subtitle")
    args = ap.parse_args(argv)

    with open(args.state, "rb") as f:
        state = pickle.load(f)
    state.cfg.commentary = load_commentary(args.commentary)
    if args.title:
        state.cfg.title = args.title
    if args.subtitle:
        state.cfg.subtitle = args.subtitle

    out = Path(args.output or state.cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_html(state))

    com = state.cfg.commentary
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"  summary: {'yes' if com.get('summary') else 'MISSING'}; "
          f"sections: {', '.join(sorted(com.get('sections') or {})) or 'none'}; "
          f"reading: {len(com.get('reading') or [])} bullets; "
          f"caveats: {len(com.get('caveats') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
