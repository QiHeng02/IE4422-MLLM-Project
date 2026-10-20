"""Collaboration extension, last step: one group's report figure (talk network + per-student table).

    python scripts/plot_wtd.py --name wtd_lora_human --group 1

Inputs   results/wtd/<name>_students.csv and <name>_edges.csv (from scripts/fuse_wtd.py)
Output   results/wtd/<name>_g01.png   (a figure: keep it out of git, like every image)

Left: who replies to whom. Circle size = share of speaking time; arrow width and number = replies.
Right: the per-student numbers under the AAC&U Teamwork rows they give evidence for.
Wording follows the project rules: evidence for a tutor to review, never a grade or a judgement.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ie4422.config import repo_path  # noqa: E402

POS = {"P1": (0.0, 0.0), "P2": (0.5, 0.8), "P3": (1.0, 0.0)}  # left, middle (drawn on top), right
ROWS = [  # (AAC&U Teamwork row, column in <name>_students.csv, label)
    ("Contributes to team meetings", "speaking_share", "share of speaking time"),
    ("Contributes to team meetings", "turns", "turns"),
    ("Facilitates contributions of others", "replies_to", "replies to others"),
    ("Facilitates contributions of others", "replies_from", "replies received"),
    ("Fosters constructive team climate", "attending_when_silent", "turned to speaker when silent"),
    ("(visual only)", "hand_raising_frames", "frames with hand raised"),
    ("(visual only)", "off_task_share", "share of frames using phone"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="e.g. wtd_lora_human")
    ap.add_argument("--group", required=True, type=int)
    a = ap.parse_args()

    d = repo_path("results/wtd")
    st = pd.read_csv(d / f"{a.name}_students.csv")
    st = st[st.group == a.group].set_index("person")
    ed = pd.read_csv(d / f"{a.name}_edges.csv")
    ed = (ed[ed.group == a.group] if len(ed) else ed).rename(columns={"from": "src"})
    if st.empty:
        raise SystemExit(f"group {a.group} not in {a.name}_students.csv")

    fig, (ax, tx) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    top = max(ed.replies.max(), 1) if len(ed) else 1
    for r in ed.itertuples():
        (x0, y0), (x1, y1) = POS[r.src], POS[r.to]
        # bend the two directions apart so A->B and B->A do not overlap
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1 + 5 * r.replies / top, color="0.35",
                                    shrinkA=28, shrinkB=28, connectionstyle="arc3,rad=0.18"))
        mx, my = (x0 + x1) / 2 + (y1 - y0) * 0.12, (y0 + y1) / 2 - (x1 - x0) * 0.12
        ax.text(mx, my, str(r.replies), ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    for p, (x, y) in POS.items():
        share = st.speaking_share.get(p, 0) or 0
        ax.scatter([x], [y], s=600 + 4000 * share, color="#4C78A8", alpha=0.25, edgecolors="#4C78A8", lw=1.5)
        ax.text(x, y, f"{p}\n{share:.0%}", ha="center", va="center", fontsize=11, weight="bold")
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.35, 1.15)
    ax.axis("off")
    ax.set_title(f"Group {a.group}: who replies to whom\n(circle = share of speaking time)", fontsize=11)

    people = [p for p in POS if p in st.index]
    cells = []
    for row, col, label in ROWS:
        vals = []
        for p in people:
            v = st.loc[p, col]
            vals.append("n/a" if pd.isna(v) else f"{v:.0%}" if "share" in col or col.startswith("attending")
                        else f"{int(v)}")
        cells.append([row, label] + vals)
    tx.axis("off")
    t = tx.table(cellText=cells, colLabels=["AAC&U Teamwork row", "evidence"] + people,
                 loc="center", cellLoc="left", colWidths=[0.36, 0.34] + [0.1] * len(people))
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.5)
    tx.set_title("Evidence for the tutor to review (not a grade)", fontsize=11)

    fig.suptitle(f"{a.name}", fontsize=9, color="0.4", x=0.99, ha="right")
    out = d / f"{a.name}_g{a.group:02d}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
