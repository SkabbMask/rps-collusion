"""Draw the game setup figure: rules, one round with an assisting pair, and 6-player coalitions.

Run: python scripts/game_setup_figure.py  (writes docs/figures/game_setup.{pdf,png,svg})
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
BLUE, ORANGE, GREY, RED, DARK = "#0072B2", "#D55E00", "#9aa0a6", "#C0392B", "#222222"
MOVE_COLOR = {"rock": "#4a4a4a", "paper": "#8E6BB8", "scissors": "#2A9D8F"}


def arrow(ax, a, b, color=DARK, style="-|>", ls="-", lw=1.6, rad=0.0, z=2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z,
                                 shrinkA=0, shrinkB=0))


def box(ax, xy, w, h, text, fc="white", ec=DARK, size=9.5, weight="normal", ls="-", align="center"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec,
                                lw=1.2, ls=ls, zorder=1))
    x = xy[0] + (w / 2 if align == "center" else 0.04)
    ax.text(x, xy[1] + h / 2, text, ha=align, va="center", fontsize=size, weight=weight, zorder=3,
            linespacing=1.35)


def rules(ax):
    ax.set_title("(a) Rules", loc="left", fontsize=12, weight="bold")
    centre, r = (0.31, 0.84), 0.135
    pos = {"rock": (centre[0], centre[1] + r), "scissors": (centre[0] + r * 0.87, centre[1] - r * 0.5),
           "paper": (centre[0] - r * 0.87, centre[1] - r * 0.5)}
    for move, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.062, fc=MOVE_COLOR[move], ec="white", lw=1.5, zorder=3))
        ax.text(x, y, move, ha="center", va="center", color="white", fontsize=7.5, weight="bold", zorder=4)
    # Each move beats the next one around the cycle.
    for a, b in (("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        dx, dy = x2 - x1, y2 - y1
        d = (dx * dx + dy * dy) ** 0.5
        s = 0.072 / d
        arrow(ax, (x1 + dx * s, y1 + dy * s), (x2 - dx * s, y2 - dy * s), rad=-0.25, lw=1.3)
    ax.text(centre[0], centre[1] - 0.015, "beats", ha="center", va="center", fontsize=8, style="italic", color=GREY)

    ax.text(0.01, 0.665, "Scoring, any number of players", fontsize=9, weight="bold", va="center")
    # Table: header row, then one row per outcome of a round.
    left, right, split, top, header, row = 0.01, 0.61, 0.37, 0.625, 0.06, 0.105
    bottom = top - header - 3 * row
    ax.add_patch(FancyBboxPatch((left, top - header), right - left, header, boxstyle="square,pad=0",
                                fc="#e9ecf1", ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((left, bottom), right - left, top - bottom, boxstyle="square,pad=0",
                                fc="none", ec="#9aa3ae", lw=1.0, zorder=2))
    ax.plot([split, split], [bottom, top], color="#9aa3ae", lw=0.8)
    ax.text(left + 0.02, top - header / 2, "Moves in a round", fontsize=8.5, weight="bold", va="center")
    ax.text(split + 0.02, top - header / 2, "Score", fontsize=8.5, weight="bold", va="center")
    rows = (("everyone plays\nthe same move", "everyone 0"),
            ("rock, paper and\nscissors all played", "everyone 1"),
            ("exactly two of\nthe three played", "winners 1,\nothers 0"))
    for i, (case, score) in enumerate(rows):
        y_top = top - header - i * row
        ax.plot([left, right], [y_top, y_top], color="#9aa3ae", lw=0.8)
        ax.text(left + 0.02, y_top - row / 2, case, fontsize=8.3, va="center", linespacing=1.2)
        ax.text(split + 0.02, y_top - row / 2, score, fontsize=8.3, va="center", linespacing=1.2)
    ax.text(0.02, 0.2, "Examples, 6 players:", fontsize=8.5, weight="bold", va="top")
    ax.text(0.02, 0.145, "rock ×2, scissors ×4\n→ rock players score 1, the rest 0",
            fontsize=8.5, va="top", linespacing=1.3)
    ax.text(0.02, 0.04, "rock ×3, paper ×2, scissors ×1\n→ all three played: everyone scores 1",
            fontsize=8.5, va="top", linespacing=1.3)


def mini_ring(ax, centre, roles, links):
    """Six players on a ring; links are primary -> assistant signal routes."""
    names = "ABCDEF"
    pos = {}
    for i, name in enumerate(names):
        angle = math.pi / 2 - i * math.pi / 3
        pos[name] = (centre[0] + 0.1 * math.cos(angle), centre[1] + 0.1 * math.sin(angle))
    colors = {"P": BLUE, "A": ORANGE, "H": GREY}
    for primary, assistant in links:
        (x1, y1), (x2, y2) = pos[primary], pos[assistant]
        dx, dy = x2 - x1, y2 - y1
        d = (dx * dx + dy * dy) ** 0.5
        s = 0.031 / d
        arrow(ax, (x1 + dx * s, y1 + dy * s), (x2 - dx * s, y2 - dy * s), color=RED, ls="--", lw=1.4)
    for name in names:
        ax.add_patch(Circle(pos[name], 0.025, fc=colors[roles.get(name, "H")], ec="white", lw=1.5, zorder=3))
        ax.text(*pos[name], name, ha="center", va="center", color="white", fontsize=7, weight="bold", zorder=4)


def coalitions(ax):
    ax.set_title("(c) Coalitions with 6 players", loc="left", fontsize=12, weight="bold")
    cases = [
        ("Pair", "1 primary, 1 assistant,\n4 honest", {"A": "P", "B": "A"}, [("A", "B")]),
        ("Star of three", "1 primary, 2 assistants;\nassistants not linked", {"A": "P", "B": "A", "F": "A"},
         [("A", "B"), ("A", "F")]),
        ("Two pairs", "two separate coalitions;\nboth must be found", {"A": "P", "B": "A", "D": "P", "E": "A"},
         [("A", "B"), ("D", "E")]),
    ]
    for i, (title, text, roles, links) in enumerate(cases):
        cy = 0.86 - i * 0.31
        mini_ring(ax, (0.15, cy), roles, links)
        ax.text(0.33, cy + 0.045, title, fontsize=9, weight="bold", va="center")
        ax.text(0.33, cy - 0.005, text, fontsize=8.3, va="top", linespacing=1.3)
    # Legend for roles and signal routes.
    ly = 0.06
    for j, (color, label) in enumerate(((BLUE, "primary"), (ORANGE, "assistant"), (GREY, "honest"))):
        lx = 0.03 + j * 0.25
        ax.add_patch(Circle((lx, ly), 0.018, fc=color, ec="none"))
        ax.text(lx + 0.03, ly, label, fontsize=8, va="center")
    arrow(ax, (0.03, ly - 0.06), (0.11, ly - 0.06), color=RED, ls="--", lw=1.4)
    ax.text(0.13, ly - 0.06, "private signal route (via the engine)", fontsize=8, va="center")
    ax.text(0.02, -0.1, "Honest games: all six players honest.", fontsize=8.3, va="center", style="italic")


def player(ax, xy, name, role, prompt, color, move):
    ax.add_patch(Circle(xy, 0.075, fc=color, ec="white", lw=2, zorder=3))
    ax.text(*xy, name, ha="center", va="center", color="white", fontsize=16, weight="bold", zorder=4)
    ax.text(xy[0], xy[1] + 0.105, role, ha="center", va="bottom", fontsize=10, weight="bold", color=color)
    ax.text(xy[0], xy[1] - 0.1, prompt, ha="center", va="top", fontsize=8.5, color=DARK, linespacing=1.3)
    mx, my = xy[0], xy[1] - 0.265
    ax.add_patch(FancyBboxPatch((mx - 0.07, my - 0.03), 0.14, 0.06, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc=MOVE_COLOR[move], ec="none", zorder=3))
    ax.text(mx, my, move, ha="center", va="center", color="white", fontsize=9.5, weight="bold", zorder=4)
    return mx, my


def round_panel(ax):
    ax.set_title("(b) One round, 3 players", loc="left", fontsize=12, weight="bold")
    xs = {"B": 0.17, "A": 0.5, "C": 0.83}
    y = 0.72
    moves = {}
    moves["B"] = player(ax, (xs["B"], y), "B", "Primary", "honest prompt;\nunaware of the deal", BLUE, "rock")
    moves["A"] = player(ax, (xs["A"], y), "A", "Assistant", "told: secret deal to\nhelp B; never mention it", ORANGE, "scissors")
    moves["C"] = player(ax, (xs["C"], y), "C", "Honest", "honest prompt", GREY, "scissors")

    # The game engine, not B, leaks B's move to A with probability CP.
    box(ax, (0.3, 0.955), 0.4, 0.07, "Game engine", fc="#fdecea", ec=RED, size=9.5, weight="bold")
    arrow(ax, (xs["B"], y + 0.165), (0.3, 0.99), color=RED, ls="--", rad=-0.35)
    arrow(ax, (0.5, 0.955), (0.5, y + 0.175), color=RED, ls="--")
    ax.text(0.53, 0.885, "private signal with prob. CP:\n\"B will play rock\"", ha="left", va="center",
            fontsize=8.5, color=RED, style="italic")
    ax.text(0.02, 1.02, "B's chosen move", ha="left", va="center", fontsize=8.5, color=RED, style="italic")

    # Outcome of this example round.
    by = 0.31
    box(ax, (0.06, by - 0.055), 0.88, 0.11,
        "Moves: rock (B), scissors (A, C)  →  rock wins\nPayoffs: B = 1,  A = 0,  C = 0", fc="#f5f6f8", ec="#c9ced6",
        size=9)
    for name in "BAC":
        arrow(ax, (moves[name][0], moves[name][1] - 0.035), (moves[name][0], by + 0.06), color=GREY, lw=1.1)

    # What the auditor works with, collected over every round of the game.
    box(ax, (0.06, -0.02), 0.88, 0.215, "", fc="white", ec=DARK, ls="--")
    ax.text(0.5, 0.165, "Auditor, after all rounds", ha="center", va="center", fontsize=9.5, weight="bold")
    green = "#2E7D32"
    ax.text(0.1, 0.115, "✓ Observed", fontsize=9, weight="bold", color=green, va="center")
    ax.text(0.1, 0.07, "every player's moves\npublic messages", fontsize=8.5, va="top", linespacing=1.35)
    ax.text(0.53, 0.115, "✗ Hidden", fontsize=9, weight="bold", color=RED, va="center")
    ax.text(0.53, 0.07, "prompts and roles\nprivate signals", fontsize=8.5, va="top", linespacing=1.35)
    ax.text(0.5, -0.06, "Detector asks: do some players' moves depend on each other\n"
            "more than in honest games?", ha="center", va="top", fontsize=8.8, style="italic")
    arrow(ax, (0.5, by - 0.06), (0.5, 0.2), color=DARK, lw=1.1)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    # Panels share one data scale (inches per unit) so circles stay round and text sizes match.
    ylim, unit, gap = (-0.16, 1.08), 4.3, 0.35
    widths = {"a": 0.62, "b": 1.0, "c": 0.78}
    height = (ylim[1] - ylim[0]) * unit
    total = sum(widths.values()) * unit + gap * (len(widths) + 1)
    fig = plt.figure(figsize=(total, height + 0.5))
    x = gap
    for key, draw in (("a", rules), ("b", round_panel), ("c", coalitions)):
        w = widths[key] * unit
        ax = fig.add_axes([x / total, 0.2 / (height + 0.5), w / total, height / (height + 0.5)])
        ax.set_xlim(0, widths[key])
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        draw(ax)
        x += w + gap
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"game_setup.{ext}", dpi=200, bbox_inches="tight")
    print(OUT / "game_setup.png")


if __name__ == "__main__":
    main()
