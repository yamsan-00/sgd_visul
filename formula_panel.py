"""
Formula / equation display panel.

Renders optimisation update-rules and the selected test-function formula
using matplotlib's mathtext renderer inside a compact tkinter-embedded
figure.
"""

import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .theme import COLORS, MPL_BG, MPL_TEXT


# =====================================================================
#  LaTeX strings for the two optimisers (static)
# =====================================================================

VANILLA_FORMULA = (
    r"$\theta_{t+1} = \theta_t \;-\; \eta \;\nabla J(\theta_t)$"
)

MOMENTUM_FORMULA = (
    r"$v_t = \gamma \, v_{t-1} + \eta \;\nabla J(\theta_t)$"
    "\n\n"
    r"$\theta_{t+1} = \theta_t \;-\; v_t$"
)


# =====================================================================
#  Public widget
# =====================================================================

class FormulaPanel(tk.Frame):
    """
    Tkinter frame that shows three cards:

    1. Selected test-function  (f and grad)
    2. Vanilla SGD update rule
    3. Momentum SGD update rule

    Call ``update_function(name, info_dict)`` whenever the user changes
    the test-function dropdown.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS["bg_dark"], **kwargs)
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        # Title
        tk.Label(
            self, text="  Algorithm Formulas  ",
            bg=COLORS["bg_dark"], fg=COLORS["accent"],
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=10, pady=(8, 2))

        # Container for the matplotlib figure
        self.fig, self.axes = plt.subplots(
            1, 3, figsize=(14, 1.55),
            facecolor=MPL_BG,
        )
        self.fig.subplots_adjust(
            left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.18
        )

        # -- Card backgrounds + titles --
        card_bg = "#16213e"
        cards = [
            (self.axes[0], "Objective Function", COLORS["accent"]),
            (self.axes[1], "Vanilla SGD",        COLORS["vanilla"]),
            (self.axes[2], "Momentum SGD",       COLORS["momentum"]),
        ]
        self.ax_func, self.ax_vanilla, self.ax_momentum = (
            self.axes[0], self.axes[1], self.axes[2]
        )

        for ax, title, colour in cards:
            ax.set_facecolor(card_bg)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(colour)
                spine.set_linewidth(1.4)
            ax.set_title(title, fontsize=9, fontweight="bold",
                         color=colour, pad=6)

        # Default: placeholder text
        self._func_artists = []
        self._update_func_card("Quadratic Bowl", r"$f(\mathbf{x}) = x_1^2 + x_2^2$",
                                r"$\nabla f = [2x_1,\; 2x_2]$")

        # Optimiser equations (static)
        self.ax_vanilla.text(
            0.5, 0.5, VANILLA_FORMULA,
            transform=self.ax_vanilla.transAxes,
            fontsize=13, ha="center", va="center",
            color=MPL_TEXT,
        )
        self.ax_momentum.text(
            0.5, 0.5, MOMENTUM_FORMULA,
            transform=self.ax_momentum.transAxes,
            fontsize=12, ha="center", va="center",
            color=MPL_TEXT,
        )

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="x", padx=10, pady=(0, 6))
        self.canvas.draw()

    # ------------------------------------------------------------------
    def _update_func_card(self, func_name, formula_tex, grad_tex):
        """Clear the objective-function axis and redraw formula + gradient."""
        ax = self.ax_func
        ax.cla()
        card_bg = "#16213e"
        ax.set_facecolor(card_bg)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLORS["accent"])
            spine.set_linewidth(1.4)
        ax.set_title("Objective Function", fontsize=9, fontweight="bold",
                     color=COLORS["accent"], pad=6)

        # Two-line formula
        ax.text(
            0.08, 0.68, func_name,
            transform=ax.transAxes,
            fontsize=9, ha="left", va="center",
            color=COLORS["accent"], fontweight="bold",
        )
        ax.text(
            0.08, 0.40, formula_tex,
            transform=ax.transAxes,
            fontsize=12, ha="left", va="center",
            color=MPL_TEXT,
        )
        ax.text(
            0.08, 0.08, grad_tex,
            transform=ax.transAxes,
            fontsize=10, ha="left", va="center",
            color=COLORS["text_dim"],
        )

    # ------------------------------------------------------------------
    def update_function(self, func_name, info):
        """
        Refresh the objective-function card.

        Parameters
        ----------
        func_name : str
        info : dict   (entry from FUNCTIONS)
        """
        formula_tex = info.get("formula_tex", "")
        grad_tex = info.get("grad_tex", "")
        self._update_func_card(func_name, formula_tex, grad_tex)
        self.canvas.draw_idle()
