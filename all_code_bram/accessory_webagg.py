"""Spatial model of fungi with a core and an (optional) accessory chromosome, with a Cacatoo-like control panel.

Every site of an L x L grid (wrapping around the edges) holds one plant and at most one fungus.
A fungus has three yes/no properties:
    core   : the gene is on the core chromosome
    acc    : the gene is on the accessory chromosome
    chrom  : it has an accessory chromosome (empty or carrying the gene)

Simulation steps (every generation, for every site in turn):
    1. plants   : the plant at a site grows logistically with an intrinsic growth rate (rho), and is damaged (kappa)
                  if the fungus at that same site carries the gene
    2. colonise : an empty site looks at one random neighbour. If it is a fungus, it colonises the site with
                  probability b * W. W is its growth factor that differs per fungal strain like in the grind model.
                  The newborn copies the parent.
    3. transfer : a fungus without an accessory chromosome looks at one random neighbour. If that one has a
                  chromosome, it gets a copy with probability beta (with or without the gene!)
    4. events   : death (d0); loss of the accessory chromosome and its gene (lam); loss of a gene copy (mu);
                  translocation of the gene, cut and paste: core -> accessory (tau1), accessory -> core (tau2)

Live figure: shown in the browser (matplotlib's WebAgg backend) and updated with FuncAnimation.
Run it from the terminal, then open the URL that is printed (http://127.0.0.1:8988). Stop with Ctrl-C.

Controls (right-hand side of the figure):
    sliders        change the parameters while the model runs (the log-scale ones show the real value)
    gens/frame     how many generations are computed between two redraws (speed)
    pause / step   pause (also: SPACE, click on the figure first) and advance one generation
    reset          start again from the same seed;  new seed: start again from a random seed
    mix            switch between neighbours on the grid and perfect mixing
    defaults       put all sliders back to the values in SETTINGS
    brush          choose what a click / drag on the grid does (left of the pictures: the kind grid or the plant grid):
                       inspect      click a cell: shows what is in it
                       N0 ... NCA   paint that kind of fungus
                       empty        remove the fungi (a disturbance)
                       heal / kill  set the plants to healthy / dead
                   brush radius sets the size of the brush (in cells)
    scatter        put the chosen kind on 2% of random sites (an invasion)
    disturb 50%    empty half of the sites at random
    heal plants    make all plants healthy
    export csv    save the time series as a csv file (in the folder you started the program from)
                  (to save the picture, use the save button in the matplotlib toolbar of the browser page)
"""
import itertools
import random
import numpy as np

try:
    from numba import njit
except ImportError:                      # no numba installed: still runs, just slowly
    print("numba not found: running in plain Python (slow). Install it with: pip install numba")

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

# ======= SETTINGS (start values: most can be changed with the sliders while the model runs) =======
L = 128                # grid is L x L
T = 50000              # generations
SEED = 2

b = 0.8                # colonisation rate of empty sites
d = 0.1                # death rate

beta = 0.035           # transfer rate of the accessory chromosome
c = 0.0                # growth cost of carrying an accessory chromosome
lam = 0.005            # loss of the accessory chromosome
mu = 4e-5              # loss of a gene copy (from either chromosome)
tau1 = 1e-4            # translocation core -> accessory
tau2 = 1e-4            # translocation accessory -> core

v = 1.0                # effector gene: gain on a healthy plant
gc = 0.2               # effector gene: fixed cost

rho = 0.005            # plant regrowth
kappa = 0.05           # plant damage when the fungus on the same site carries the gene

# start: initial frequencies of each kind of fungus (share of all sites, placed at random).
START = {"N0": 0.1, "NC": 0.1, "NE": 0.1, "NA": 0.1, "NCE": 0.1, "NCA": 0.1}
P0 = 1.0               # start health of the plants

PRINT_EVERY = 100      # print a line to the terminal every this many generations
GENS_PER_FRAME = 10    # start value of the speed slider
MIX = False            # start value of the mix button
PLOT = True            # live figure (needs matplotlib); False = only print
BACKEND = "WebAgg"     # "WebAgg" = in the browser; "QtAgg" or "TkAgg" = native window
FIGSIZE = (15, 7)       # width, height of the figure in inches (make the height smaller if it does not fit your screen)
# =========================================================================================================================

# NOTE: numba compiles step() once and freezes L (the grid size) into it, so L can only be changed by editing this
# file and restarting. All other parameters are passed to step() as an array (prm) and a flag (mix), so they can
# be changed while the model runs.
NBS = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])

# For the brushes: (core, acc, chrom) of each kind of fungus
STRAINS = {"N0": (0, 0, 0), "NC": (1, 0, 0), "NE": (0, 0, 1), "NA": (0, 1, 1), "NCE": (1, 0, 1), "NCA": (1, 1, 1)}
KIND_NAMES = ["empty"] + list(STRAINS)
COLORS = ["#8C8C8C", "#1E9BE8", "#F59BBB", "#E03131", "#C2A8EE", "#7A1590", "#2E8B57"]


def params_from_settings():
    """The parameters as one array, in the order that step() reads them."""
    return np.array([b, d, beta, c, lam, mu, tau1, tau2, v, gc, rho, kappa], dtype=np.float64)


@njit(cache=True)
def seed_numba(seed):
    """numba has its own random number generator, separate from Python's, so it is seeded separately."""
    np.random.seed(seed)


@njit(cache=True)
def random_neighbour(i, j, mix):
    """Row and column of one random neighbour of site (i, j): one of the 8 sites around it (the grid wraps around
    at the edges). With mix = True the 'neighbour' is a random site anywhere on the grid (perfect mixing)."""
    if mix:
        return np.random.randint(0, L), np.random.randint(0, L)
    k = np.random.randint(0, 8)
    return (i + NBS[k, 0]) % L, (j + NBS[k, 1]) % L


def start_state():
    occ = np.zeros((L, L), bool)
    core = np.zeros((L, L), bool)
    acc = np.zeros((L, L), bool)
    chrom = np.zeros((L, L), bool)
    for i in range(L):
        for j in range(L):
            occ[i, j] = True
            choice = random.choices(list(START), weights=START.values(), k=1)[0]
            core[i, j] = choice in ["NC", "NCE", "NCA"]
            acc[i, j] = choice in ["NA", "NCA"]
            chrom[i, j] = choice in ["NE", "NA", "NCE", "NCA"]
    return occ, core, acc, chrom


def kind_shares(occ, core, acc, chrom):
    """Share of each of the six kinds among the fungi."""
    n = occ.sum()
    if n == 0:                                  # everything is dead (or was wiped out with the brush)
        return [0.0] * 6
    k = [occ & ~core & ~chrom,                  # N0
         occ & core & ~chrom,                   # NC
         occ & ~core & chrom & ~acc,            # NE
         occ & ~core & acc,                     # NA
         occ & core & chrom & ~acc,             # NCE
         occ & core & acc]                      # NCA
    return [x.sum() / n for x in k]


def kind_map(occ, core, acc, chrom):
    """For the picture: 0 empty, 1 N0, 2 NC, 3 NE, 4 NA, 5 NCE, 6 NCA."""
    k = np.zeros((L, L), int)
    k[occ & ~core & ~chrom] = 1
    k[occ & core & ~chrom] = 2
    k[occ & ~core & chrom & ~acc] = 3
    k[occ & ~core & acc] = 4
    k[occ & core & chrom & ~acc] = 5
    k[occ & core & acc] = 6
    return k


def where_is_the_gene(occ, core, acc):
    """% of the gene carriers with the gene on the accessory / on the core (a cell with both counts in both)."""
    carriers = (occ & (core | acc)).sum()  # of all occupied sites, how many have the gene?
    if carriers == 0:
        return 0.0, 0.0
    return 100 * (occ & acc).sum() / carriers, 100 * (occ & core).sum() / carriers


@njit(cache=True)
def step(occ, core, acc, chrom, P, prm, mix):
    """Update the whole grid once. Every site looks at the old state; the result goes in the new_ arrays."""
    b = prm[0]
    d = prm[1]
    beta = prm[2]
    c = prm[3]
    lam = prm[4]
    mu = prm[5]
    tau1 = prm[6]
    tau2 = prm[7]
    v = prm[8]
    gc = prm[9]
    rho = prm[10]
    kappa = prm[11]

    # First make a copy of the current state to update into. If we don't do that, individuals will not be updated 'at the same time',
    # but from the top left to the bottom right, which would introduce a weird bias.
    new_occ, new_core, new_acc, new_chrom = occ.copy(), core.copy(), acc.copy(), chrom.copy()

    for i in range(L):
        for j in range(L):

            # ---- 1. plant at this site
            carrier = occ[i, j] and (core[i, j] or acc[i, j])
            P[i, j] = P[i, j] + rho * (1 - P[i, j])  # Logistic growth
            if carrier:
                P[i, j] = P[i, j] - kappa * P[i, j]  # Killing by fungus on this spot
            P[i, j] = min(max(P[i, j], 0.0), 1.0)

            if not occ[i, j]:
                # ---- 2. empty site: maybe colonised by a random neighbour (the "parent")
                ni, nj = random_neighbour(i, j, mix)
                if occ[ni, nj]:
                    W = 1.0
                    if chrom[ni, nj]:
                        W = W * (1 - c)
                    if core[ni, nj] or acc[ni, nj]:
                        W = W * (1 + v * P[i, j]) * (1 - gc)
                    if np.random.random() < b * W:
                        new_occ[i, j] = True                      # the newborn copies its parent
                        new_core[i, j] = core[ni, nj]
                        new_acc[i, j] = acc[ni, nj]
                        new_chrom[i, j] = chrom[ni, nj]
            else:
                # ---- 3. fungus without the accessory chromosome: maybe gets one from a random neighbour (the "donor")
                if not chrom[i, j]:
                    ni, nj = random_neighbour(i, j, mix)
                    if chrom[ni, nj] and np.random.random() < beta:
                        new_chrom[i, j] = True
                        if acc[ni, nj]:
                            new_acc[i, j] = True                  # the donor's chromosome carries the gene

                # ---- 4. mutational events, note that these happen even during the lifetime of an indvidual (not just at birth)
                if core[i, j] and np.random.random() < mu:           # gene lost from the core
                    new_core[i, j] = False
                if acc[i, j] and np.random.random() < mu:            # gene lost from the accessory chromosome
                    new_acc[i, j] = False
                if core[i, j] and chrom[i, j] and not acc[i, j] and np.random.random() < tau1:
                    new_core[i, j] = False                        # translocation core -> accessory (cut and paste)
                    new_acc[i, j] = True
                if acc[i, j] and not core[i, j] and np.random.random() < tau2:
                    new_acc[i, j] = False                         # translocation accessory -> core
                    new_core[i, j] = True
                if chrom[i, j] and np.random.random() < lam:         # chromosome lost (the gene on it goes too)
                    new_chrom[i, j] = False
                    new_acc[i, j] = False
                if np.random.random() < d:                           # death
                    new_occ[i, j] = False
                    new_core[i, j] = False
                    new_acc[i, j] = False
                    new_chrom[i, j] = False

    return new_occ, new_core, new_acc, new_chrom, P


class Run:
    """Holds the state of the simulation. advance(n) computes n generations; nothing in here knows about plotting,
    so the same object can be used for a live figure, for a headless run, or for saving figures."""

    def __init__(self):
        self.prm = params_from_settings()      # parameters, changed by the sliders
        self.mix = MIX                         # perfect mixing on/off
        self.paused = False
        self.gens_per_frame = GENS_PER_FRAME
        self.seed = SEED
        self.reset(SEED)

    def reset(self, seed=None):
        """Start again (the parameters, mix and pause are kept as they are)."""
        if seed is not None:
            self.seed = seed
        random.seed(self.seed)                 # Python's generator: used only for the start state
        seed_numba(self.seed)                  # numba's generator: used for everything that happens during the run
        self.occ, self.core, self.acc, self.chrom = start_state()
        self.P = np.full((L, L), P0)
        self.t = 0
        self.history = np.zeros((T + 1, 8))    # columns: t, 6 kind shares, mean plant health
        self.finished_reported = False
        self.record()

    def record(self):
        """Store the numbers of the current generation and print a line every PRINT_EVERY generations."""
        t = self.t
        self.history[t] = [t] + kind_shares(self.occ, self.core, self.acc, self.chrom) + [self.P.mean()]
        if t % PRINT_EVERY == 0:
            acc_pct, core_pct = where_is_the_gene(self.occ, self.core, self.acc)
            print(f"t={t:6d}  occupied {self.occ.mean():.2f}  plants {self.P.mean():.2f}  "
                  f"gene on accessory {acc_pct:5.1f}%  on core {core_pct:5.1f}%", flush=True)

    def advance(self, n):
        for _ in range(n):
            if self.t >= T:
                break
            self.occ, self.core, self.acc, self.chrom, self.P = step(
                self.occ, self.core, self.acc, self.chrom, self.P, self.prm, self.mix)
            self.t += 1
            self.record()

    @property
    def done(self):
        return self.t >= T

    def summary(self):
        """Final numbers: average over the last 20% of the run."""
        if self.finished_reported:
            return
        self.finished_reported = True
        hist = self.history[:self.t + 1]
        last = hist[int(len(hist) * 0.8):]
        print("\nshares of the six kinds (last 20% of the run):",
              dict(zip(["N0", "NC", "NE", "NA", "NCE", "NCA"], last[:, 1:7].mean(0).round(3))))
        print("plant health:", last[:, 7].mean().round(3))
        print("at the end: % of gene carriers with the gene on the accessory / on the core:",
              [round(float(x), 1) for x in where_is_the_gene(self.occ, self.core, self.acc)], flush=True)

    # ---- things the user can do to the grid with the mouse / buttons (all work in place on the current arrays)
    def paint(self, brush, i, j, radius):
        """Apply a brush to the cells within `radius` of (i, j) (distances wrap around the edges, like the grid)."""
        ii, jj = np.ogrid[:L, :L]
        di = np.abs(ii - i)
        dj = np.abs(jj - j)
        di = np.minimum(di, L - di)
        dj = np.minimum(dj, L - dj)
        mask = di ** 2 + dj ** 2 <= radius ** 2
        self.apply(brush, mask)

    def apply(self, brush, mask):
        if brush in STRAINS:
            core, acc, chrom = STRAINS[brush]
            self.occ[mask] = True
            self.core[mask] = bool(core)
            self.acc[mask] = bool(acc)
            self.chrom[mask] = bool(chrom)
        elif brush == "empty":
            self.occ[mask] = False
            self.core[mask] = False
            self.acc[mask] = False
            self.chrom[mask] = False
        elif brush == "heal plants":
            self.P[mask] = 1.0
        elif brush == "kill plants":
            self.P[mask] = 0.0


def make_figure(run, plt):
    """Build the figure once. Returns the figure, a dict with the axes, and the drawing functions. The drawing
    functions only update the data of the existing plot objects (no new plots are created: that is what keeps it fast)."""
    from matplotlib.colors import ListedColormap

    names = ["N0", "NC", "NE", "NA", "NCE", "NCA", "plants (mean health)"]
    fig = plt.figure(figsize=FIGSIZE)

    ax_grid = fig.add_axes([0.03, 0.50, 0.22, 0.40])             # the grid, coloured by kind
    picture = ax_grid.imshow(kind_map(run.occ, run.core, run.acc, run.chrom), vmin=0, vmax=6,
                             interpolation="nearest", cmap=ListedColormap(["white"] + COLORS[:6]))
    ax_grid.axis("off")
    ax_plants = fig.add_axes([0.27, 0.50, 0.22, 0.40])           # the plants: dark green = healthy, white = dead
    plant_picture = ax_plants.imshow(run.P, vmin=0, vmax=1, cmap="Greens", interpolation="nearest")
    ax_plants.axis("off")
    ax_plants.set_title("plant health", fontsize=10)
    ax_bar = fig.add_axes([0.55, 0.54, 0.15, 0.33])              # where is the gene?
    bars = ax_bar.bar(["on accessory", "on core"], [0, 0], color=["#E03131", "#1E9BE8"])
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("% of gene carriers")
    ax = fig.add_axes([0.06, 0.08, 0.64, 0.33])                  # shares of the kinds over time
    lines = [ax.plot([], [], color=COLORS[i], lw=2, label=names[i])[0] for i in range(7)]
    ax.set_ylim(0, 1)                                            # the x-axis grows with the run (no fixed limit)
    ax.set_xlabel("generation")
    ax.set_ylabel("share")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    def draw_grids(run):
        picture.set_data(kind_map(run.occ, run.core, run.acc, run.chrom))
        plant_picture.set_data(run.P)
        acc_pct, core_pct = where_is_the_gene(run.occ, run.core, run.acc)
        bars[0].set_height(acc_pct)
        bars[1].set_height(core_pct)
        ax_grid.set_title(f"generation {run.t}", fontsize=10)

    def draw_series(run):
        h = run.history[:run.t + 1]
        for i in range(7):
            lines[i].set_data(h[:, 0], h[:, 1 + i])
        ax.relim()                                               # the x-axis follows the data so far
        ax.autoscale_view(scalex=True, scaley=False)

    def draw_all(run):
        draw_grids(run)
        draw_series(run)

    return fig, {"grid": ax_grid, "plants": ax_plants}, draw_grids, draw_all


def add_controls(fig, axes, run, draw_grids, draw_all):
    """All sliders, buttons and mouse handling. Returns a dict with the widgets (matplotlib forgets widgets that
    nothing refers to, so the caller must keep this dict alive)."""
    from matplotlib.widgets import Button, RadioButtons, Slider

    status_text = fig.text(0.50, 0.965, "", ha="center", fontsize=11)
    info_text = fig.text(0.03, 0.955, "", ha="left", fontsize=9, color="0.25")

    def status():
        if run.done:
            state = "FINISHED - press reset"
        elif run.paused:
            state = "PAUSED"
        else:
            state = "running"
        status_text.set_text(state)

    def say(text):
        info_text.set_text(text)
        fig.canvas.draw_idle()

    controls = {"sliders": [], "buttons": {}}

    # ---------------------------------------------------------------- sliders
    def add_slider(label, y, lo, hi, init, setter, log=False, step=None):
        s = Slider(fig.add_axes([0.80, y, 0.12, 0.02]), label, lo, hi,
                   valinit=np.log10(max(init, 1e-7)) if log else init, valstep=step)
        s.label.set_fontsize(8)
        s.valtext.set_fontsize(8)

        def show(val):                        # log sliders: show the real value, not its logarithm
            s.valtext.set_text(f"{10 ** val:.1e}" if log else f"{val:.3g}")

        def changed(val):
            setter(10 ** val if log else val)
            show(val)

        s.on_changed(changed)
        show(s.val)
        controls["sliders"].append(s)
        return s

    def set_param(index):
        def setter(value):
            run.prm[index] = value
        return setter

    def set_speed(value):
        run.gens_per_frame = int(value)

    brush_radius = {"r": 2}

    def set_radius(value):
        brush_radius["r"] = int(value)

    # (label, index in prm, min, max, log scale?)
    slider_specs = [("b  colonise", 0, 0, 1, False), ("d  death", 1, 0, 0.5, False),
                    ("beta  transfer", 2, 0, 0.2, False), ("c  chrom cost", 3, 0, 0.5, False),
                    ("lam  chrom loss", 4, 0, 0.05, False), ("mu  gene loss", 5, -7, -1, True),
                    ("tau1  core>acc", 6, -7, -1, True), ("tau2  acc>core", 7, -7, -1, True),
                    ("gc  gene cost", 9, 0, 0.5, False), ("kappa  damage", 11, 0, 0.2, False)]
    y = 0.93
    for label, index, lo, hi, log in slider_specs:
        add_slider(label, y, lo, hi, run.prm[index], set_param(index), log=log)
        y -= 0.031
    add_slider("gens / frame", y, 1, 100, run.gens_per_frame, set_speed, step=1)
    y -= 0.031
    add_slider("brush radius", y, 0, 25, brush_radius["r"], set_radius, step=1)
    param_sliders = controls["sliders"][:len(slider_specs)]

    # ---------------------------------------------------------------- brush (radio buttons)
    brushes = ["inspect"] + list(STRAINS) + ["empty", "heal plants", "kill plants"]
    ax_radio = fig.add_axes([0.75, 0.02, 0.12, 0.23])
    ax_radio.set_title("brush", fontsize=9)
    radio = RadioButtons(ax_radio, brushes, active=0)
    for k, label in enumerate(radio.labels):
        label.set_fontsize(8)
        if brushes[k] in STRAINS:
            label.set_color(COLORS[list(STRAINS).index(brushes[k])])
    controls["radio"] = radio

    # ---------------------------------------------------------------- buttons
    def add_button(name, label, col, row, callback):
        bt = Button(fig.add_axes([0.75 + col * 0.115, 0.54 - row * 0.052, 0.105, 0.04]), label)
        bt.label.set_fontsize(8)
        bt.on_clicked(lambda event: callback())
        controls["buttons"][name] = bt
        return bt

    def toggle_pause():
        run.paused = not run.paused
        controls["buttons"]["pause"].label.set_text("resume (space)" if run.paused else "pause (space)")
        status()
        fig.canvas.draw_idle()

    def do_step():
        run.advance(1)
        draw_all(run)
        status()
        fig.canvas.draw_idle()

    def do_reset(seed=None):
        run.reset(seed)
        draw_all(run)
        status()
        say(f"reset, seed {run.seed}")

    def new_seed():
        do_reset(random.randrange(2 ** 31))

    def toggle_mix():
        run.mix = not run.mix
        show_mix()

    def show_mix():
        bt = controls["buttons"]["mix"]
        bt.label.set_text("mix: ON" if run.mix else "mix: OFF")
        bt.color = "#F4B183" if run.mix else "0.85"           # (the button resets its colour to these on hover)
        bt.hovercolor = "#F8CBAD" if run.mix else "0.95"
        bt.ax.set_facecolor(bt.color)
        fig.canvas.draw_idle()

    def defaults():
        for s in param_sliders:
            s.reset()
        say("sliders back to the values in SETTINGS")

    def export_csv():
        name = f"history_seed{run.seed}.csv"
        np.savetxt(name, run.history[:run.t + 1], delimiter=",", comments="",
                   header="t,N0,NC,NE,NA,NCE,NCA,plant_health")
        say(f"saved {name}")

    def redraw_after_change():
        draw_all(run)
        fig.canvas.draw_idle()

    def scatter():
        if radio.value_selected not in STRAINS:
            say("scatter: first choose a kind of fungus as the brush")
            return
        run.apply(radio.value_selected, np.random.random((L, L)) < 0.02)
        say(f"scattered {radio.value_selected} over 2% of the sites")
        redraw_after_change()

    def disturb():
        run.apply("empty", np.random.random((L, L)) < 0.5)
        say("emptied 50% of the sites")
        redraw_after_change()

    def heal_all():
        run.apply("heal plants", np.ones((L, L), bool))
        say("all plants healthy")
        redraw_after_change()

    add_button("pause", "pause (space)", 0, 0, toggle_pause)
    add_button("step", "step", 1, 0, do_step)
    add_button("reset", "reset", 0, 1, do_reset)
    add_button("newseed", "new seed", 1, 1, new_seed)
    add_button("mix", "", 0, 2, toggle_mix)
    add_button("defaults", "defaults", 1, 2, defaults)
    add_button("csv", "export csv", 0, 3, export_csv)
    add_button("scatter", "scatter 2%", 1, 3, scatter)
    add_button("disturb", "disturb 50%", 0, 4, disturb)
    add_button("heal", "heal plants", 1, 4, heal_all)
    show_mix()

    # ---------------------------------------------------------------- keyboard and mouse
    def on_key(event):
        if event.key in (" ", "space"):
            toggle_pause()

    painting = {"on": False}

    def toolbar_busy():                       # the browser toolbar's pan / zoom tools also use the mouse
        toolbar = getattr(fig.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", ""))

    def on_grid(event):
        """Left mouse button on one of the two grids: inspect a cell or paint with the brush."""
        if event.xdata is None or event.ydata is None:
            return
        i = min(max(int(round(event.ydata)), 0), L - 1)
        j = min(max(int(round(event.xdata)), 0), L - 1)
        brush = radio.value_selected
        if brush == "inspect":
            kind = KIND_NAMES[kind_map(run.occ, run.core, run.acc, run.chrom)[i, j]]
            say(f"cell (row {i}, column {j}): fungus {kind}, plant health {run.P[i, j]:.2f}")
        else:
            run.paint(brush, i, j, brush_radius["r"])
            draw_grids(run)
            fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes in (axes["grid"], axes["plants"]) and event.button == 1 and not toolbar_busy():
            painting["on"] = True
            on_grid(event)

    def on_move(event):
        if painting["on"] and event.inaxes in (axes["grid"], axes["plants"]):
            on_grid(event)

    def on_release(event):
        painting["on"] = False

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)

    status()
    say("choose a brush and click or drag on the grids (inspect = click a cell to see what is in it)")
    controls["status"] = status
    return controls


def build_animation(run):
    """Set up the figure and the animation. Importing matplotlib happens here, after the backend is chosen."""
    import matplotlib
    matplotlib.use(BACKEND)                                       # must come before pyplot is imported
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, axes, draw_grids, draw_all = make_figure(run, plt)
    controls = add_controls(fig, axes, run, draw_grids, draw_all)

    def update(frame):
        if run.paused or run.done:                                # nothing to compute (reset starts it again)
            return []
        run.advance(run.gens_per_frame)                           # compute some generations ...
        draw_all(run)                                             # ... then show them
        if run.done:
            run.summary()
            controls["status"]()
        return []

    anim = FuncAnimation(fig, update, frames=itertools.count,     # endless: so that 'reset' can restart a finished run
                         init_func=lambda: draw_all(run),         # draw the start state without advancing the model
                         interval=20, repeat=False, blit=False, cache_frame_data=False)
    anim.controls = controls                                      # keep references: widgets and animations that nothing
    return plt, anim                                              # refers to are garbage collected and stop working


if __name__ == "__main__":
    run = Run()
    if PLOT:
        plt, anim = build_animation(run)
        plt.show()             # with WebAgg this serves the figure at http://127.0.0.1:8988 and blocks until you stop it
    else:
        run.advance(T)
        run.summary()