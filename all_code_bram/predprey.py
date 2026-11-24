#!/usr/bin/env python3
"""
Predator-prey IBM with a heritable on/off gene: interactive explorer.

    pip install numpy matplotlib PyQt6        (PyQt5 / PySide6 also work)
    python3 predprey_gui.py

Model (Rosenzweig-MacArthur, individual-based predators)
  prey      logistic growth r, carrying capacity K (stochastic per-capita events)
  predators ON  : capture a prey with prob f(N) dt,  f(N) = a N / (1 + a h N)   (saturating)
                  each capture -> offspring with prob e; offspring inherit strain and gene state
                  die at rate m + c            (c = cost of carrying the active gene, always paid)
            OFF : cannot capture; die at rate m * (OFF death multiplier)
  strains   resident : gene always ON
            switcher : gene flips ON->OFF at rate s_off and OFF->ON at rate s_on
                       (a stand-in for slip-strand frameshifts; later replaced by the sequence model)
"""
import sys
import numpy as np
from collections import deque

# =============================================================================================
# 1. MODEL (no GUI code in here)
# =============================================================================================
DEFAULTS = dict(r=1.0, K=3000, a=0.002, h=1.0, e=0.5, m=0.2, c=0.15, off_mult=1.0, s_off=0.1, s_on=0.1, dt=0.05)


class Sim:
    REC_EVERY = 10                                     # record every 10 steps (= 0.5 time units)

    def __init__(self, seed=None):
        self.p = dict(DEFAULTS)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, N0=10000, P0=100):
        self.N = N0
        self.on = np.ones(P0, bool)
        self.strain = np.zeros(P0, np.int8)            # 0 resident, 1 switcher
        self.t, self.k = 0.0, 0
        self.hist = deque(maxlen=6000)                 # rows: t, prey, resident, switcher ON, switcher OFF
        self.record()

    def add(self, n, strain):
        self.on = np.concatenate((self.on, np.ones(n, bool)))
        self.strain = np.concatenate((self.strain, np.full(n, strain, np.int8)))

    def clear_switchers(self):
        keep = self.strain == 0
        self.on, self.strain = self.on[keep], self.strain[keep]

    def step(self):
        p, rng, P = self.p, self.rng, self.on.size
        dt, N = p["dt"], self.N
        pc = p["a"] * N / (1 + p["a"] * p["h"] * N) * dt
        cap = self.on & (rng.random(P) < pc)
        ncap = int(cap.sum())
        if ncap > N:                                   # cannot eat more prey than exist
            cap[rng.choice(np.flatnonzero(cap), ncap - N, replace=False)] = False
            ncap = N
        births = rng.binomial(N, min(1.0, p["r"] * dt))
        deaths = rng.binomial(N, min(1.0, p["r"] * N / p["K"] * dt))
        self.N = max(0, N + births - deaths - ncap)
        kids = np.flatnonzero(cap & (rng.random(P) < p["e"]))
        kid_on, kid_strain = self.on[kids], self.strain[kids]
        mort = np.where(self.on, p["m"] + p["c"], p["m"] * p["off_mult"]) * dt
        die = rng.random(P) < mort
        u = rng.random(P)
        sw = self.strain == 1
        flip = sw & np.where(self.on, u < p["s_off"] * dt, u < p["s_on"] * dt)
        on = self.on ^ flip
        keep = ~die
        self.on = np.concatenate((on[keep], kid_on))
        self.strain = np.concatenate((self.strain[keep], kid_strain))
        self.k += 1
        self.t = self.k * dt
        if self.k % self.REC_EVERY == 0:
            self.record()

    def record(self):
        res = int((self.strain == 0).sum())
        son = int(((self.strain == 1) & self.on).sum())
        soff = int(((self.strain == 1) & ~self.on).sum())
        self.hist.append((self.t, self.N, res, son, soff))

    def arrays(self, window):
        H = np.array(self.hist, float)
        return H[H[:, 0] >= self.t - window] if len(H) else H


# =============================================================================================
# 2. PLOTS (matplotlib only: testable without Qt)
# =============================================================================================
THEMES = {
    "light": dict(bg="#f6f7f9", panel="#ffffff", ink="#1d2733", muted="#6b7a8c", grid="#e6eaef", accent="#2f6fd0",
                  prey="#2f9e6e", res="#d9822b", swon="#7a4fd6", swoff="#b8a4ea"),
    "dark": dict(bg="#0f1620", panel="#16212d", ink="#e6edf4", muted="#93a5b6", grid="#243344", accent="#5aa2f0",
                 prey="#49c28f", res="#f0a24a", swon="#a98af0", swoff="#6a5aa3"),
}


class Plots:
    def __init__(self, fig, theme="light"):
        self.fig = fig
        gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.42, wspace=0.22, left=0.07, right=0.985, top=0.93, bottom=0.09)
        self.ax1 = fig.add_subplot(gs[0, :])
        self.ax2 = fig.add_subplot(gs[1, 0])
        self.ax3 = fig.add_subplot(gs[1, 1])
        self.window = 300.0
        self.l_prey, = self.ax1.plot([], [], lw=2)
        self.l_res, = self.ax1.plot([], [], lw=2)
        self.l_son, = self.ax1.plot([], [], lw=2)
        self.l_soff, = self.ax1.plot([], [], lw=1.6, ls="--")
        self.l_ph, = self.ax2.plot([], [], lw=1.2, alpha=0.85)
        self.p_ph, = self.ax2.plot([], [], "o", ms=6)
        self.l_share, = self.ax3.plot([], [], lw=2)
        self.l_onfrac, = self.ax3.plot([], [], lw=1.6, ls="--")
        self.ax1.set_title("Population dynamics", loc="left", fontsize=11, fontweight="semibold")
        self.ax2.set_title("Phase portrait", loc="left", fontsize=11, fontweight="semibold")
        self.ax3.set_title("Switchers: share and state", loc="left", fontsize=11, fontweight="semibold")
        self.ax1.set_xlabel("time"); self.ax1.set_ylabel("individuals")
        self.ax2.set_xlabel("prey"); self.ax2.set_ylabel("predators")
        self.ax3.set_xlabel("time"); self.ax3.set_ylim(-0.02, 1.02)
        self.apply_theme(theme)

    def apply_theme(self, name):
        T = THEMES[name]; self.T = T
        self.fig.set_facecolor(T["bg"])
        for ax in (self.ax1, self.ax2, self.ax3):
            ax.set_facecolor(T["panel"])
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            for s in ("left", "bottom"): ax.spines[s].set_color(T["grid"])
            ax.tick_params(colors=T["muted"], labelsize=8.5, length=3)
            ax.grid(True, color=T["grid"], lw=0.8); ax.set_axisbelow(True)
            ax.xaxis.label.set_color(T["muted"]); ax.yaxis.label.set_color(T["muted"])
            ax.title.set_color(T["ink"])
        for ln, col in ((self.l_prey, "prey"), (self.l_res, "res"), (self.l_son, "swon"), (self.l_soff, "swoff"),
                        (self.l_ph, "muted"), (self.l_share, "swon"), (self.l_onfrac, "swoff")):
            ln.set_color(T[col])
        self.p_ph.set_color(T["accent"])
        self.ax1.legend([self.l_prey, self.l_res, self.l_son, self.l_soff],
                        ["prey", "residents (gene always ON)", "switchers, gene ON", "switchers, gene OFF"],
                        loc="upper left", ncol=4, fontsize=8, frameon=False, labelcolor=T["ink"])
        self.ax3.legend([self.l_share, self.l_onfrac], ["share of predators", "fraction with gene ON"],
                        loc="upper left", fontsize=8, frameon=False, labelcolor=T["ink"])

    def update(self, sim, window):
        H = sim.arrays(window)
        if len(H) < 2: return
        t, prey, res, son, soff = H.T
        sw = son + soff
        tot = res + sw
        self.l_prey.set_data(t, prey); self.l_res.set_data(t, res); self.l_son.set_data(t, son); self.l_soff.set_data(t, soff)
        self.ax1.set_xlim(t[0], max(t[-1], t[0] + 1))
        top = max(prey.max(), tot.max(), 10) * 1.12
        self.ax1.set_ylim(0, top)
        self.l_ph.set_data(prey, tot); self.p_ph.set_data([prey[-1]], [tot[-1]])
        self.ax2.set_xlim(0, max(prey.max(), 10) * 1.1); self.ax2.set_ylim(0, max(tot.max(), 10) * 1.1)
        share = np.divide(sw, tot, out=np.zeros_like(sw), where=tot > 0)
        onf = np.divide(son, sw, out=np.zeros_like(sw), where=sw > 0)
        self.l_share.set_data(t, share); self.l_onfrac.set_data(t, onf)
        self.ax3.set_xlim(t[0], max(t[-1], t[0] + 1))


# =============================================================================================
# 3. GUI (Qt)
# =============================================================================================
def main():
    try:
        from PyQt6 import QtCore, QtWidgets, QtGui
    except ImportError:
        try:
            from PyQt5 import QtCore, QtWidgets, QtGui
        except ImportError:
            from PySide6 import QtCore, QtWidgets, QtGui
    import matplotlib
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    Qt = QtCore.Qt
    HORIZ = Qt.Orientation.Horizontal if hasattr(Qt, "Orientation") else Qt.Horizontal

    def qss(T):
        return f"""
        QWidget {{ background: {T['bg']}; color: {T['ink']}; font-family: 'Inter','Segoe UI','Helvetica Neue',Arial; font-size: 13px; }}
        QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
        QFrame#card {{ background: {T['panel']}; border: 1px solid {T['grid']}; border-radius: 10px; }}
        QLabel#h {{ color: {T['muted']}; font-size: 11px; font-weight: 600; letter-spacing: 1px; background: transparent; }}
        QLabel#kpi {{ font-size: 20px; font-weight: 600; background: transparent; }}
        QLabel#kpil {{ color: {T['muted']}; font-size: 11px; background: transparent; }}
        QLabel#val {{ color: {T['ink']}; font-family: 'SF Mono','Menlo','Consolas',monospace; background: transparent; }}
        QLabel {{ background: transparent; }}
        QPushButton {{ background: {T['panel']}; border: 1px solid {T['grid']}; border-radius: 8px; padding: 7px 12px; font-weight: 500; }}
        QPushButton:hover {{ border-color: {T['accent']}; }}
        QPushButton#primary {{ background: {T['accent']}; border-color: {T['accent']}; color: white; }}
        QSpinBox {{ background: {T['panel']}; border: 1px solid {T['grid']}; border-radius: 6px; padding: 4px 6px; }}
        QSlider::groove:horizontal {{ height: 4px; background: {T['grid']}; border-radius: 2px; }}
        QSlider::sub-page:horizontal {{ background: {T['accent']}; border-radius: 2px; }}
        QSlider::handle:horizontal {{ background: {T['panel']}; border: 2px solid {T['accent']}; width: 12px; height: 12px; margin: -6px 0; border-radius: 8px; }}
        """

    class ParamSlider(QtWidgets.QWidget):
        def __init__(self, label, key, lo, hi, fmt, log=False, on_change=None, tip=""):
            super().__init__()
            self.key, self.lo, self.hi, self.fmt, self.log, self.cb = key, lo, hi, fmt, log, on_change
            lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
            row = QtWidgets.QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0)
            self.name = QtWidgets.QLabel(label); self.val = QtWidgets.QLabel(); self.val.setObjectName("val")
            row.addWidget(self.name); row.addStretch(1); row.addWidget(self.val); lay.addLayout(row)
            self.s = QtWidgets.QSlider(HORIZ); self.s.setRange(0, 1000); lay.addWidget(self.s)
            if tip: self.setToolTip(tip)
            self.s.valueChanged.connect(self._moved)

        def set(self, v):
            x = (np.log(v / self.lo) / np.log(self.hi / self.lo)) if self.log else (v - self.lo) / (self.hi - self.lo)
            self.s.blockSignals(True); self.s.setValue(int(round(1000 * min(max(x, 0), 1)))); self.s.blockSignals(False)
            self.val.setText(self.fmt.format(v))

        def value(self):
            x = self.s.value() / 1000
            return self.lo * (self.hi / self.lo) ** x if self.log else self.lo + x * (self.hi - self.lo)

        def _moved(self):
            v = self.value(); self.val.setText(self.fmt.format(v))
            if self.cb: self.cb(self.key, v)

    def card():
        f = QtWidgets.QFrame(); f.setObjectName("card")
        l = QtWidgets.QVBoxLayout(f); l.setContentsMargins(14, 12, 14, 14); l.setSpacing(9)
        return f, l

    def header(text):
        l = QtWidgets.QLabel(text.upper()); l.setObjectName("h"); return l

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Predator-prey: on/off gene explorer")
            self.resize(1380, 860)
            self.sim = Sim(); self.theme = "light"; self.playing = True; self.frame = 0; self.window_len = 300.0
            central = QtWidgets.QWidget(); self.setCentralWidget(central)
            root = QtWidgets.QHBoxLayout(central); root.setContentsMargins(16, 16, 16, 16); root.setSpacing(16)

            # ---- left: controls ----------------------------------------------------------------
            scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setFixedWidth(330)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff if hasattr(Qt, "ScrollBarPolicy") else Qt.ScrollBarAlwaysOff)
            side = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(side); sl.setContentsMargins(0, 0, 6, 0); sl.setSpacing(12)
            scroll.setWidget(side); root.addWidget(scroll)
            self.sliders = {}

            def group(title, items):
                f, l = card(); l.addWidget(header(title))
                for (label, key, lo, hi, fmt, log, tip) in items:
                    w = ParamSlider(label, key, lo, hi, fmt, log, self.param_changed, tip)
                    w.set(self.sim.p[key]); self.sliders[key] = w; l.addWidget(w)
                sl.addWidget(f)

            group("Gene", [
                ("Cost of gene ON (c)", "c", 0.0, 0.4, "{:.3f}", False, "Extra death rate paid by predators whose gene is ON, whether or not prey are available"),
                ("OFF death multiplier", "off_mult", 0.0, 2.0, "{:.2f}", False, "Death rate of OFF predators as a multiple of the base rate m (below 1 = OFF is a refuge)")])
            group("Switching", [
                ("Rate ON → OFF", "s_off", 0.001, 3.0, "{:.3f}", True, "Rate per unit time at which a switcher's gene turns OFF"),
                ("Rate OFF → ON", "s_on", 0.001, 3.0, "{:.3f}", True, "Rate per unit time at which a switcher's gene turns ON")])
            group("Ecology", [
                ("Prey growth r", "r", 0.2, 2.0, "{:.2f}", False, ""), ("Carrying capacity K", "K", 500, 6000, "{:.0f}", False, ""),
                ("Attack rate a", "a", 0.0005, 0.006, "{:.4f}", False, ""), ("Handling time h", "h", 0.2, 3.0, "{:.2f}", False, "Longer handling saturates the functional response sooner"),
                ("Conversion e", "e", 0.1, 1.0, "{:.2f}", False, "Offspring per capture"), ("Predator death m", "m", 0.05, 0.6, "{:.3f}", False, "")])

            f, l = card(); l.addWidget(header("Add individuals"))
            row = QtWidgets.QHBoxLayout(); self.n_add = QtWidgets.QSpinBox(); self.n_add.setRange(1, 5000); self.n_add.setValue(100)
            row.addWidget(QtWidgets.QLabel("Number")); row.addStretch(1); row.addWidget(self.n_add); l.addLayout(row)
            b1 = QtWidgets.QPushButton("Add switchers"); b1.clicked.connect(lambda: self.sim.add(self.n_add.value(), 1))
            b2 = QtWidgets.QPushButton("Add residents"); b2.clicked.connect(lambda: self.sim.add(self.n_add.value(), 0))
            b3 = QtWidgets.QPushButton("Remove all switchers"); b3.clicked.connect(self.sim.clear_switchers)
            for b in (b1, b2, b3): l.addWidget(b)
            sl.addWidget(f)

            f, l = card(); l.addWidget(header("Display"))
            self.speed = ParamSlider("Speed (time per second)", "speed", 2, 120, "{:.0f}", True)
            self.speed.set(25); self.win = ParamSlider("Window (time units)", "win", 100, 1500, "{:.0f}", True, self.win_changed)
            self.win.set(300); l.addWidget(self.speed); l.addWidget(self.win)
            sl.addWidget(f); sl.addStretch(1)

            # ---- right: KPIs, plots, buttons ---------------------------------------------------
            right = QtWidgets.QVBoxLayout(); right.setSpacing(12); root.addLayout(right, 1)
            kp = QtWidgets.QHBoxLayout(); kp.setSpacing(10); self.kpi = {}
            for key, name in (("t", "Time"), ("prey", "Prey"), ("pred", "Predators"), ("share", "Switcher share"), ("onf", "Switchers with gene ON")):
                f = QtWidgets.QFrame(); f.setObjectName("card"); v = QtWidgets.QVBoxLayout(f); v.setContentsMargins(14, 8, 14, 8); v.setSpacing(0)
                a = QtWidgets.QLabel("–"); a.setObjectName("kpi"); b = QtWidgets.QLabel(name); b.setObjectName("kpil")
                v.addWidget(a); v.addWidget(b); kp.addWidget(f); self.kpi[key] = a
            right.addLayout(kp)

            self.fig = Figure(figsize=(9, 6)); self.canvas = FigureCanvasQTAgg(self.fig)
            self.plots = Plots(self.fig, self.theme)
            pf = QtWidgets.QFrame(); pf.setObjectName("card"); pl = QtWidgets.QVBoxLayout(pf); pl.setContentsMargins(6, 6, 6, 6); pl.addWidget(self.canvas)
            right.addWidget(pf, 1)

            bar = QtWidgets.QHBoxLayout(); bar.setSpacing(8)
            self.btn_play = QtWidgets.QPushButton("Pause"); self.btn_play.setObjectName("primary"); self.btn_play.clicked.connect(self.toggle)
            br = QtWidgets.QPushButton("Reset"); br.clicked.connect(self.reset)
            bp = QtWidgets.QPushButton("Defaults"); bp.clicked.connect(self.defaults)
            self.btn_theme = QtWidgets.QPushButton("Dark mode"); self.btn_theme.clicked.connect(self.toggle_theme)
            self.status = QtWidgets.QLabel(""); self.status.setObjectName("kpil")
            for b in (self.btn_play, br, bp, self.btn_theme): bar.addWidget(b)
            bar.addStretch(1); bar.addWidget(self.status); right.addLayout(bar)

            self.setStyleSheet(qss(THEMES[self.theme]))
            self.timer = QtCore.QTimer(self); self.timer.setInterval(40); self.timer.timeout.connect(self.tick); self.timer.start()

        # ---- slots ---------------------------------------------------------------------------
        def param_changed(self, key, v):
            self.sim.p[key] = v
        def win_changed(self, key, v):
            self.window_len = v
        def toggle(self):
            self.playing = not self.playing; self.btn_play.setText("Pause" if self.playing else "Play")
        def reset(self):
            self.sim.reset(); self.plots.update(self.sim, self.window_len); self.canvas.draw_idle()
        def defaults(self):
            for k, w in self.sliders.items(): self.sim.p[k] = DEFAULTS[k]; w.set(DEFAULTS[k])
            self.reset()
        def toggle_theme(self):
            self.theme = "dark" if self.theme == "light" else "light"
            self.btn_theme.setText("Light mode" if self.theme == "dark" else "Dark mode")
            self.setStyleSheet(qss(THEMES[self.theme])); self.plots.apply_theme(self.theme)
            self.plots.update(self.sim, self.window_len); self.canvas.draw_idle()

        def tick(self):
            if not self.playing: return
            sim = self.sim
            steps = max(1, int(round(self.speed.value() * 0.04 / sim.p["dt"])))
            for _ in range(steps):
                if sim.on.size == 0: break
                sim.step()
            self.frame += 1
            if self.frame % 3 == 0:
                self.plots.update(sim, self.window_len); self.canvas.draw_idle(); self.update_kpis()

        def update_kpis(self):
            s = self.sim; res = int((s.strain == 0).sum()); sw = int((s.strain == 1).sum())
            son = int(((s.strain == 1) & s.on).sum()); tot = res + sw
            self.kpi["t"].setText(f"{s.t:.0f}"); self.kpi["prey"].setText(f"{s.N}"); self.kpi["pred"].setText(f"{tot}")
            self.kpi["share"].setText(f"{100 * sw / tot:.0f} %" if tot else "–")
            self.kpi["onf"].setText(f"{100 * son / sw:.0f} %" if sw else "–")
            self.status.setText("Predators extinct: press Reset" if tot == 0 else ("Prey extinct" if s.N == 0 else ""))

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = Window(); w.show()
    sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())


if __name__ == "__main__":
    main()