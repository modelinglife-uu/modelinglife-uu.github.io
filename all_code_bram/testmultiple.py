"""
Several fungi growing in a piece of sandy soil (30 x 30 cm, hexagon grid).

Soil:
  - every hexagon has 12 units, each unit is pore with prob. POROSITY  -> room ~ Bin(12, POROSITY)
  - glucose from one Perlin noise layer, log-normal so mean and CV are fixed
  - concentration = glucose / pore water,  pore water = room/12 * 0.042 mL

Fungus (same rules as model2.py), per hour:
  - growth once: tips with status >= 2*cost make 2 cells (branch + extension or 2 extensions)
    branching prob = max(0.1, C^10 / (15.19^10 + C^10))
  - then every minute: uptake, glucose diffusion, translocation, parent-child diffusion, maintenance
  - death: cells < 5 ug (older than 6 h) die with prob 1 - exp(-0.001)

Run:   python testingmultiple.py            -> window
       python testingmultiple.py --days 3   -> no window, prints 3 days
Keys:  space run/pause, S +1 day, H +1 hour, R new start positions, +/- speed
       1 fungi (on concentration)   2 room   3 glucose (mg)   4 concentration (g/L)
"""
import colorsys
import math
import random
import sys

import numpy as np

# soil
FIELD_CM = 90.0
SIDE_CM = 2               # hexagon side, 2 mm
UNITS = 12                  # units per hexagon
POROSITY = 0.43             # B1 sand
GLUCOSE_MEAN_MG = 0.57      # mean glucose per hexagon (0.84 = 20 g/L in a full hexagon)
PATCH_CM = 3.0              # patch size of the noise
PATCH_CV = 1.0              # patchiness
MAP_SEED = 1
VHEX_ML = 25 / 595          # water in a full hexagon (25 mL plate / 595 hexagons)

# fungi
N_FUNGI = 100
FUNGUS_SEED = 1
EDGE_CM = 1.0               # min. distance of a start point to the border

RULES = {
    "vmax_mg_h": 0.004,     # max uptake per cell per hour
    "Km": 0.005,            # g/L
    "D_glc": 0,             # glucose diffusion cm2/day (0 = off)
    "transloc": 80,         # per day
    "D_link": 0.2,          # parent-child diffusion, per day
    "extra_cost": 0.0005,   # mg lost per new cell
    "newstatus": 0.0,
    "maint_mg_h": 0.001,    # maintenance per cell per hour
    "startstatus": 0.15,    # mg per start cell
    "Kb": 15.19,            # branching
    "hb": 10,
    "p_floor": 0.1,         # min. branching prob
    "turn": 0.05,
    "straight": 0.33,       # share of branches going "straight"
    "ntips": 6,             # 1 + 6 + 6 = 13 start cells
    "death_status": 0.005,  # mg
    "death_rate": 0.001,    # per hour
    "death_grace": 6,       # hours
    "substeps": 60,         # steps per hour
}
CELL_GLUCOSE_MG = 0.225 / 3.333     # glucose to build one unit (0.225 mg FW, yield 3.333)

# window
WINDOW = 900
TICKS_PER_FRAME = 4         # hours per frame, change with +/-
PRINT_EVERY_DAY = True

HEX_CIRCLE = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]   # axial directions


# ---------------- maps ----------------
M32 = 0xFFFFFFFF


def mulberry32(a):
    # small seeded random generator (same one as in the js pages, so the maps match)
    a &= M32

    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return rnd


def make_perlin(seed):
    # 2d perlin noise
    rnd = mulberry32(seed * 7919 + 13)
    p = list(range(256))
    for i in range(255, 0, -1):
        j = math.floor(rnd() * (i + 1))
        p[i], p[j] = p[j], p[i]
    P = [p[i & 255] for i in range(512)]
    G = [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def grad(h, x, y):
        g = G[h & 7]
        return g[0] * x + g[1] * y

    def noise(x, y):
        X, Y = math.floor(x), math.floor(y)
        xf, yf, xi, yi = x - X, y - Y, X & 255, Y & 255
        u, v = fade(xf), fade(yf)
        aa, ab = P[P[xi] + yi], P[P[xi] + yi + 1]
        ba, bb = P[P[xi + 1] + yi], P[P[xi + 1] + yi + 1]
        x1 = grad(aa, xf, yf) + u * (grad(ba, xf - 1, yf) - grad(aa, xf, yf))
        x2 = grad(ab, xf, yf - 1) + u * (grad(bb, xf - 1, yf - 1) - grad(ab, xf, yf - 1))
        return x1 + v * (x2 - x1)
    return noise


class Space:
    # hexagon grid + room + glucose

    def __init__(self, seed=MAP_SEED):
        s = SIDE_CM
        self.w = s * math.sqrt(3)
        self.h = 1.5 * s
        self.cols = math.floor((FIELD_CM - self.w / 2) / self.w)
        self.rows = math.floor((FIELD_CM - s / 2) / self.h)
        self.n = self.cols * self.rows
        self.cx = np.array([self.w / 2 + self.w * (c + 0.5 * (r & 1)) for r in range(self.rows) for c in range(self.cols)])
        self.cy = np.array([s + self.h * r for r in range(self.rows) for c in range(self.cols)])
        # neighbours, -1 = outside
        self.nbr = []
        for r in range(self.rows):
            for c in range(self.cols):
                q = c - (r - (r & 1)) // 2
                row = []
                for dq, dr in HEX_CIRCLE:
                    q2, r2 = q + dq, r + dr
                    c2 = q2 + (r2 - (r2 & 1)) // 2
                    row.append(r2 * self.cols + c2 if 0 <= r2 < self.rows and 0 <= c2 < self.cols else -1)
                self.nbr.append(row)
        self.wall = [min(row) < 0 for row in self.nbr]

        # room
        rb = mulberry32(seed * 31 + 555)
        self.room = np.array([sum(1 for _ in range(UNITS) if rb() < POROSITY) for _ in range(self.n)], dtype=int)

        # glucose (sigma found by bisection to get the CV)
        noise, ro = make_perlin(seed), mulberry32(seed + 7)
        ox, oy = ro() * 200, ro() * 200
        z = np.array([noise(x / PATCH_CM + ox, y / PATCH_CM + oy) for x, y in zip(self.cx, self.cy)])
        z = (z - z.mean()) / (z.std() or 1)

        def cv_of(sg):
            e = np.exp(sg * z)
            return e.std() / e.mean()
        lo, hi = 0.0, 1.0
        while cv_of(hi) < PATCH_CV and hi < 50:
            hi *= 2
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cv_of(mid) < PATCH_CV else (lo, mid)
        sigma = (lo + hi) / 2 if PATCH_CV > 0 else 0.0
        e = np.exp(sigma * z)
        self.glucose_mg = e * GLUCOSE_MEAN_MG * self.n / e.sum()

    def concentration_gL(self, glucose_mg):
        # g/L in the pore water, nan if no pores
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.room > 0, glucose_mg / (self.room / UNITS * VHEX_ML), np.nan)

    def vec(self, d):
        dq, dr = HEX_CIRCLE[d]
        return self.w * (dq + dr / 2), self.w * dr * math.sqrt(3) / 2


# ---------------- fungi ----------------
class Cell:
    __slots__ = ("hex", "status", "branch", "parent", "active", "direction", "progress", "fungus", "idx", "born")

    def __init__(self, hex_, status, branch, parent, active, direction, fungus, idx, born=0):
        self.hex = hex_
        self.status = status          # mg glucose
        self.branch = branch
        self.parent = parent          # None for the start cell or if the parent died
        self.active = active          # tip?
        self.direction = direction    # 0..5
        self.progress = 1             # for the stay rule
        self.fungus = fungus
        self.idx = idx
        self.born = born              # hour


class Fungus:
    def __init__(self, id_, start):
        self.id = id_
        self.start = start            # start hexagon, tips don't grow back towards it
        self.next_branch = 0
        r, g, b = colorsys.hsv_to_rgb((id_ * 0.618034) % 1, 0.75, 0.95)
        self.colour = (int(r * 255), int(g * 255), int(b * 255))


class Field:
    def __init__(self, space, fungus_seed=FUNGUS_SEED, rules=RULES):
        self.space, self.rng, R = space, random.Random(fungus_seed), rules
        self.V = R["vmax_mg_h"] * 24          # per day
        self.K = R["Km"]
        self.K_b, self.h_b, self.p_floor = R["Kb"], R["hb"], R["p_floor"]
        self.turn_prob, self.straight = R["turn"], R["straight"]
        self.transloc, self.D_link, self.D_glc = R["transloc"], R["D_link"], R["D_glc"]
        self.cost = CELL_GLUCOSE_MG + R["extra_cost"]
        self.new_status, self.maint = R["newstatus"], R["maint_mg_h"]
        self.death_status, self.death_rate, self.death_grace = R["death_status"], R["death_rate"], R["death_grace"]
        self.substeps = max(1, round(R["substeps"]))
        self.grow_threshold = max(2 * self.cost + 3 * self.new_status, self.cost + self.new_status)
        self.n_split = math.isqrt(UNITS)      # stay rule: 3 cells per hexagon
        self.food = space.glucose_mg.copy()
        self.food_start = self.food.sum()
        self.water = space.room.copy()
        self.vol = self.water / UNITS * VHEX_ML
        self.room = space.room.copy()
        self.count = np.zeros(space.n, dtype=int)   # cells per hexagon
        self.cells, self.fungi, self.branch_count = [], [], {}
        self.hour, self.taken_up, self.maint_paid = 0, 0.0, 0.0
        self.died, self.died_status = 0, 0.0
        # neighbour pairs with water on both sides (for diffusion)
        a, b = [], []
        for i in range(space.n):
            for d in range(3):
                j = space.nbr[i][d]
                if j >= 0 and self.vol[i] > 0 and self.vol[j] > 0:
                    a.append(i)
                    b.append(j)
        self.pair_a, self.pair_b = np.array(a, dtype=int), np.array(b, dtype=int)

        # random start points (with pores, not too close to the border)
        ok = [i for i in range(space.n) if self.water[i] > 0 and min(space.cx[i], space.cy[i], FIELD_CM - space.cx[i],
                                                                       FIELD_CM - space.cy[i]) >= EDGE_CM]
        n_start = R["ntips"]
        for k, h in enumerate(self.rng.sample(ok, N_FUNGI)):
            f = Fungus(k, h)
            self.fungi.append(f)
            # start: 1 cell -> 6 cells -> 6 tips, each in the first hexagon along its direction with room
            used = {h: 1}

            def place(d, k0, h=h, used=used):
                x, steps = h, 0
                while steps < k0:
                    x = space.nbr[x][d]
                    steps += 1
                    if x < 0:
                        return h, 0
                while x >= 0:
                    if used.get(x, 0) < self.room[x]:
                        used[x] = used.get(x, 0) + 1
                        return x, steps
                    x = space.nbr[x][d]
                    steps += 1
                return h, 0

            first = self.rng.randrange(6)
            spore = self.new_cell(h, R["startstatus"], 0, None, False, first, f)
            dirs = [(first + round(i * 6 / n_start)) % 6 for i in range(n_start)]
            middles = []
            for i, d in enumerate(dirs):
                x, st = place(d, 0)
                middles.append((self.new_cell(x, R["startstatus"], i + 1, spore, False, d, f), st))
            for i, d in enumerate(dirs):
                kid, st = middles[i]
                x, _ = place(d, st)
                self.new_cell(x, R["startstatus"], i + 1, kid, True, d, f)
            f.next_branch = n_start + 1

    def new_cell(self, h, status, branch, parent, active, direction, fungus):
        c = Cell(h, status, branch, parent, active, direction, fungus, len(self.cells), self.hour)
        self.cells.append(c)
        self.count[h] += 1
        key = (fungus.id, h, branch)
        self.branch_count[key] = self.branch_count.get(key, 0) + 1
        return c

    def conc(self, h):
        return self.food[h] / self.vol[h] if self.vol[h] > 0 else 0.0

    def has_room(self, h):
        return h >= 0 and self.count[h] < self.room[h]

    def free_places(self, h):
        return max(0, self.room[h] - self.count[h]) if h >= 0 else 0

    def outward(self, h, d, fungus):
        # false if d points back to the start
        s = self.space
        vx, vy = s.vec(d)
        return (s.cx[h] - s.cx[fungus.start]) * vx + (s.cy[h] - s.cy[fungus.start]) * vy >= -1e-9

    def not_inward(self, cell, directions):
        return [d for d in directions if self.outward(cell.hex, d, cell.fungus)]

    def branch_directions(self, d):
        lr = [1, -1]
        self.rng.shuffle(lr)
        return [(d + lr[0]) % 6, (d + lr[1]) % 6]

    def turned(self, h, d, fungus):
        if self.rng.random() < self.turn_prob:
            t = (d + self.rng.choice([1, -1])) % 6
            if self.outward(h, t, fungus):
                return t
        return d

    def count_branches(self):
        self.branch_count = {}
        for c in self.cells:
            key = (c.fungus.id, c.hex, c.branch)
            self.branch_count[key] = self.branch_count.get(key, 0) + c.progress

    def pick(self, options, weights):
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(options)
        r, i = self.rng.random() * total, 0
        while r >= weights[i]:
            r -= weights[i]
            i += 1
        return options[i]

    # growth
    def extend(self, cell):
        # stay in the hexagon until 3 cells, then straight or +-60, never back
        if self.space.wall[cell.hex]:
            cell.active = False
            return False
        n = self.branch_count.get((cell.fungus.id, cell.hex, cell.branch), 0)
        if n < self.n_split and self.has_room(cell.hex):
            cell.active = False
            self.new_cell(cell.hex, self.new_status, cell.branch, cell, True, self.turned(cell.hex, cell.direction, cell.fungus), cell.fungus)
            return True
        for d in [cell.direction] + self.not_inward(cell, self.branch_directions(cell.direction)):
            h = self.space.nbr[cell.hex][d]
            if self.has_room(h):
                cell.active = False
                self.new_cell(h, self.new_status, cell.branch, cell, True, self.turned(h, d, cell.fungus), cell.fungus)
                return True
        cell.active = False
        return False

    def place_branch(self, cell):
        # first where (own hexagon, else a free neighbour), then which direction (angle share * free places)
        nb, f, h = self.space.nbr, cell.fungus, cell.hex
        if not self.has_room(h):
            forward = self.not_inward(cell, self.branch_directions(cell.direction)) + [cell.direction]
            free = [self.free_places(nb[h][d]) for d in forward]
            if sum(free) == 0:
                return False
            h = nb[h][self.pick(forward, free)]
        sides = [d for d in self.branch_directions(cell.direction) if self.outward(h, d, f)]
        options = sides + [cell.direction]
        angle = [(1 - self.straight) / 2] * len(sides) + [self.straight]
        weights = [a * self.free_places(nb[h][d]) for a, d in zip(angle, options)]
        d = self.pick(options, weights if sum(weights) > 0 else angle)
        b = f.next_branch
        f.next_branch += 1
        new = self.new_cell(h, self.new_status, b, cell, True, d, f)
        if h == cell.hex:       # same hexagon: takes over the parent's count
            pr = max(1, self.branch_count.get((f.id, h, cell.branch), 0))
            new.progress = pr
            self.branch_count[(f.id, h, b)] = pr
        return True

    def grow(self):
        tips = [c for c in self.cells if c.active]
        self.rng.shuffle(tips)
        self.count_branches()
        price = self.cost + self.new_status
        for cell in tips:
            if self.space.wall[cell.hex] or cell.status < self.grow_threshold:
                continue
            c = self.conc(cell.hex)
            p = c ** self.h_b / (self.K_b ** self.h_b + c ** self.h_b) if self.K_b > 0 and c > 0 else 0.0
            if self.rng.random() < max(self.p_floor, p) and self.place_branch(cell):
                cell.status -= price
                if self.extend(cell):
                    cell.status -= price
            elif self.extend(cell):
                cell.status -= price
                t1 = self.cells[-1]
                if not self.space.wall[t1.hex] and self.extend(t1):
                    cell.status -= price

    # every minute (with numpy)
    def feed(self):
        nc = len(self.cells)
        hexes = np.fromiter((c.hex for c in self.cells), dtype=int, count=nc)
        status = np.fromiter((c.status for c in self.cells), dtype=float, count=nc)
        parent = np.fromiter((c.parent.idx if c.parent is not None else -1 for c in self.cells), dtype=int, count=nc)
        kids = np.nonzero(parent >= 0)[0]
        pk = parent[kids]
        n_kids = np.bincount(pk, minlength=nc)
        f = 1 / 24 / self.substeps
        rate = min(1.0, self.transloc * f)
        has_water = self.vol > 0
        vol = np.where(has_water, self.vol, 1.0)
        c_diff = min(1.0, 4 * self.D_glc * f / self.space.w ** 2) / 6
        va, vb = self.vol[self.pair_a], self.vol[self.pair_b]
        vmin = np.minimum(va, vb)
        pay = self.maint * 24 * f
        for _ in range(self.substeps):
            # uptake
            C = np.where(has_water, self.food / vol, 0.0)
            ch = C[hexes]
            a = np.where(ch > 0, self.V * ch / (self.K + ch) * f, 0.0)
            demand = np.bincount(hexes, a, minlength=self.space.n)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(demand > self.food, np.maximum(self.food, 0.0) / demand, 1.0)
            take = a * scale[hexes]
            status += take
            got = np.bincount(hexes, take, minlength=self.space.n)
            self.food = np.maximum(0.0, self.food - got)
            self.taken_up += got.sum()
            # glucose diffusion
            if self.D_glc > 0 and len(self.pair_a):
                flow = c_diff * (self.food[self.pair_a] / va - self.food[self.pair_b] / vb) * vmin
                change = np.bincount(self.pair_b, flow, minlength=self.space.n) - np.bincount(self.pair_a, flow, minlength=self.space.n)
                self.food = np.maximum(0.0, self.food + change)
            # translocation parent -> children
            if rate > 0 and len(kids):
                offer = np.where(n_kids > 0, rate * np.maximum(0.0, status) / np.maximum(n_kids, 1), 0.0)
                status -= offer * n_kids
                status[kids] += offer[pk]
            # parent <-> child diffusion
            if self.D_link > 0 and len(kids):
                flow = self.D_link * f * (status[kids] - status[pk])
                status[kids] -= flow
                status += np.bincount(pk, flow, minlength=nc)
            # maintenance
            if pay > 0:
                p = np.minimum(pay, np.maximum(0.0, status))
                status -= p
                self.maint_paid += p.sum()
        for c, s in zip(self.cells, status):
            c.status = s

    def die(self):
        if self.death_rate <= 0:
            return
        p = 1 - math.exp(-self.death_rate)
        dead = [c for c in self.cells
                if self.hour - c.born >= self.death_grace and c.status < self.death_status and self.rng.random() < p]
        if not dead:
            return
        dead_set = set(dead)
        for c in self.cells:
            if c.parent in dead_set:
                c.parent = None
        for c in dead:
            self.count[c.hex] -= 1
            self.died_status += max(0.0, c.status)
        self.cells = [c for c in self.cells if c not in dead_set]
        for i, c in enumerate(self.cells):
            c.idx = i
        self.died += len(dead)

    def tick(self):
        # one hour
        self.grow()
        self.feed()
        self.hour += 1
        self.die()

    def summary(self):
        per = [0] * len(self.fungi)
        tips = 0
        for c in self.cells:
            per[c.fungus.id] += 1
            tips += c.active
        return {"day": self.hour / 24, "cells": len(self.cells), "tips": tips, "per_fungus": per,
                "hexagons": int((self.count > 0).sum()), "food_left": self.food.sum() / self.food_start,
                "taken_mg": self.taken_up, "maint_mg": self.maint_paid, "died": self.died}


# ---------------- window ----------------
VIEWS = {"1": "fungi", "2": "room", "3": "glucose", "4": "concentration"}


def hex_polygons(space):
    k = WINDOW / FIELD_CM
    s = SIDE_CM * k * 1.03
    corners = [(math.cos(math.radians(60 * j - 30)), math.sin(math.radians(60 * j - 30))) for j in range(6)]
    return [[(x * k + s * cx, y * k + s * cy) for cx, cy in corners] for x, y in zip(space.cx, space.cy)]


def pixel_index(space, pygame):
    # hexagon number for every pixel (-1 = none); draw each hexagon once with its number as colour
    surf = pygame.Surface((WINDOW, WINDOW))
    surf.fill((0, 0, 0))
    for i, poly in enumerate(hex_polygons(space)):
        v = i + 1
        pygame.draw.polygon(surf, ((v >> 16) & 255, (v >> 8) & 255, v & 255), poly)
    rgb = pygame.surfarray.array3d(surf).astype(np.int64)
    return (rgb[..., 0] << 16) + (rgb[..., 1] << 8) + rgb[..., 2] - 1


def ramp(lo, hi, t):
    t = np.clip(np.nan_to_num(t), 0, 1)[:, None]
    return np.array(lo, float) + (np.array(hi, float) - np.array(lo, float)) * t


class Painter:
    def __init__(self, space, pygame):
        self.space, self.pygame = space, pygame
        self.index = pixel_index(space, pygame)
        self.top_glc = np.quantile(space.glucose_mg, 0.99)
        self.top_conc = np.nanquantile(space.concentration_gL(space.glucose_mg), 0.99)
        self.grey, self.dark = np.array((158, 151, 140), float), np.array((40, 38, 34), float)

    def draw(self, screen, field, view):
        sp, n = self.space, self.space.n
        glc = field.food
        if view == "glucose":
            col = ramp((241, 246, 238), (22, 80, 58), glc / self.top_glc)
        elif view == "concentration":
            conc = sp.concentration_gL(glc)
            col = ramp((251, 241, 230), (154, 59, 12), conc / self.top_conc)
            col[np.isnan(conc)] = self.grey
        elif view == "fungi":
            conc = sp.concentration_gL(glc)
            col = ramp((241, 246, 238), (22, 80, 58), conc / self.top_conc)
            col[np.isnan(conc)] = self.grey
        else:
            col = ramp((241, 235, 224), (31, 95, 139), sp.room / UNITS)
        if view == "fungi" and field.cells:
            # colour of the fungus with most cells in the hexagon, lighter where tips are
            nf = len(field.fungi)
            hexes = np.fromiter((c.hex for c in field.cells), int, len(field.cells))
            fid = np.fromiter((c.fungus.id for c in field.cells), int, len(field.cells))
            tip = np.fromiter((c.active for c in field.cells), bool, len(field.cells))
            per = np.bincount(hexes * nf + fid, minlength=n * nf).reshape(n, nf)
            has = field.count > 0
            owner = per.argmax(axis=1)
            fcol = np.array([f.colour for f in field.fungi], float)
            shade = 0.45 + 0.55 * np.minimum(1, field.count / UNITS)
            base = fcol[owner] * shade[:, None]
            tips = np.zeros(n, bool)
            tips[hexes[tip]] = True
            base[tips] = base[tips] + (255 - base[tips]) * 0.5
            col[has] = base[has]
        col = np.vstack([col, self.dark]).astype(np.uint8)
        self.pygame.surfarray.blit_array(screen, col[self.index])
        self.pygame.display.flip()
        s = field.summary()
        self.pygame.display.set_caption(f"day {s['day']:.2f} | {view} | cells {s['cells']} | tips {s['tips']} | "
                                        f"{N_FUNGI} fungi | glucose left {100 * s['food_left']:.1f} % | "
                                        f"{field.speed} h per frame (+/-)")


def print_day(field):
    s = field.summary()
    print(f"day {s['day']:.0f}: {s['cells']} cells, {s['tips']} tips, {s['hexagons']} hexagons, glucose taken up "
          f"{s['taken_mg']:.1f} mg ({100 * (1 - s['food_left']):.2f} %), maintenance {s['maint_mg']:.1f} mg, "
          f"{s['died']} died, cells per fungus {s['per_fungus']}")


def new_field(space, seed):
    field = Field(space, fungus_seed=seed)
    for f in field.fungi:
        x, y = space.cx[f.start], space.cy[f.start]
        print(f"fungus {f.id}: starts at ({x:.1f}, {y:.1f}) cm, room {space.room[f.start]}, "
              f"glucose {space.glucose_mg[f.start]:.3f} mg, C = {field.conc(f.start):.1f} g/L")
    return field


def run_ticks(field, n):
    for _ in range(n):
        field.tick()
        if PRINT_EVERY_DAY and field.hour % 24 == 0:
            print_day(field)


def describe(space):
    conc = space.concentration_gL(space.glucose_mg)
    print(f"space: {space.cols} x {space.rows} = {space.n} hexagons | porosity {space.room.mean() / UNITS:.3f} | "
          f"glucose mean {space.glucose_mg.mean():.2f} mg, CV {space.glucose_mg.std() / space.glucose_mg.mean():.2f} | "
          f"concentration mean {np.nanmean(conc):.1f} g/L, median {np.nanmedian(conc):.1f} g/L")


def main():
    space = Space()
    describe(space)
    if "--days" in sys.argv:
        days = float(sys.argv[sys.argv.index("--days") + 1])
        field = new_field(space, FUNGUS_SEED)
        run_ticks(field, round(24 * days))
        return
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW, WINDOW))
    painter = Painter(space, pygame)
    seed = FUNGUS_SEED
    field, view = new_field(space, seed), "fungi"
    speed = TICKS_PER_FRAME
    clock = pygame.time.Clock()
    running, paused = True, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_s:
                    run_ticks(field, 24)
                elif event.key == pygame.K_h:
                    run_ticks(field, 1)
                elif event.key == pygame.K_r:
                    seed = random.randrange(10 ** 6)
                    field = new_field(space, seed)
                elif event.unicode in ("+", "="):
                    speed = min(48, speed * 2)
                elif event.unicode in ("-", "_"):
                    speed = max(1, speed // 2)
                elif event.unicode in VIEWS:
                    view = VIEWS[event.unicode]
        if not paused:
            run_ticks(field, speed)
        field.speed = speed
        painter.draw(screen, field, view)
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
