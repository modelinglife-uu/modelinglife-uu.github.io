"""Spatial model of fungi with a core and an (optional) accessory chromosome.

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

"""
import random
import numpy as np

# ======= SETTINGS =======
L = 128                # grid is L x L
T = 50000              # generations
SEED = 2

b = 0.8                # colonisation rate of empty sites
d = 0.1               # death rate

beta = 0.035            # transfer rate of the accessory chromosome
c = 0.0                # growth cost of carrying an accessory chromosome
lam = 0.005            # loss of the accessory chromosome
mu = 4e-5              # loss of a gene copy (from either chromosome)
tau1 = 1e-4            # translocation core -> accessory
tau2 = 1e-4            # translocation accessory -> core

v = 1.0                # effector gene: gain on a healthy plant
gc = 0.2               # effector gene: fixed cost

rho = 0.005            # plant regrowth
kappa = 0.05            # plant damage when the fungus on the same site carries the gene

# start: initial frequencies of each kind of fungus (share of all sites, placed at random).
START = {"N0": 0.1, "NC": 0.1, "NE": 0.1, "NA": 0.1, "NCE": 0.1, "NCA": 0.1}
P0 = 1.0               # start health of the plants

PRINT_EVERY = 100      # print a line (and update the plot) every this many generations
MIX = False            # True: perfect mixing (every neighbour is a random site of the grid)
PLOT = True            # live plot (needs matplotlib); False = only print
# =========================================================================================================================

if PLOT:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

random.seed(SEED)


def random_neighbour(i, j):
    """Row and column of one random neighbour of site (i, j): one of the 8 sites around it (the grid wraps around
    at the edges). With MIX = True the 'neighbour' is a random site anywhere on the grid (perfect mixing)."""
    if MIX:
        return random.randrange(L), random.randrange(L)
    else:
        nbs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        dy, dx = random.choice(nbs)
    return (i + dy) % L, (j + dx) % L

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
    carriers = (occ & (core | acc)).sum() # of all occupied sites, how many have the gene?
    return 100 * (occ & acc).sum() / carriers, 100 * (occ & core).sum() / carriers


def step(occ, core, acc, chrom, P):
    """Update the whole grid once. Every site looks at the old state; the result goes in the new_ arrays."""
    # First make a copy of the current state to update into. If we don't do that, individuals will not be updated 'at the same time',
    # but from the top left to the bottom right, which would introduce a weird bias.
    new_occ, new_core, new_acc, new_chrom = occ.copy(), core.copy(), acc.copy(), chrom.copy()

    for i in range(L):
        for j in range(L):

            # ---- 1. plant at this site
            carrier = occ[i, j] and (core[i, j] or acc[i, j])
            P[i, j] = P[i, j] + rho * (1 - P[i, j]) # Logistic growth
            if carrier:
                P[i, j] = P[i, j] - kappa * P[i, j] # Killing by fungus on this spot
            P[i, j] = min(max(P[i, j], 0), 1)

            if not occ[i, j]:
                # ---- 2. empty site: maybe colonised by a random neighbour (the "parent")
                ni, nj = random_neighbour(i, j)
                if occ[ni, nj]:
                    W = 1.0
                    if chrom[ni, nj]:
                        W = W * (1 - c)
                    if core[ni, nj] or acc[ni, nj]:
                        W = W * (1 + v * P[i, j]) * (1 - gc)
                    if random.random() < b * W:
                        new_occ[i, j] = True                      # the newborn copies its parent
                        new_core[i, j] = core[ni, nj]
                        new_acc[i, j] = acc[ni, nj]
                        new_chrom[i, j] = chrom[ni, nj]
            else:
                # ---- 3. fungus without the accessory chromosome: maybe gets one from a random neighbour (the "donor")
                if not chrom[i, j]:
                    ni, nj = random_neighbour(i, j)
                    if chrom[ni, nj] and random.random() < beta:
                        new_chrom[i, j] = True
                        if acc[ni, nj]:
                            new_acc[i, j] = True                  # the donor's chromosome carries the gene

                # ---- 4. mutational events, note that these happen even during the lifetime of an indvidual (not just at birth)
                if core[i, j] and random.random() < mu:           # gene lost from the core
                    new_core[i, j] = False
                if acc[i, j] and random.random() < mu:            # gene lost from the accessory chromosome
                    new_acc[i, j] = False
                if core[i, j] and chrom[i, j] and not acc[i, j] and random.random() < tau1:
                    new_core[i, j] = False                        # translocation core -> accessory (cut and paste)
                    new_acc[i, j] = True
                if acc[i, j] and not core[i, j] and random.random() < tau2:
                    new_acc[i, j] = False                         # translocation accessory -> core
                    new_core[i, j] = True
                if chrom[i, j] and random.random() < lam:         # chromosome lost (the gene on it goes too)
                    new_chrom[i, j] = False
                    new_acc[i, j] = False
                if random.random() < d:                          # death
                    new_occ[i, j] = False
                    new_core[i, j] = False
                    new_acc[i, j] = False
                    new_chrom[i, j] = False

    return new_occ, new_core, new_acc, new_chrom, P


def run():
    occ, core, acc, chrom = start_state()
    P = np.full((L, L), P0)
    history = []

    if PLOT:
        plt.ion()                                                 # interactive mode: the plot updates while the model runs
        names = ["N0", "NC", "NE", "NA", "NCE", "NCA", "plants (mean health)"]
        colors = ["#8C8C8C", "#1E9BE8", "#F59BBB", "#E03131", "#C2A8EE", "#7A1590", "#2E8B57"]
        fig = plt.figure(figsize=(13, 7))
        ax_grid = fig.add_subplot(2, 3, 1)                        # the grid, coloured by kind
        picture = ax_grid.imshow(kind_map(occ, core, acc, chrom), vmin=0, vmax=6, interpolation="nearest",
                                 cmap=ListedColormap(["white"] + colors[:6]))
        ax_grid.axis("off")
        ax_plants = fig.add_subplot(2, 3, 2)                      # the plants: dark green = healthy, white = dead
        plant_picture = ax_plants.imshow(P, vmin=0, vmax=1, cmap="Greens", interpolation="nearest")
        ax_plants.axis("off")
        ax_plants.set_title("plant health")
        ax_bar = fig.add_subplot(2, 3, 3)                         # where is the gene?
        bars = ax_bar.bar(["on accessory", "on core"], [0, 0], color=["#E03131", "#1E9BE8"])
        ax_bar.set_ylim(0, 100)
        ax_bar.set_ylabel("% of gene carriers")
        ax = fig.add_subplot(2, 1, 2)                             # shares of the kinds over time
        lines = [ax.plot([], [], color=colors[i], lw=2, label=names[i])[0] for i in range(7)]
        ax.set_ylim(0, 1)                                         # the x-axis grows with the run (no fixed limit)
        ax.set_xlabel("generation")
        ax.set_ylabel("share")
        ax.legend(loc="upper right")

    for t in range(T + 1):
        # ---- record and print
        history.append([t] + kind_shares(occ, core, acc, chrom) + [P.mean()])
        if t % PRINT_EVERY == 0:
            acc_pct, core_pct = where_is_the_gene(occ, core, acc)
            print(f"t={t:6d}  occupied {occ.mean():.2f}  plants {P.mean():.2f}  "
                  f"gene on accessory {acc_pct:5.1f}%  on core {core_pct:5.1f}%")
            if PLOT:
                picture.set_data(kind_map(occ, core, acc, chrom))
                plant_picture.set_data(P)
                bars[0].set_height(acc_pct)
                bars[1].set_height(core_pct)
                ax_grid.set_title(f"generation {t}")
                h = np.array(history)
                for i in range(7):
                    lines[i].set_data(h[:, 0], h[:, 1 + i])
                ax.relim()                                        # the x-axis follows the data so far
                ax.autoscale_view(scalex=True, scaley=False)
                plt.pause(0.01)
        if t == T:
            break

        occ, core, acc, chrom, P = step(occ, core, acc, chrom, P)

    return np.array(history), (occ, core, acc, chrom, P)


if __name__ == "__main__":
    history, (occ, core, acc, chrom, P) = run()

    # final numbers: average over the last 20% of the run
    last = history[int(len(history) * 0.8):]
    print("\nshares of the six kinds (last 20% of the run):",
          dict(zip(["N0", "NC", "NE", "NA", "NCE", "NCA"], last[:, 1:7].mean(0).round(3))))
    print("plant health:", last[:, 7].mean().round(3))
    print("at the end: % of gene carriers with the gene on the accessory / on the core:",
          [round(x, 1) for x in where_is_the_gene(occ, core, acc)])

    if PLOT:
        plt.ioff()
        plt.show()             # keep the window open at the end