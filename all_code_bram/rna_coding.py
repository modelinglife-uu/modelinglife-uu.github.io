import random
import math
from collections import Counter
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from RNA import fold  # pip install ViennaRNA

random.seed(0)
plt.ion()

# --- Parameters ---
alphabet = "ACGU"
rna_length = 32
n_targets = 2             # 1 = fixed target, 2 = alternate between two targets
switch_every = 1000       # season length: generations between target switches

population_size = 1024
generations = 10000000
mutation_rate = 0.001     # per base (higher than DNA model: sequences are longer, space is huge)
min_target_difference = 0.5   # targets must differ in at least this fraction of positions
sample_interval = 10
num_competitors = 10
stable_for = 20           # target counts as 'found' only if best fitness = 1 for this many consecutive generations


# --- Core functions ---
def random_rna():
    return ''.join(random.choice(alphabet) for _ in range(rna_length))


@lru_cache(maxsize=None)
def get_structure(rna):
    """Fold RNA (cached: the same sequence is folded many times)."""
    return fold(rna)[0]


def fitness(rna, target):
    """Fitness = fraction of positions where structure matches target."""
    structure = get_structure(rna)
    matches = sum(1 for a, b in zip(structure, target) if a == b)
    return matches / len(target)


def mutate(rna, rate=mutation_rate):
    return ''.join(
        random.choice([b for b in alphabet if b != base]) if random.random() < rate else base
        for base in rna
    )


def structure_difference(a, b):
    """Fraction of positions where two structures differ."""
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)


def diversity(items):
    """Shannon diversity of any list of hashable items."""
    counts = Counter(items)
    total = len(items)
    return -sum((c / total) * math.log(c / total) for c in counts.values())


# --- Targets: fold random sequences; keep ones with base pairs that differ clearly from earlier targets ---
targets = []
while len(targets) < n_targets:
    candidate = get_structure(random_rna())
    if '(' in candidate and all(structure_difference(candidate, t) >= min_target_difference for t in targets):
        targets.append(candidate)
print("Targets:", *targets, sep="\n  ")

# --- Initialize ---
initial_sequence = random_rna()
population = [initial_sequence] * population_size

avg_fitness, max_fitness = [], []
seq_diversity, struct_diversity = [], []

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(13, 8))
fig.subplots_adjust(right=0.82)
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color='tab:blue')
ax1.set_ylim(0, 1)
line0, = ax1.plot([], [], color='black', linewidth=1, label='Max fitness')
line1, = ax1.plot([], [], color='tab:blue', linewidth=2, label='Avg fitness')
ax2 = ax1.twinx()
ax2.set_ylabel("Diversity", color='tab:purple')
line2, = ax2.plot([], [], color='tab:green', linestyle=':', linewidth=2, label='Sequence diversity')
line3, = ax2.plot([], [], color='tab:orange', linestyle=':', linewidth=2, label='Structure diversity')
ax3 = ax1.twinx()                      # third axis: time to find the target per season
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel("Generations to find target (within season)", color='tab:red')
ax3.set_ylim(0, switch_every)
bar_width = switch_every * 0.15
fig.legend(handles=[line0, line1, line2, line3, Patch(color='tab:red', alpha=0.4, label='Time to find target')],
           loc='upper left')
plt.grid(True)

# --- Evolution loop ---
found = False
streak = 0                 # consecutive generations with a perfect individual
for gen in range(generations):
    # Current target (alternates if n_targets > 1)
    t_index = (gen // switch_every) % n_targets if n_targets > 1 else 0
    target = targets[t_index]
    season_start = gen - gen % switch_every
    if gen == season_start:
        found = False
        streak = 0

    fitnesses = [fitness(ind, target) for ind in population]
    best = max(fitnesses)
    avg_fitness.append(sum(fitnesses) / population_size)
    max_fitness.append(best)

    streak = streak + 1 if best == 1 else 0
    if streak == 1:
        streak_start = gen
    if streak == stable_for and not found:
        found = True
        time_taken = streak_start - season_start  # when the stable perfect streak began
        ax3.bar(streak_start, time_taken, width=bar_width, color='tab:red', alpha=0.4)
        print(f"Season {gen // switch_every}: found target after {time_taken} generations")

    if gen % sample_interval == 0:
        seq_diversity.append(diversity(population))
        struct_diversity.append(diversity([get_structure(s) for s in population]))

        x_all = range(len(avg_fitness))
        x_sample = [i * sample_interval for i in range(len(seq_diversity))]
        line0.set_data(x_all, max_fitness)
        line1.set_data(x_all, avg_fitness)
        line2.set_data(x_sample, seq_diversity)
        line3.set_data(x_sample, struct_diversity)
        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()
        ax1.set_ylim(0, 1)
        best_seq = population[fitnesses.index(best)]
        fig.suptitle(f"Gen {gen}\ntarget: {target}\nbest:   {get_structure(best_seq)}", fontsize=10)
        plt.pause(0.001)

    # Selection: tournament, then fitness-proportional choice among competitors
    new_pop = []
    for _ in range(population_size):
        competitors = random.sample(range(population_size), num_competitors)
        weights = [fitnesses[i] + 1e-9 for i in competitors]  # avoid all-zero weights
        winner = random.choices(competitors, weights=weights, k=1)[0]
        new_pop.append(mutate(population[winner]))
    population = new_pop

plt.ioff()
plt.show()