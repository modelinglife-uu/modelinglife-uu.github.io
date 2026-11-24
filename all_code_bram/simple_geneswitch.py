"""
simple_model.py - a gene that switches ON/OFF because of a DNA repeat that slips.
Plain Python only (no numpy). Run:  python3 simple_model.py
"""
import math
import random
from functools import lru_cache

# ---------- PART 1: DNA and the genetic code ----------
STOP_CODONS = ["TAA", "TAG", "TGA"]

def orf_length(dna):
    """Count codons from the start until the first stop codon."""
    codons = 0
    for i in range(0, len(dna) - 2, 3):
        if dna[i:i + 3] in STOP_CODONS:
            break
        codons += 1
    return codons

# ---------- PART 2: a toy gene ----------
def make_gene(n_repeats, unit="CAAT", n_codons=60, seed=1):
    """ATG + random codons + stop. The repeat is inserted near the start."""
    rng = random.Random(seed)
    sense = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"
             if a + b + c not in STOP_CODONS and a + b + c != "ATG"]
    body = "".join(rng.choice(sense) for _ in range(n_codons))
    return "ATG" + body[:30] + unit * n_repeats + body[30:] + "TAA"

FOUNDER = make_gene(6)
WORKING_LENGTH = orf_length(FOUNDER)

@lru_cache(maxsize=None)          # remembers answers so we never repeat the same work
def is_on(dna):
    """The gene is ON if the protein is still at least 90% of full length."""
    return orf_length(dna) >= 0.9 * WORKING_LENGTH

# ---------- PART 3: finding repeats ----------
@lru_cache(maxsize=None)
def find_repeats(dna, max_unit=6, min_length=8):
    """Return a list of (start, unit_size, total_length) for tandem repeats."""
    found = []
    for u in range(1, max_unit + 1):
        i = 0
        while i + 2 * u <= len(dna):
            unit = dna[i:i + u]
            n = 1
            while dna[i + n * u:i + (n + 1) * u] == unit:
                n += 1
            if n >= 2 and n * u >= min_length:
                found.append((i, u, n * u))
                i += n * u
            else:
                i += 1
    return found

# ---------- PART 4: how often does a repeat slip? ----------
SLIP_BOOST = 1   # multiply all slip rates (we raise it so that things happen in a small, fast simulation)

def slip_rate(length):
    """Chance per cell division. Longer repeats slip much more often."""
    return min(0.05, SLIP_BOOST * 1e-4 * math.exp(0.12 * (length - 40)))

def chance_of_adding(length):
    """Short repeats tend to grow, long ones tend to shrink."""
    return 0.5 if length < 50 else 0.35

# ---------- PART 5: mutations ----------
def slip(dna, repeat):
    start, unit_size, length = repeat
    end = start + length
    unit = dna[start:start + unit_size]
    if random.random() < chance_of_adding(length):
        return dna[:end] + unit + dna[end:]
    if length > unit_size:
        return dna[:end - unit_size] + dna[end:]
    return dna

def point_mutation(dna):
    i = random.randrange(len(dna))
    new_base = random.choice([b for b in "ACGT" if b != dna[i]])
    return dna[:i] + new_base + dna[i + 1:]

def small_duplication(dna, max_size=6):
    k = random.randint(1, max_size)
    i = random.randrange(len(dna) - k)
    return dna[:i + k] + dna[i:i + k] + dna[i + k:]

def small_deletion(dna, max_size=6):
    k = random.randint(1, max_size)
    i = random.randrange(len(dna) - k)
    return dna[:i] + dna[i + k:]

def mutate(dna, point_rate, indel_rate):
    for repeat in find_repeats(dna):
        if random.random() < slip_rate(repeat[2]):
            dna = slip(dna, repeat)
    if random.random() < point_rate * len(dna):
        dna = point_mutation(dna)
    if random.random() < indel_rate * len(dna):
        dna = small_duplication(dna)
    if random.random() < indel_rate * len(dna):
        dna = small_deletion(dna)
    return dna

# ---------- PART 6: the population ----------
def run(start_dna, n_cells=300, generations=1500, epoch=100, s=0.5,
        point_rate=1e-6, indel_rate=0.0, seed=1):
    """Environment flips every `epoch` generations: ON favoured, then OFF favoured."""
    random.seed(seed)
    cells = [start_dna] * n_cells
    history = []
    for gen in range(generations):
        env_wants_on = (gen // epoch) % 2 == 0
        cells = [mutate(c, point_rate, indel_rate) for c in cells]
        weights = [1 + s if is_on(c) == env_wants_on else 1.0 for c in cells]
        cells = random.choices(cells, weights=weights, k=n_cells)
        if gen % 25 == 0:
            frac_on = sum(is_on(c) for c in cells) / n_cells
            lengths = [max([r[2] for r in find_repeats(c)], default=0) for c in cells]
            longest = sum(lengths) / n_cells
            history.append((gen, frac_on, longest))
    return history, cells

# ---------- PART 7: does a repeat appear from nothing? ----------
def longest_repeat(dna):
    repeats = find_repeats(dna)
    return max(repeats, key=lambda r: r[2]) if repeats else None

def emergence_experiment(n_runs=19, generations=20000):
    """Start from a gene without a designed repeat; report the repeat unit size that wins."""
    from collections import Counter
    winners = []
    for seed in range(n_runs):
        start = make_gene(0, seed=seed + 10)
        _, cells = run(start, generations=generations, point_rate=1e-5,
                       indel_rate=1e-5, seed=seed)
        best = Counter(longest_repeat(c)[1] if longest_repeat(c) else 0 for c in cells)
        unit_size, count = best.most_common(1)[0]
        winners.append(unit_size)
        print(f"  run {seed}: most common repeat unit = {unit_size} bp ({count}/{len(cells)} cells)")
    print("  unit sizes over all runs:", dict(Counter(winners)))

if __name__ == "__main__":
    print("PART 2: copies of CAAT -> gene state")
    for n in (3, 5, 6, 7, 8, 9):
        g = make_gene(n)
        print(f"  {n} copies: protein {orf_length(g):3d} codons -> {'ON ' if is_on(g) else 'OFF'}")
    print("PART 3: repeats in the 6-copy gene:", find_repeats(FOUNDER)[:3])
    print("PART 4: slip rate by repeat length:",
          {L: f"{slip_rate(L):.0e}" for L in (12, 24, 40, 60, 80)})
    SLIP_BOOST = 200
    print("PART 6: population with a designed repeat (slip rates x200 so it is fast)")
    hist, _ = run(FOUNDER, generations=600, point_rate=0)
    for gen, on, longest in hist[::2]:
        print(f"  gen {gen:4d}  fraction ON {on:.2f}  mean longest repeat {longest:.1f} bp")

    print("PART 7: repeats from nothing (slip rates x200, point/indel rates 1e-5)")
    emergence_experiment()
