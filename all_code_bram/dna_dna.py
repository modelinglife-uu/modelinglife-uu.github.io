import random
import math
import matplotlib.pyplot as plt

random.seed(42)

# --- PARAMETERS ---
population_size = 500
generations = 2000
mutation_rate = 0.001
no_reproduction_chance = 1
sample_interval = 100

# --- SETUP ---
def mutate(fitness):
    if random.random() < mutation_rate:
        return min(1.0, max(0.0, fitness + random.uniform(-0.1, 0.1)))
    return fitness

def calculate_diversity(population):
    """Calculate diversity of the population - NOT YET IMPLEMENTED!"""
    return 0

# Initialize
population = [0.1] * population_size
avg_fitness = []
diversity_over_time = []

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Left axis: fitness
ax1.set_xlabel("Generation")
ax1.set_ylabel("Average Fitness", color='tab:blue')
ax1.set_ylim(0, 1)
line1, = ax1.plot([], [], color='tab:blue', linewidth=2, label='Fitness')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Right axis: diversity
ax2 = ax1.twinx()
ax2.set_ylabel("Diversity", color='tab:green')
line2, = ax2.plot([], [], color='tab:green', linestyle=':', linewidth=2, label='Genotype diversity')
ax2.tick_params(axis='y', labelcolor='tab:green')

fig.suptitle("Evolution of Fitness and Diversity")
fig.tight_layout()
fig.legend(loc='upper right')
plt.grid(True)
plt.ion()

# --- EVOLUTION LOOP ---
for gen in range(generations):
    best = max(population)

    # Moran process: roulette wheel with no-reproduction option
    competitors_with_dummy = list(range(population_size)) + [None]
    probs = population + [no_reproduction_chance]

    winner = random.choices(competitors_with_dummy, weights=probs, k=1)[0]

    if winner is not None:
        dead_idx = random.randint(0, population_size - 1)
        population[dead_idx] = mutate(population[winner])

    # Sample and plot
    if gen % sample_interval == 0:
        avg_fitness.append(sum(population) / population_size)
        diversity_over_time.append(calculate_diversity(population))

        x_vals = [i * sample_interval for i in range(len(avg_fitness))]

        line1.set_data(x_vals, avg_fitness)
        line2.set_data(x_vals, diversity_over_time)
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        fig.suptitle(f"Generation {gen}: Best Fitness = {best:.3f}")
        plt.pause(0.01)

plt.ioff()
plt.show()