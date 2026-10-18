import random
import matplotlib.pyplot as plt

random.seed(12) # Ensures reproducability

side = 32  # Chessboard sized grid because why not
max_time = 250000 # Max number of generations
mutation_rate = 0.005 # Probability of mutation occurring for an individual
no_reproduction_chance = 0.1  # Probability of no reproduction event occurring
sample_interval = 500 # Moran step interval at which the plot is updated (higher = more speed)

## To keep the main loop more 'readable', we first defined functions of often-repeated procedures
# mutate fitness (uniform step +/- 0.2, clipped to [0, 1])
def mutate(fitness):
    if random.random() < mutation_rate:
        new_fit = fitness + random.uniform(-0.2, 0.2)
        return max(0.0, min(1.0, new_fit))
    return fitness

# collect all neighbours of a grid point and return that list
def get_neighbors(i, j):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            neighbors.append(((i + di) % side, (j + dj) % side))
    return neighbors

# Initialize two populations one will be on the grid, one will updated 'classic Moran' style
spatial_pop = [[0.1] * side for _ in range(side)]
mixed_pop = [0.1] * (side * side)

# Setup plot with matplotlib 
fig, (ax_grid, ax_compare) = plt.subplots(1, 2, figsize=(14, 5)) # layout with 2 sub plots
im = ax_grid.imshow(spatial_pop, cmap='viridis', vmin=0, vmax=1) # heatmap setup
ax_grid.set_title("Spatial") # title for the spatial population heatmap
fig.colorbar(im, ax=ax_grid) # colour bar for fitness

line_spatial, = ax_compare.plot([], [], label='Spatial population', linewidth=2)
line_simple, = ax_compare.plot([], [], label='Mixed population', linewidth=2, linestyle='--')
ax_compare.set_xlabel("Generation") # x-axis label for the comparison plot
ax_compare.set_ylabel("Fitness") # y-axis label for the comparison plot
ax_compare.set_ylim(0, 1) # set y-axis limits
ax_compare.legend() # show legend
ax_compare.grid(True) # enable grid for better readability

plt.ion() # enable 'interactive mode' for matplotlib to speed up the plotting during the evolution loop

avg_spatial, avg_simple = [], [] # keep two lists with averages for the plotting

# Evolution loop
for gen in range(max_time):
    # Spatial population on a grid
    i, j = random.randint(0, side-1), random.randint(0, side-1)
    neighbors = get_neighbors(i, j)
    fits = [spatial_pop[ni][nj] for ni, nj in neighbors]
    
    # Add dummy "no reproduction" event
    neighbors_with_dummy = neighbors + [None]
    probs = fits + [no_reproduction_chance]
    
    winner = random.choices(neighbors_with_dummy, weights=probs, k=1)[0]
    
    if winner is not None:  # Only reproduce if winner is not the dummy
        new_fitness = mutate(spatial_pop[winner[0]][winner[1]])
        spatial_pop[i][j] = new_fitness
    
    # Non-spatial (moran) population
    fits_moran = mixed_pop.copy()
    
    simple_with_dummy = list(range(len(mixed_pop))) + [None]
    probs_simple = fits_moran + [no_reproduction_chance]
    
    winner_moran = random.choices(simple_with_dummy, weights=probs_simple, k=1)[0]
    
    if winner_moran is not None:
        dead_idx = random.randint(0, len(mixed_pop)-1)
        mixed_pop[dead_idx] = mutate(mixed_pop[winner_moran])
    
    
    # Update plot
    if gen % sample_interval == 0:
        avg_spatial.append(sum(sum(row) for row in spatial_pop) / (side*side))
        avg_simple.append(sum(mixed_pop) / len(mixed_pop))
        x = [i*sample_interval for i in range(len(avg_spatial))]
        
        im.set_array(spatial_pop)
        line_spatial.set_data(x, avg_spatial)
        line_simple.set_data(x, avg_simple)
        ax_compare.relim()
        ax_compare.autoscale_view()
        fig.suptitle(f"Time step {gen}")
        plt.pause(0.01)

plt.show()