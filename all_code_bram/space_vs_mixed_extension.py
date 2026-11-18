import random
import matplotlib.pyplot as plt

random.seed(12) 

side = 32  
max_time = 250000 
mutation_rate = 0.005 
no_reproduction_chance = 0.1  
sample_interval = 500 

nutrient_cost = 0.2     
nutrient_inflow = 0.01  


def mutate(fitness):
    if random.random() < mutation_rate:
        new_fit = fitness + random.uniform(-0.2, 0.2)
        return max(0.0, min(1.0, new_fit))
    return fitness

def get_neighbors(i, j):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            neighbors.append(((i + di) % side, (j + dj) % side))
    return neighbors

spatial_pop = [[0.1] * side for _ in range(side)]
mixed_pop = [0.1] * (side * side)
nutrients = [[1.0] * side for _ in range(side)]
mixed_nut = 1.0  # well-mixed population shares one global nutrient pool (= mean nutrient level)
n_sites = side * side

# Setup plot with matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax_grid, ax_nut, ax_compare) = plt.subplots(
    1, 3, gridspec_kw={'width_ratios': [1, 1, 1.25]},
    constrained_layout=True)

def style_grid(ax, title):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)

im = ax_grid.imshow(spatial_pop, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
style_grid(ax_grid, "Spatial population (fitness)")
fig.colorbar(im, ax=ax_grid, fraction=0.046, pad=0.03, label="Fitness")

im_nut = ax_nut.imshow(nutrients, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
style_grid(ax_nut, "Nutrients")
fig.colorbar(im_nut, ax=ax_nut, fraction=0.046, pad=0.03, label="Nutrient level")

line_spatial, = ax_compare.plot([], [], label='Spatial population', linewidth=2.2, color='#2a9d8f')
line_simple, = ax_compare.plot([], [], label='Mixed population', linewidth=2.2, linestyle='--', color='#e76f51')
line_nut, = ax_compare.plot([], [], label='Mean nutrients (spatial)', linewidth=1.8, linestyle=':', color='#264653')
line_nut_mixed, = ax_compare.plot([], [], label='Mean nutrients (mixed)', linewidth=1.8, linestyle='-.', color='#e9c46a')
ax_compare.set_title("Mean fitness over time", fontsize=12, fontweight='bold')
ax_compare.set_xlabel("Generation")
ax_compare.set_ylabel("Mean fitness / nutrient level")
ax_compare.set_ylim(0, 1)
ax_compare.set_xlim(0, max_time)
ax_compare.legend(loc='upper left', frameon=True)

plt.ion() # enable 'interactive mode' for matplotlib to speed up the plotting during the evolution loop

avg_spatial, avg_simple, avg_nut, avg_nut_mixed = [], [], [], [] # keep two lists with averages for the plotting

# Evolution loop
for gen in range(max_time):
    # Spatial population on a grid
    i, j = random.randint(0, side-1), random.randint(0, side-1)
    neighbors = get_neighbors(i, j)
    #fits = [spatial_pop[ni][nj] for ni, nj in neighbors]
    fits = [spatial_pop[ni][nj] * nutrients[ni][nj] for ni, nj in neighbors]
    # Add dummy "no reproduction" event
    neighbors_with_dummy = neighbors + [None]
    probs = fits + [no_reproduction_chance]
    
    winner = random.choices(neighbors_with_dummy, weights=probs, k=1)[0]
    
    if winner is not None:
        wi, wj = winner
        parent_fit = spatial_pop[wi][wj]
        spatial_pop[i][j] = mutate(parent_fit)
        nutrients[wi][wj] = max(0.0, nutrients[wi][wj] - nutrient_cost * parent_fit)

    ri, rj = random.randint(0, side-1), random.randint(0, side-1)
    nutrients[ri][rj] = min(1.0, nutrients[ri][rj] + nutrient_inflow)
    
    # Non-spatial (moran) population
    fits_moran = mixed_pop.copy()
    
    simple_with_dummy = list(range(len(mixed_pop))) + [None]
    probs_simple = [f * mixed_nut for f in fits_moran] + [no_reproduction_chance]
    
    winner_moran = random.choices(simple_with_dummy, weights=probs_simple, k=1)[0]
    
    if winner_moran is not None:
        dead_idx = random.randint(0, len(mixed_pop)-1)
        parent_fit_m = mixed_pop[winner_moran]
        mixed_pop[dead_idx] = mutate(parent_fit_m)
        # same total nutrient budget as the grid, expressed as a per-site mean
        mixed_nut = max(0.0, mixed_nut - nutrient_cost * parent_fit_m / n_sites)
    mixed_nut = min(1.0, mixed_nut + nutrient_inflow / n_sites)
    
    
    # Update plot
    if gen % sample_interval == 0:
        avg_spatial.append(sum(sum(row) for row in spatial_pop) / (side*side))
        avg_simple.append(sum(mixed_pop) / len(mixed_pop))
        avg_nut.append(sum(sum(row) for row in nutrients) / (side*side))
        avg_nut_mixed.append(mixed_nut)
        x = [k*sample_interval for k in range(len(avg_spatial))]
        
        im.set_array(spatial_pop)
        im_nut.set_array(nutrients)
        line_spatial.set_data(x, avg_spatial)
        line_simple.set_data(x, avg_simple)
        line_nut.set_data(x, avg_nut)
        line_nut_mixed.set_data(x, avg_nut_mixed)
        fig.suptitle(f"Time step {gen:,}", fontsize=14)
        plt.pause(0.01)

plt.ioff()
plt.show()