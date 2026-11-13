import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

N = 100 # Default population size 
mu = 0.001 # Default mutation rate
# Updated parameters
steps = 50000

def simulate(N=N,mut=mu): 
    avg_fitness = []
    fitnesses = np.full(N, 0.05)
    # Moran process with mutation (logging every 10 steps)
    for step in range(steps):
        probs = fitnesses / fitnesses.sum()
        parent = np.random.choice(N, p=probs)
        dead = np.random.choice(N)

        # Copy with mutation
        new_fit = fitnesses[parent]
        if np.random.rand() < mut:
            new_fit = np.clip(new_fit + np.random.normal(0, 0.1), 0, 1)
                
        fitnesses[dead] = new_fit

        # Save average fitness every 10 steps
        if step % 10 == 0:
            avg_fitness.append(fitnesses.mean())
            
    # Plotting
    plt.plot(np.arange(0, steps, 10), avg_fitness, label=f"mu={mut}")

simulate(N,0.0001)
simulate(N,0.001)
simulate(N,0.01)
simulate(N,0.1)
simulate(N,1)

# Plotting
plt.xlabel("Step")
plt.ylabel("Average fitness")
plt.title("Evolution of Fitness in a Moran Process")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()