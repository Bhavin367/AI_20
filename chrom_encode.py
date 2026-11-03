import numpy as np 
import config 


def generate_genome():
    return np.degrees(np.random.uniform(config.low,config.high)) 

def generate_population():
    return np.array([generate_genome() for i in range(config.population_size)])


def save_population(population, filename="population.txt"):
    np.savetxt(filename, population, fmt="%.4f")       # one chromosome per line, 4 decimals

def load_population(filename="population.txt"):
    return np.loadtxt(filename)

if __name__ == "__main__":
    pop = generate_population()
    save_population(pop)
    print(f"Saved {pop.shape[0]} chromosomes of {pop.shape[1]} genes to population.txt")
