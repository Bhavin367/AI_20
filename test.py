import chrom_encode as ce 
import fitness as fit
import ga_op as go 
import numpy as np

# the fitness is so small, thats why tournament selection works better since it only pick the highest out of all 
pop = ce.generate_population()

fits = np.array([fit.fitness(genome) for genome in pop])

p1 = go.tournament_select(pop, fits)
p2 = go.tournament_select(pop, fits)

c1, c2 = go.block_uniform_crossover(p1, p2)

print("parent")
print(p1)

print("\n\n")
print("child")

print(c1)


