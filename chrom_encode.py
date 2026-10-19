import numpy as np 
import config 


def generate_genome():
    return np.random.uniform(config.low,config.high) 

def generate_population():
    return np.array([generate_genome() for i in range(config.population_size)])


