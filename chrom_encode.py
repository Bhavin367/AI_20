import numpy as np 
import config 


def generate_genome():
    return np.degrees(np.random.uniform(config.low,config.high)) 

def generate_population():
    return np.array([generate_genome() for i in range(config.population_size)])


LEG_NAMES = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
JOINTS = ["coxa", "femur", "tibia"]

def print_genome(genome):
    legs = np.asarray(genome).reshape(8, 3)
    for name, leg in zip(LEG_NAMES, legs):
        joints = "   ".join(f"{j}: {a:7.2f}°" for j, a in zip(JOINTS, leg))
        print(f"{name}:  {joints}")


print_genome(generate_genome())
