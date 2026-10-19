import numpy as np
import config 

def limit_cost(genome):
    too_high = genome > config.high # gives something liek [ true false false ] 
    too_low = genome < config.low  

    return np.sum(genome[too_high] - config.high[too_high]) + np.sum(genome[too_low] - genome[too_low]) 

# genome[true false false] - high[tre false false ] 
# gives [12] - [2] , picks only stuff that gave true and returns the difference 


