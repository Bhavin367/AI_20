import chrom_encode as ce 
import fitness as fit 


ge = ce.generate_genome()

print(ge)

print(fit.limit_cost(ge))

