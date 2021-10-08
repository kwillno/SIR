# This is the file containing the Markov-chain simulation

import numpy as np
from sirClass import SIR

sir = SIR(1000,0.005,0.01,0.1)

sir.simulate(20)

lonelySir = SIR(1,0.005,0.01,0.1)

lonelySir.numericalLimitingDistributions(years=2, v=True)

print(lonelySir.totalDays)

sir.plot()
