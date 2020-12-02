# Packages needed
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Inputs
temperature = ctrl.Antecedent(np.arange(10, 31, 1), 'temperature')
luminosity = ctrl.Antecedent(np.arange(2500, 10001, 100), 'luminosity')
soil_moisture = ctrl.Antecedent(np.arange(1, 11, 1), 'soil_moisture')

# Output
condition_rating = ctrl.Antecedent(np.arange(1, 11, 1), 'condition_rating')

# Membership functions
temperature.automf(3)
luminosity.automf(3)
soil_moisture.automf(3)

condition_rating['poor'] = fuzz.trimf(condition_rating.universe, [1, 1, 5])
condition_rating['mediocre'] = fuzz.trimf(condition_rating.universe, [1, 5, 10])
condition_rating['great'] = fuzz.trimf(condition_rating.universe, [5, 10, 10])

# View membership functions as graphs with matplotlib
temperature['average'].view()
luminosity['average'].view()
soil_moisture['average'].view()
condition_rating.view()

# Uncomment this line to see the graphs
# plt.show()

# Fuzzy rules
