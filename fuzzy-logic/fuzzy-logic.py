# Packages needed
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Inputs
temperature = ctrl.Antecedent(np.arange(10, 31, 1), 'temperature')
luminosity = ctrl.Antecedent(np.arange(2500, 10001, 100), 'luminosity')
soil_moisture = ctrl.Antecedent(np.arange(0, 11, 1), 'soil_moisture')

# Output
condition_rating = ctrl.Consequent(np.arange(0, 11, 1), 'condition_rating')

# Membership functions
temperature.automf(3)
luminosity.automf(3)
soil_moisture.automf(3)

condition_rating['poor'] = fuzz.trimf(condition_rating.universe, [0, 0, 5])
condition_rating['great'] = fuzz.trimf(condition_rating.universe, [0, 5, 10])
condition_rating['mediocre'] = fuzz.trimf(condition_rating.universe, [5, 10, 10])

# View membership functions as graphs with matplotlib
temperature['average'].view()
luminosity['average'].view()
soil_moisture['average'].view()
condition_rating.view()

# Uncomment this line to see the graphs
# plt.show()

# Fuzzy rules
rule1 = ctrl.Rule(temperature['poor'] & luminosity['poor'] & soil_moisture['poor'], condition_rating['poor'])
rule2 = ctrl.Rule(temperature['average'] & luminosity['average'] & soil_moisture['average'], condition_rating['mediocre'])
rule3 = ctrl.Rule(temperature['good'] & luminosity['good'] & soil_moisture['good'], condition_rating['great'])

rule1.view()
# plt.show()

# Control System
condition_rating_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# Control System Simulation
plant = ctrl.ControlSystemSimulation(condition_rating_ctrl)

# Input values / conditions
plant.input['temperature'] = 20
plant.input['luminosity'] = 5000
plant.input['soil_moisture'] = 5

plant.compute()
print(plant.output['condition_rating'])
# condition_rating.view(sim=plant)
plt.show()

