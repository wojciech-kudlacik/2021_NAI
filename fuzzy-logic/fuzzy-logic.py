# Fuzzy Logic Potted Plant Conditions Rating
# Author: Wojciech Kudłacik and Norbert Daniluk
# Inspiration: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


'''
List of inputs 

They hold universe values and membership functions 
'''
temperature = ctrl.Antecedent(np.arange(10, 31, 1), 'temperature')
luminosity = ctrl.Antecedent(np.arange(2500, 10001, 100), 'luminosity')
soil_moisture = ctrl.Antecedent(np.arange(0, 11, 1), 'soil_moisture')

'''
List of outputs

They hold universe values and membership functions
'''
condition_rating = ctrl.Consequent(np.arange(0, 11, 1), 'condition_rating')

'''
Inputs membership functions population

In this case functions are populated automatically using .automf function 
'''
temperature.automf(3)
luminosity.automf(3)
soil_moisture.automf(3)


'''
Output membership function population

In this case functions are populated manually using .trimf function 
'''
condition_rating['poor'] = fuzz.trimf(condition_rating.universe, [0, 0, 5])
condition_rating['great'] = fuzz.trimf(condition_rating.universe, [0, 5, 10])
condition_rating['mediocre'] = fuzz.trimf(condition_rating.universe, [5, 10, 10])

'''
View membership functions as graphs with matplotlib

Uncomment these lines if you want to see them
'''
# temperature['average'].view()
# luminosity['average'].view()
# soil_moisture['average'].view()
# condition_rating.view()

'''
List of rules for the fuzzy logic
'''
rule1 = ctrl.Rule(temperature['poor'] & luminosity['poor'] & soil_moisture['poor'], condition_rating['poor'])
rule2 = ctrl.Rule(temperature['average'] & luminosity['average'] & soil_moisture['average'], condition_rating['mediocre'])
rule3 = ctrl.Rule(temperature['good'] & luminosity['good'] & soil_moisture['good'], condition_rating['great'])

'''
View rules as graphs with matplotlib

Uncomment these lines if you want to see them
'''
# rule1.view()
# rule2.view()
# rule3.view()

'''
Create an instance of ControlSystem with rules
'''
condition_rating_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

'''
Object representing condition_rating_ctrl applied to our set of conditions
'''
plant = ctrl.ControlSystemSimulation(condition_rating_ctrl)

'''
Ask for input from the user

Input is being cast as int and passed down to ControlSystemSimulation.input
'''
temperature_value = int(input("Enter temperature value (10 - 30): "))
luminosity_value = int(input("Enter luminosity value (2500 - 10000): "))
soil_moisture_value = int(input("Enter soil moisture value (0 - 10): "))

plant.input['temperature'] = temperature_value
plant.input['luminosity'] = luminosity_value
plant.input['soil_moisture'] = soil_moisture_value

'''
Compute the result and print it to the console
'''
plant.compute()
print(plant.output['condition_rating'])
