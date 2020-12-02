- [Assignment 3 - Fuzzy Logic](#assignment-3---fuzzy-logic)
  * [How to run](#how-to-run)
    + [Installing numpy](#installing-numpy)
    + [Installing skfuzzy](#installing-skfuzzy)
    + [Installing matplotlib](#installing-matplotlib)
  * [Problem Description](#problem-description)
    + [Inputs, Output, Rules and Usage](#inputs--output--rules-and-usage)
  * [Membership Functions visualization](#membership-functions-visualization)
    + [Temperature](#temperature)
    + [Luminosity](#luminosity)
    + [Soil Moisture](#soil-moisture)
    + [Conditions Rating](#conditions-rating)
    + [Rules Visualization](#rules-visualization)
  * [Useful Links](#useful-links)
    + [Fuzzy Logic](#fuzzy-logic)
    + [SkFuzzy](#skfuzzy)
    + [Docstring](#docstring)

# Assignment 3 - Fuzzy Logic
The goal of this assignment was to implement fuzzy logic algorithm using **skfuzzy** library. Additionally, the solution was to be documented using **Docstring**.

## How to run
In order to run the solution you need to have three packages installed: 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all of them.

### Installing numpy
```bash
pip install numpy
```
### Installing skfuzzy
```bash
pip install -U scikit-fuzzy
```
### Installing matplotlib
```bash
pip install matplotlib
```

## Problem Description
The program estimates the conditions rating (on a scale from 0 to 10) for a potted plant by taking in and "fuzzyfing" three inputs: temperature in the room, luminosity and soil moisture.

The ideal conditions for a plant would be the values from the middle of the given input range - when there's not too cold or hot, not too dark or bright or when the soil is not too dry or too wet.

### Inputs, Output, Rules and Usage
**Universe:** crisp value range
**Fuzzy set:** fuzzy value range

**Antecedents (Inputs)**
- Temperature
    - **Universe:** What is the current temperature on a scale of 10 to 30 (Celcius)? 
    - **Fuzzy set:** cold, hot, perfect
- Luminosity
    - **Universe:** What is the current brightness on a scale of 2500-10000 (Lux)?
    - **Fuzzy set:** dark, bright, perfect
- Soil Moisture 
    - **Universe:** What is the current soil moisture on a scale of 1 to 10?
    - **Fuzzy set:** dry, wet, perfect

**Consequents (Outputs)**
- Condition rating
    - **Universe:** How good are the conditions on a scale of 0 to 10?
    - **Fuzzy set:** poor, mediocre, great

**Rules**
- **IF** the *temperature* was good and the *luminosity* was good and *soil moisture* was good **THEN** the condition rating will be great
- **IF** the *temperature* was average and the *luminosity* was average and *soil moisture* was average **THEN** the condition rating will be mediocre
- **IF** the *temperature* was poor and the *luminosity* was poor and *soil moisture* was poor **THEN** the condition rating will be poor

**Usage**
- If I tell this controller that:
    - The temperature was: **20**
    - The luminosity was: **6250**
    - The soil moisture was: **5**

**The result would be:**  +/- 8.3

## Membership Functions visualization
### Temperature
![Imgur](https://i.imgur.com/bbKTcN2l.png)

### Luminosity
![Imgur](https://i.imgur.com/829Syhul.png)

### Soil Moisture
![Imgur](https://i.imgur.com/fQxjN4Fl.png)

### Conditions Rating
![Imgur](https://i.imgur.com/hoWe4YFl.png)

### Rules Visualization
![Imgur](https://i.imgur.com/zLmuHIhl.png)


## Useful Links
### Fuzzy Logic
[Fuzzy Logic](https://en.wikipedia.org/wiki/Fuzzy_logic)

### SkFuzzy
[SkFuzzy Documentation](https://pythonhosted.org/scikit-fuzzy/)

### Docstring
[Python Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
[Python Docstring Tutorial](https://www.datacamp.com/community/tutorials/docstrings-python)


