# Assignment 4 - Movie Recommendations
The goal of this assignment was to implement a movie recommendation system that would utilize **Euclidean distance** and **Pearson score**. Additionally, the solution was to be documented using **Docstring**.

## Setup
In order to run the solution you need to have numpy installed: 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all it.

### Installing numpy
```bash
pip install numpy
```

## How to run
The program utilizes **argparse** package to supply additional argumants to standard python script execution method.
The list of arguments:
1. --user1: User **to** whom you want to recommend movies. *Example*: "Name Surname"
2. --user2: User **that recommends** movies to the first user. *Example*: "Name Surname"
3. --dataset: File with Json formated data with movies. *Example*: path/to/file.json
4. --score-type: Euclidean distance or Pearson score *Example*: Pearson

```bash
python3 movie_recommendations.py --user1 "Name Surname" --user2 "Name Surname" --dataset ratings.json --score-type Pearson
```

## Results
### Euclidean
![Imgur](https://i.imgur.com/0orEuUi.png)
![Imgur](https://i.imgur.com/VNwYDaT.png)

### Pearson
![Imgur](https://i.imgur.com/Q8YvQBR.png)
![Imgur](https://i.imgur.com/eu871hu.png)


## Useful Links

### Docstring
[Python Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
[Python Docstring Tutorial](https://www.datacamp.com/community/tutorials/docstrings-python)


