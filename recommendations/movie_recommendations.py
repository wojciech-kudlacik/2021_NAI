"""
Movie recommendations system
Authors: Wojciech Kudłacik, Norbert Daniluk

Program to find movie recommendations between users based on their similarity.
Recommendations are found using either Euclidean distance of Pearson score between users.
The program utilizes argparse as the execution method.
In order to run the program you need to supply all the required params to the exec command.
You can find examples and how tos in the README.md.
"""

import argparse
import json
import numpy as np


class MovieRecommendations:
    """
    A class to represent Movie Recommendations program.
    """
    def __init__(self, dataset: dict) -> None:
        self.dataset = dataset

    @staticmethod
    def sorted_movies(movies: dict) -> dict:
        """
        Method to sort common / not_common movies
        :param movies: Movies dictionary
        :return: sorted_movies: Sorted Movies dictionary in asc order
        """
        return {k: v for k, v in sorted(movies.items(), key=lambda item: item[1])}

    @staticmethod
    def best_movies(sorted_movies: dict) -> list:
        """
        This method finds the highest rated movies by user2
        :param sorted_movies: Sorted Movies dictionary in asc order
        :return: best_movies: List of highest rated movies by user2
        """
        return list(sorted_movies.keys())[-6:]

    @staticmethod
    def worst_movies(sorted_movies: dict) -> list:
        """
        This method finds the lowest rated movies by user2
        :param sorted_movies: Sorted Movies dictionary in asc order
        :return: best_movies: List of highest rated movies by user2
        """
        return list(sorted_movies.keys())[:6]

    def find_movies(self, user1: str, user2: str, common: bool) -> dict:
        """
        Method to find movies
        :param user1: Name and surname of the first user
        :param user2: Name and surname of the second user
        :param common: Bool to check if you want to find common or not common movies
        :return: movies: Dictionary of common or not common movies
        """
        self.__verify_users(user1, user2)
        user1_movies = self.dataset[user1]
        user2_movies = self.dataset[user2]
        if common:
            movies = {k: v for k, v in user2_movies.items() if k in user1_movies}
        else:
            movies = {k: v for k, v in user2_movies.items() if k not in user1_movies}

        return movies

    def euclidean_score(self, user1: str, user2: str) -> float:
        """
        Method to compute the Euclidean distance score between users
        :param user1: Name and surname of the first user
        :param user2: Name and surname of the second user
        :return: euclidean_score: Correlation score between users
        """
        common_movies = self.find_movies(user1, user2, True)

        if len(common_movies) == 0:
            return 0

        squared_diff = []

        for item in self.dataset[user1]:
            if item in self.dataset[user2]:
                squared_diff.append(np.square(self.dataset[user1][item] - self.dataset[user2][item]))

        euclidean_score = 1 / (1 + np.sqrt(np.sum(squared_diff)))

        return euclidean_score

    def pearson_score(self, user1: str, user2: str) -> float:
        """
        Method to compute the Pearson correlation score between users
        :param user1: Name and surname of the first user
        :param user2: Name and surname of the second user
        :return: pearson_score: Correlation score between users
        """
        common_movies = self.find_movies(user1, user2, True)

        # If there are no common movies between user1 and user2, then the score is 0
        if len(common_movies) == 0:
            return 0

        # Calculate the sum of ratings of all the common movies
        user1_sum = np.sum([self.dataset[user1][item] for item in common_movies])
        user2_sum = np.sum([self.dataset[user2][item] for item in common_movies])

        # Calculate the sum of squares of ratings of all the common movies
        user1_squared_sum = np.sum([np.square(self.dataset[user1][item]) for item in common_movies])
        user2_squared_sum = np.sum([np.square(self.dataset[user2][item]) for item in common_movies])

        # Calculate the sum of products of the ratings of the common movies
        sum_of_products = np.sum([self.dataset[user1][item] * self.dataset[user2][item] for item in common_movies])

        # Calculate the Pearson correlation score
        sxy = sum_of_products - (user1_sum * user2_sum / len(common_movies))
        sxx = user1_squared_sum - np.square(user1_sum) / len(common_movies)
        syy = user2_squared_sum - np.square(user2_sum) / len(common_movies)

        if sxx * syy == 0:
            return 0

        pearson_score = sxy / np.sqrt(sxx * syy)

        return pearson_score

    def __verify_users(self, user1: str, user2: str) -> None:
        for user in [user1, user2]:
            if user not in self.dataset:
                raise TypeError("Cannot find {0} in the dataset".format(user))


class SetupProgram:

    @staticmethod
    def __build_arg_parser() -> argparse:
        parser = argparse.ArgumentParser(description='Compute similarity score')
        parser.add_argument('--user1', dest='user1', required=True, help='First user')
        parser.add_argument('--user2', dest='user2', required=True, help='Second user')
        parser.add_argument('--dataset', dest='dataset', required=True, help='Dataset in JSON format')
        parser.add_argument("--score-type", dest="score_type", required=True,
                            choices=['Euclidean', 'Pearson'], help='Similarity metric to be used')
        return parser

    @staticmethod
    def __get_dataset_from_file(file_path: str) -> dict:
        with open(file_path, 'r') as f:
            dataset = json.loads(f.read())
            return dataset

    def run(self) -> None:
        """
        Function to run the program
        """
        args = self.__build_arg_parser().parse_args()
        dataset = self.__get_dataset_from_file(args.dataset)
        mr = MovieRecommendations(dataset)
        sorted_mv = mr.sorted_movies(mr.find_movies(args.user1, args.user2, False))

        if args.score_type == 'Euclidean':
            print("\nEuclidean score:\n{0}".format(mr.euclidean_score(args.user1, args.user2)))
        else:
            print("\nPearson score:\n{0}".format(mr.pearson_score(args.user1, args.user2)))

        print("\nMovies to watch:\n{0}".format(mr.best_movies(sorted_mv)))
        print("\nMovies not to watch:\n{0}".format(mr.worst_movies(sorted_mv)))


if __name__ == '__main__':
    SetupProgram().run()
