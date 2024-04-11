# Hangman Problem
Python project inspired by the Trequant problem.

## Basic Setup

As far as the actual game the setup is this, at the start you are given blank spaces for each letter in the target word and 6 lives. You guess a letter, and if it is correct the blanks corresponding to the correct letter are replaced, and if you guess incorrectly you lose a life.

Instead of being given the entire dictionary, you are given an initial test data set of the training dictionary and must use it to predict words in a larger test dictionary.

## Solving the problem

The naive implementation is to guess the most common letter among words in the training dictionary, pruning the dictionary and recalculating based on incorrect/correct guesses. This has ~15% accuracy. I tried an algorithm that guesses the letter that maximizes information gain and that performed about the same. The goal is to create an algorithm that is at least 50% accurate on the test data.

I have 2 approaches so far, one that uses an RNN to predict the best letter and one that uses n-grams. They are both WIP. I am almost done with the n-gram approach, I just need to fix the n-gram weighting.
