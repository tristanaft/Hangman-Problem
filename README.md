# Hangman Problem
Python project inspired by the Trequant problem.

You are given an initial test data set of the training dictionary, and use it to predict words in a larger test dictionary.

The naive implementation is to guess the most common letter among words in the training dictionary, pruning the dictionary and recalculating based on incorrect/correct guesses. This has ~15% accuracy.

The goal is to create an algorithm that is at least 50% accurate on the test data.

I have 2 approaches so far, one that uses an RNN to predict the best letter and one that uses n-grams. They are both WIP.

I am almost done with the n-gram approach, I just need to fix the n-gram weighting.
