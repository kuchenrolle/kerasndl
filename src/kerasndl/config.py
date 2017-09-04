#!/usr/bin/python3
class DefaultPreprocessingConfig:
    grams = [1,2]
    num_outcomes = 3
    unique_cues = True
    unique_outcomes = True
    lowercase = True
    remove_punctuation = True


class DefaultNetworkConfig:
    init = "zero"
    activation = "linear"
    dropout = False
    bias = False
    learning_rate = 0.01


class LearnerConfig:
    unique_cues = True
    unique_outcomes = True
    lowercase = True