#!/usr/bin/python3
import sys
import warnings

import numpy as np
import pandas as pd

from kerasndl.model import NDL
from kerasndl.config import DefaultNetworkConfig
from kerasndl.filehandler import EventFileHandler


class Learner:
    """High-level interface between network and corpus.
    Parameters
    ----------
    event_file:     str or path
                    path to file with events
    config:         kerasndl.config, optional
                    configuration object
    init_weights:   [np.array]
                    list with array of weights to initialize network with to continue learning.
    """
    def __init__(self, event_file, config = None, init_weights = None):
        self.filehandler = EventFileHandler(event_file)

        if not config:
            config = DefaultNetworkConfig()
        config.num_inputs = self.num_cues + 1 # +1 for unknown (index = 0)
        config.num_outputs = self.num_outcomes + 1
        
        events = self.filehandler.event_generator()
        self.network = NDL(config, init_weights, events)

        self._num_events_learnt = 0


    def count_cues_outcomes(self):
        """Counts cues and outcomes of event file.
        """
        return self.filehandler.count_cues_outcomes()


    def learn(self, num_events_to_learn = None):
        """Learn the next events.
        Parameters
        ----------
        num_events_to_learn:    int, optional
                                number of events to learn, defaults to all
        """
        if num_events_to_learn == None:
            num_events_to_learn = self.num_events_left

        if num_events_to_learn > self.num_events_left:
            warnings.warn(f"Can't learn {num_events_to_learn} events, only {self.num_events_left} events left!")
            num_events_to_learn = self.num_events_left

        self.network.learn(num_events_to_learn)
        self._num_events_learnt += num_events_to_learn


    def get_weights(self, cues = None, outcomes = None, named = False):
        """Get learned weights between cues and outcomes.
        Parameters
        ----------
        cues:       list, optional 
                    cues to get weights from, defaults to all
        outcomes:   list, optional
                    outcomes to get weights to, defaults to all
        named:      boolean
                    return results as a data frame with cues/outcomes as row/column names
        """
        if not cues:
            return self.get_weights(self.cues, outcomes, named = named)
        if not outcomes:
            return self.get_weights(cues, self.outcomes, named = named)

        unknown_cues = [cue for cue in cues if not cue in self.cues]
        unknown_outcomes = [outcome for outcome in outcomes if not outcome in self.outcomes]
        cues = sorted([cue for cue in cues if cue if not cue in unknown_cues])
        outcomes = sorted([outcome for outcome in outcomes if not outcome in unknown_outcomes])

        cue_indices = [self.cue_index(cue) for cue in cues]
        outcome_indices = [self.outcome_index(outcome) for outcome in outcomes]
        weights = self.network.get_weights(cue_indices, outcome_indices)

        if named:
            weights = pd.DataFrame(weights)
            weights.index = cues
            weights.columns = outcomes

        if unknown_cues:
            sys.stderr.write(f"Unknown cues (ignored): {unknown_cues}\n")
        if unknown_outcomes:
            sys.stderr.write(f"Unknown outcomes (ignored): {unknown_outcomes}\n")

        return weights


    # requires pandas version > 0.2
    def save_as_feather(self, output_file, cues = None, outcomes = None):
        """Save weights as between cues and outcomes as feather data frame.
        Parameters
        ----------
        output_file:    str or path
                        file to save feather data frame in
        cues:           list, optional 
                        cues to get weights from, defaults to all
        outcomes:       list, optional
                        outcomes to get weights to, defaults to all        
        """
        df = self.get_weights(cues, outcomes, named = True)
        # can't save df with row names, save to column and remove
        df["cue_names"] = df.index
        df.reset_index().to_feather(output_file)


    def cue_index(self, cue):
        """Return index of for cue provided.
        """
        return self.filehandler.cue_index(cue)


    def outcome_index(self, outcome):
        """Return index for outcome provided.
        """
        return self.filehandler.outcome_index(outcome)


    @property
    def num_events_left(self):
        return self.num_events - self.num_events_learnt

    @property
    def num_events_learnt(self):
        return self._num_events_learnt

    @property
    def num_events(self):
        return self.filehandler.num_events

    @property
    def num_cues(self):
        return self.filehandler.num_cues

    @property
    def num_outcomes(self):
        return self.filehandler.num_outcomes

    @property
    def cues(self):
        return self.filehandler.cue_names

    @property
    def outcomes(self):
        return self.filehandler.outcome_names

    @property
    def info(self):
        """Information on the status the learner.
        """
        info_string = f"""Event File: {self.filehandler.event_file} \nNumber of Events: {self.num_events} ({self.num_events_learnt} learnt) \nNumber of Cues: {self.num_cues} \nNumber of Outcomes: {self.num_outcomes} \nCues and Outcomes are lowercased: {self.filehandler.lowercase} \nLearning Rate: {self.network.learning_rate}"""
        return info_string