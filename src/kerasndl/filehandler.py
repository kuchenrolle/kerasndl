#!/usr/bin/python3
import sys

from kerasndl.utils import Numberer


class EventFileHandler:
    """Interface to manage file with events and keep track of number of cues and outcomes.
    Parameters
    ----------
    event_file: str or path
                path to file with events
    lowercase:  boolean, optional
                lowercase all cues and outcomes
    """

    def __init__(self, event_file, lowercase = True):
        self.event_file = event_file
        self.events = self.event_generator()
        self._processed = 0

        self.lowercase = lowercase
        self.cues = Numberer()
        self.outcomes = Numberer()

        self._num_events, self._num_cues, self._num_outcomes = self.count_cues_outcomes()


    def count_cues_outcomes(self):
        """Counts cues and outcomes of event file.
        """
        num_events = 0
        cues = set()
        outcomes = set()

        with open(self.event_file, "r") as events:
            _ = events.readline() # skip header
            for event in events:
                new_cues, new_outcomes = self.process_event(event)
                cues.update(new_cues)
                outcomes.update(new_outcomes)
                num_events += 1

        return num_events, len(cues), len(outcomes)


    def event_generator(self):
        """Creates a generator object that yields one event at a time from the event file. Yields empty events indefinitely.
        """
        with open(self.event_file, "r") as events:
            _ = events.readline() # skip header
            for event in events:
                yield self.process_event(event)
        # generator needs to be indefinite
        while True:
            yield [0],[0]


    # def generate_events(self, num_events_to_generate):
    #     num_remaining = self.num_events - self.processed
    #     if num_events_to_generate > num_remaining:
    #         warnings.warn(f"Can't train on {num_events_to_generate} events, only {num_remaining} events left!")
    #     self._processed = max(self.processed + num_events_to_generate, self.num_events)
    #     self.unfreeze()
    #     events = [next(self.events) for _ in range(num_events_to_generate)]
    #     self.freeze()
    #     return events


    def process_event(self, event):
        if self.lowercase:
            event = event.lower()
        cues, outcomes = event.strip().split("\t")[:2]
        cues = [self.cues.number(cue) for cue in cues.split("_")]
        outcomes = [self.outcomes.number(outcome) for outcome in outcomes.split("_")]
        return cues, outcomes


    def cue_index(self, cue):
        idx = self.cues.number(cue)
        return idx

    def outcome_index(self, outcome):
        idx = self.outcomes.number(outcome)
        return idx

    def freeze(self):
        self.cues.freeze()
        self.outcomes.freeze()

    def unfreeze(self):
        self.cues.unfreeze()
        self.outcomes.unfreeze()

    @property
    def processed(self):
        return self._processed

    @property
    def num_cues(self):
        return self._num_cues

    @property
    def num_outcomes(self):
        return self._num_outcomes

    @property
    def num_events(self):
        return self._num_events

    @property
    def cue_names(self):
        return self.cues.names

    @property
    def outcome_names(self):
        return self.outcomes.names