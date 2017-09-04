#!/usr/bin/python3
import os
import gzip
import string

from kerasndl.config import DefaultPreprocessingConfig


class Preprocessor:
    """Interface to prepare event files.
    Parameters
    ----------
    config  kerasndl.config object
            specification of event structure
    """

    def __init__(self, config = None):
        if not config:
            config = DefaultPreprocessingConfig()
        self.config = config


    def process_file(self, in_file, out_file, binary = False):
        """Processes file line by line into events.
        Parameters
        ----------
        in_file:    str or path
                    corpus file to process, one sentence per line.
        out_file:   str or path
                    file to save events to
        binary      boolean, optional
                    store events in binary format (for pyndl), defaults to false
        """
        if binary:
            out_file, tmp = "_tmp", out_file
        with open(in_file, "r") as in_file:
            with open(out_file, "w") as out_file:
                out_file.write("Cues\tOutcomes\tFrequency\n")
                for line in in_file:
                    for event in self.line_to_events(line):
                        out_file.write(event + "\n")
        if binary:
            self.events_to_binary("_tmp", tmp)
            os.remove("_tmp")


    def events_to_binary(self, event_file, out_file):
        """Transform traditional event file to binary (for pyndl).
        """
        with gzip.open(out_file, "wb") as out_file:
            with open(event_file, "r") as event_file:
                for line in event_file:
                    out_line = line.rsplit("\t",1)[0]+"\n"
                    out_file.write(str.encode(out_line))


    def line_to_events(self, line):
        """Turn line into set of events
        Parameters
        ----------
        line    str
        """
        if self.config.lowercase:
            line = line.lower()

        if self.config.remove_punctuation:
            table = str.maketrans("", "", string.punctuation)
            line = line.translate(table)

        tokens = line.strip().split(" ")
        tokens = [token for token in tokens if token] # remove empty tokens

        length = self.config.num_outcomes
        num = len(tokens) - length + 1
        outcomes = [tokens[i:i+length] for i in range(num)]
        outcomes = ["_".join(_outcomes) for _outcomes in outcomes]
        cues = [self.make_cues(outcomes) for outcomes in outcomes]

        for _cues, _outcomes in zip(cues,outcomes):
            event = "\t".join([_cues, _outcomes, "1"])
            yield event


    def make_cues(self, outcomes):
        """Extract cues contained in string.
        """
        outcomes = str.replace(outcomes, "_", "#")

        all_cues = list()
        for length in self.config.grams:
            num = len(outcomes) - length + 1
            cues = [outcomes[i:i+length] for i in range(num)]
            all_cues += cues

        return "_".join(all_cues)
