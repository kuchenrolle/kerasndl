#!/usr/bin/python3
class Numberer:
    """Object to index items.
    """

    def __init__(self):
        self.known = dict()
        self.items = list()
        self.current = 0
        self.add = True


    def number(self, item):
        """Get index for item provided.
        Parameters
        ----------
        item    hashable
                item to index
        """
        idx = self.known.get(item)
        if idx is None:
            if self.add:
                self.current += 1
                self.items.append(item)
                self.known[item] = self.current
                return self.current
            else:
                return 0
        else:
            return idx

    def name(self, idx):
        """Return item corresponding to index.
        Parameters
        ----------
        idx int
            index of item to retrieve
        """
        if idx < 1:
            return "UNKNOWN"
        return self.items[idx-1]

    def freeze(self):
        """Return 0 for unknown items.
        """
        self.add = False

    def unfreeze(self):
        """Return new index for unknown items.
        """
        self.add = True

    @property
    def names(self):
        return self.items