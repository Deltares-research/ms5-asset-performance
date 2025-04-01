from abc import ABC, abstractmethod

class GeoModelBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        raise NotImplementedError


if __name__ == "__main__":

    pass

