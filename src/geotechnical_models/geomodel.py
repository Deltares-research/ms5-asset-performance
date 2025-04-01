from abc import ABC, abstractmethod

class GeoModelBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        """
        Execute the geotechnical model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_results(self):
        """
        Extract the stage results of the geotechnical model.
        """
        raise NotImplementedError

    @abstractmethod
    def save_results(self):
        """
        Save the results of the geotechnical model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_results(self):
        """
        Load the results of the geotechnical model.
        """
        raise NotImplementedError
    
if __name__ == "__main__":
    pass

