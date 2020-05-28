from abc import abstractmethod

class Metrics:
    def __init__(self,name='Metric'):
        self.name = name
        
    def get_name(self):
        return self.name
    
    def set_name(self,name):
        self.name=name
        
    @abstractmethod
    def get_score(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass

