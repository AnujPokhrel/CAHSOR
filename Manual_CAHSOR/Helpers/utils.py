import math
import numpy as np
import pandas as pd

class CacheROSMessage:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = []
        self.time = []

    #add element in the cache
    def add_element(self, data):
        if len(self.cache) >= self.max_size:
            self.time.pop(0)
            self.cache.pop(0)

        self.cache.append(data)
        self.time.append(data.header.stamp.secs)

    #get all the elements in the cache    
    def get_all(self):
        return self.cache
    
    #clear the cache
    def clear_cache(self):
        self.cache = []
        self.time = []
    
    #get element at a particular index
    def get_index(self, index):
        return self.cache[index]

    #get element from a particular time or return None
    def get_element_from_time(self, time):
        if time not in self.time:
            return None

        index = self.time.index(time)
        return self.cache[index]

    #get the oldest element in the cache       
    def get_oldest_element(self):
        return self.cache[0]
   
    #get the latest n elements in the cache, if n > cache size, return None
    def get_last_n_elements(self, n):
        if n > len(self.cache):
            return self.get_all()

        return self.cache[-n:]

class CacheHeaderlessROSMessage:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = []

    #add element in the cache
    def add_element(self, data):
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)

        self.cache.append(data)

    #get all the elements in the cache    
    def get_all(self):
        return self.cache
    
    #clear the cache
    def clear_cache(self):
        self.cache = []
        self.time = []
    
    #get element at a particular index
    def get_index(self, index):
        return self.cache[index]

    #get the oldest element in the cache       
    def get_oldest_element(self):
        return self.cache[0]

    def get_last_element(self):
        if len(self.cache) == 0:
            return None
        
        return self.cache[-1]

    #get the latest n elements in the cache, if n > cache size, return None
    def get_last_n_elements(self, n):
        if n > len(self.cache):
            return self.get_all()

        return self.cache[-n:]