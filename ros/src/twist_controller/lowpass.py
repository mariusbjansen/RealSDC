
class LowPassFilter(object):
    def __init__(
        self,
        samples_num): # Number of samples used for averaging

        self.a = 1.0 / (samples_num + 1.0)
        self.b = samples_num / (samples_num + 1.0)

        self.is_initialized = False
        self.last_val = 0.0

    def get(self):
        return self.last_val

    def filt(self, val):
        if self.is_initialized:
            val = self.a * val + self.b * self.last_val
        else:
            self.is_initialized = True

        self.last_val = val
        
        return val
