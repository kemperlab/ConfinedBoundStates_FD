import numpy as np

class probes_f:

    def __init__(self, sigma, t0, amp, w, t):

        self.sigma = sigma
        self.t0 = t0
        self.amp = amp
        self.w = w
        self.t = t

    def probe(self, n):

        if n == 'GAUSSIAN':
            return self.gaussian_hit()
        elif n == 'DELTA':
            return self.delta_hit()
        elif n == 'SELECTIVE':
            return self.f_selective()

    def gaussian_hit(self):

        return self.amp * (np.exp(-(self.t - self.t0) ** 2 / (2 * self.sigma ** 2)))

    def delta_hit(self):
        
        d_h = np.zeros(len(self.t))
        d_h[int((self.t0)*len(self.t)/self.t[-1])] = self.amp
      
        return d_h

    def f_selective(self):

        return self.amp * (np.exp(-(self.t - self.t0) ** 2 / (2 * self.sigma ** 2))) * np.sin(self.w * (self.t-self.t0))