from cmath import nan
import numpy as np
from typing import List
import matplotlib.pyplot as plt

class Logistic_reg:
    def __init__(self,alpha, n : int):
        self.alpha = alpha
        self.betas = np.array([np.random.uniform()*1e-6 for i in range(n+1)])
        pass

    def h_b(self,x) -> float:
        y = np.sum(self.betas[1:] * x) + self.betas[0]
        exp = 1+np.exp(y)
        return 1/exp
    
    def dh_b(self,h_b : float):
        return h_b*(1-h_b)

    def cost_func(self,h,y):
        return (-np.log10(h)*y) + (-np.log10(1-h)*(1-y))

    def dy_cost_func(self,h : float,y : bool):
        temp = y-h
        return temp

    def teach(self, x : List[np.ndarray], y : List[bool], num_of_iter : int):
        # if x.shape!=self.betas[1:].shape:
        #    raise ValueError("dimensions not match!")
        for k in range(num_of_iter):
            sum = np.array([0.0 for i in range(len(self.betas))])
            for n,i in enumerate(x):
                h = self.h_b(i)
                dcost = self.dy_cost_func(h,y[n])
                # if dcost == np.NaN:
                #     dcost = 0
                try:
                    sum[0] += dcost
                    #print()#sum[0]
                except:
                    dcost += 0
                for j,_ in enumerate(sum[1:]):
                    sum[j+1] += dcost*i[j]
                
            print(sum/len(y))
            dJ = (sum/len(y))
            if np.sqrt(np.sum(dJ*dJ)) < (1e-13):
                print((1e-8) *self.alpha)
                break
            if self.alpha>100*np.sqrt(np.sum(dJ*dJ)):
                self.alpha = self.alpha/10
            self.betas -= self.alpha*dJ
        pass

    def predict(self,x):
        return self.h_b(x) > 0.5

    def get_line_param(self):
        return -(self.betas[1]/self.betas[2]), -(self.betas[0]/self.betas[2])