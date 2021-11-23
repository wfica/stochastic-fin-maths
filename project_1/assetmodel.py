import numpy as np

class Asset:
    def __init__(self, string, prices, rate):
        self.name = string
        self.prices = prices
        self.logReturns = self.getLogReturns(self.prices)
        self.rate = rate / len(prices)
        self.calibrate()

    def getLogReturns(self, prices):
        prices = np.array(prices)
        removedFirst = prices[1:]
        removedLast = prices[:-1]
        return np.log(removedFirst / removedLast)

    def calibrate(self):
        self.beta = np.mean(self.logReturns)
        self.alpha = np.std(self.logReturns)

    def changeAlpha(self, alpha):
        self.alpha = alpha

    def changeBeta(self, beta):
        self.beta = beta

    def getName(self):
        return self.name

    def getAssetCoef(self):
        return ((self.rate - self.beta)/self.alpha) - (self.alpha/2.)

    def modelReturn(self, random):
        return np.exp(self.alpha * random + self.beta)

    def getParameters(self):
        return [self.name, self.alpha, self.beta]
