import numpy as np
from scipy.stats import norm

class Option:
    NUM_TRADING_DAYS = 250
    def __init__(self, prices, rate, strike, trading_days_till_expiry):
        self.E = strike
        self.sigma = self.getSigma(self.getReturns(prices))
        self.r = rate
        self.trading_days_till_expiry = trading_days_till_expiry

    def getReturns(self, prices):
        prices = np.array(prices)
        removedFirst = prices[1:]
        removedLast = prices[:-1]
        return (removedFirst - removedLast) / removedLast

    def getSigma(self, returns):
        return np.std(returns, ddof = 1) * np.sqrt(len(returns))

    def update(self, price, time):
        self.S = price
        self.time = time/self.NUM_TRADING_DAYS

    def d1(self):
        return (np.log(self.S/self.E) + (self.r + 0.5*(self.sigma*self.sigma)) * self.time) / \
                (self.sigma * np.sqrt(self.time))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.time)


class CallOption(Option):
    def payoff(self, trajectory):
        return max(0, trajectory[-1] - self.E)

    def value(self):
        return self.S * norm.cdf(self.d1()) - \
                self.E * np.exp(-self.r*self.time) * norm.cdf(self.d2())

    def delta(self):
        return norm.cdf(self.d1())

    def gamma(self):
        return norm.pdf(self.d1()) /\
                (self.sigma * self.S * np.sqrt(self.time))

class PutOption(Option):
    def payoff(self, trajectory):
        return max(0, self.E - trajectory[-1])

    def value(self):
        return -self.S * norm.cdf(-self.d1()) - \
                self.E * np.exp(-self.r*self.time) * norm.cdf(-self.d2())

    def delta(self):
        return norm.cdf(self.d1()) - 1

    def gamma(self):
        return norm.pdf(self.d1()) /\
                (self.sigma * self.S * np.sqrt(self.time))

class UAOOption:
    def __init__(self, strike, barrier):
        self.strike = strike
        self.barrier = barrier

    def payoff(self, trajectory):
        return max(0, trajectory[-1] - self.strike) if max(trajectory) < self.barrier else 0

class UAIOption:
    def __init__(self, strike, barrier):
        self.strike = strike
        self.barrier = barrier

    def payoff(self, trajectory):
        return max(0, trajectory[-1] - self.strike) if max(trajectory) >= self.barrier else 0

class ParisOption:
    def __init__(self, strike, barrier, dayAmount):
        self.strike = strike
        self.barrier = barrier
        self.dayAmount = dayAmount

    def payoff(self, trajectory):
        counter = 0
        optionExec = False
        for p in trajectory:
            if p > self.barrier:
                counter += 1
            else:
                counter = 0
            if counter == 10:
                optionExec = True
        if optionExec:
            return max(0, trajectory[-1]-2200)
        return 0

class LookbackOption:
    def __init__(self):
        pass

    def payoff(self, wig20trajectory, kghmTrajectory):
        if kghmTrajectory[-1] > kghmTrajectory[0]:
            return wig20trajectory[-1] - min(wig20trajectory)
        return 0
