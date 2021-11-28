import assetmodel
import numpy as np
import pandas as pd

class Model:
    RNDM = -0.5 # Radon-Nikodym derivative mean
    def __init__(self, assets, rate):
        self.assets = []
        self.rate = rate
        if type(assets) == dict:
            self.processDictAssets(assets)
        else:
            raise NotImplementedError("Model currently only accepts dictionaries.")
        self.calculateDistribution()

    def processDictAssets(self, assets):
        for key, value in assets.items():
            self.assets.append(assetmodel.Asset(key, value, self.rate))

    def calculateDistribution(self):
        allPrices = []
        for asset in self.assets:
           allPrices.append(asset.logReturns)
        noc = len(allPrices)
        corrcoef = np.corrcoef(allPrices)
        self.covMat = np.zeros((noc+1, noc+1))
        self.covMat[:-1, :-1] = corrcoef
        for i in range(noc):
            coef = self.assets[i].getAssetCoef()
            self.covMat[i][-1] = coef
            self.covMat[-1][i] = coef
        self.covMat[-1][-1] = 1
        self.means = np.zeros(noc+1)
        self.means[-1] = self.RNDM

    def changeAssetParameter(self, name, alpha = None, beta = None):
        for asset in self.assets:
            if asset.name == name:
                if alpha != None:
                    asset.changeAlpha(alpha)
                if beta != None:
                    asset.changeBeta(beta)
                break

    def getAssetParameters(self, name = None):
        checkCondition = name == None
        parameters = []
        for asset in self.assets:
            if checkCondition or name == asset.name:
                parameters.append(asset.getParameters())
        return pd.DataFrame(parameters, columns = ["Name", "alpha", "beta"])



    def getAssetIdx(self, name):
        if name == None:
            return -1
        for i in range(len(self.assets)):
            if self.assets[i].name == name:
                return i

    def changeCorrelation(self, name1, value, name2 = None):
        idx1 = self.getAssetIdx(name1)
        idx2 = self.getAssetIdx(name2)
        self.covMat[idx1][idx2] = value
        self.covMat[idx2][idx1] = value

    def MonteCarloRealRN(self, iters):
        """ Produces random trajectories for each asset.
        Args:
            - iter: number of trading days to generate. 
        Returns: 
            - a tuple (random trajectories, RN derivative).  
            trajectories.shape = (iters+1, number of assets)
        """
        noa = len(self.assets)
        trajectories = np.zeros((iters+1, noa))
        K = 1.
        for i, asset in enumerate(self.assets):
            trajectories[0][i] = asset.prices[-1]
        for i in range(1, iters+1):
            randomSample = np.random.multivariate_normal(self.means, self.covMat)
            for j in range(noa):
                trajectories[i][j] = trajectories[i-1][j]*self.assets[j].modelReturn(randomSample[j])
            K *= randomSample[noa]
        return trajectories, np.exp(K)

    def MonteCarloQ(self, iters):
        mis = np.zeros(len(self.assets))
        for i, a in enumerate(self.assets):
            mis[i] = a.getAssetCoef()
        allPrices = []
        for asset in self.assets:
           allPrices.append(asset.logReturns)
        corrcoef = np.exp(np.corrcoef(allPrices))/2
        for i in range(len(self.assets)):
            corrcoef[i][i] = 1
            for j in range(i+1, len(self.assets)):
                corrcoef[i][j] -= mis[i]*mis[j]
                corrcoef[j][i] -= mis[i]*mis[j]
        trajectories = np.zeros((iters+1, len(self.assets)))
        for i, asset in enumerate(self.assets):
            trajectories[0][i] = asset.prices[-1]
        for i in range(1, iters+1):
            randomSample = np.random.multivariate_normal(mis, corrcoef)
            for j in range(len(self.assets)):
                trajectories[i][j] = trajectories[i-1][j]*self.assets[j].modelReturn(randomSample[j])
        return trajectories

    def PriceOptionR(self, option, base_asset_idx, NUM_ITER = 100):
        s = 0
        # trajectories_hist = []
        for _ in range(NUM_ITER):
            trajectories, rd = self.MonteCarloRealRN(option.trading_days_till_expiry)
            s += option.payoff(trajectories[:,base_asset_idx])*rd/np.exp(option.r * option.trading_days_till_expiry / option.NUM_TRADING_DAYS)
            # trajectories_hist.append(trajectories)
        
        return s / NUM_ITER #, trajectories_hist

    def PriceOptionQ(self, option, base_asset_idx, NUM_ITER = 100):
        s = 0
        # trajectories_hist = []
        for _ in range(NUM_ITER):
            trajectories = self.MonteCarloQ(option.trading_days_till_expiry)
            s += option.payoff(trajectories[:,base_asset_idx])
            # trajectories_hist.append(trajectories)

        return s / NUM_ITER #, trajectories_hist
