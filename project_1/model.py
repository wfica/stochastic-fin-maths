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
