import sys
import numpy as np
import pandas as pd

from sklearn import preprocessing


class RPca(object):
    def __init__(self, data):
        self.L = np.zeros(data.shape)
        self.S = np.zeros(data.shape)
        self.Y = np.zeros(data.shape)
        self._initMatrix(data)      
  
    def _shrink(self, data, tau):
        return np.sign(data) * np.maximum((np.abs(data) - tau), np.zeros(data.shape))    

    def _rsvd(self, data, tau):
        U,S,V = np.linalg.svd(data, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self._shrink(S, tau)), V)) 

    def _initMatrix(self, data):
        mu = np.prod(data.shape) / (4 * np.linalg.norm(data, 1) + 1e-7)
        lmbda = 1 / np.sqrt(np.max(data.shape))
       
        threshold = 1e-7 * np.linalg.norm(data)

        err = np.Inf

        while err > threshold:
            self.L = self._rsvd(data - self.S + (1 / mu) * self.Y, 1/mu)
            self.S = self._shrink(data - self.L + ((1/mu) * self.Y), ((1/mu)* lmbda))
            self.Y = self.Y + mu * (data - self.L - self.S) 
            err = np.linalg.norm(data - self.L - self.S)

    def getL(self):
        return self.L

    def getS(self):
        return self.S
    
    def getY(self):
        return self.Y



class Scorer(object):
    def __init__(self, Y0, updater='SU'):
        Y0 = preprocessing.normalize(Y0, norm='l2', axis=0)
        #print('Y0:' +  str(Y0))
        m = Y0.shape[0]
        if m < 5:
            k = m - 1
        elif m < 10:
            k = m -1
        else:
           k = int(m/5)
        if updater == 'GU':
            self.updater = GlobalUpdate(k)
        elif updater == 'SU':
            self.updater = SketchUpdate(k)
        else:
            self.updater = RandomizedSketchUpdate(k)
        
        self.U = self.updater.update(Y0)
        print('U:' + str(self.U))        
       
    def scores(self, Y):
        Y = preprocessing.normalize(Y, norm='l2', axis=0)
        #print('Y:' + str(Y))
        #print('m:' + str(np.dot((np.identity(Y.shape[0]) - np.dot(self.U, self.U.T)), Y)))
        return np.linalg.norm(np.dot(np.identity(Y.shape[0]) - np.dot(self.U, self.U.T), Y), axis = 0, ord=2)

    def update(self, Y):
        Y = preprocessing.normalize(Y, norm='l2', axis=0)
        self.U = self.updater.update(Y)
        print('U:' + str(self.U))

class Updater(object):
    def __init__(self, k):
        self.k = k

    def update(self, Y):
        pass

class GlobalUpdate(Updater):
    def update(self, Y):
        if not hasattr(self, 'S'):
            self.U, self.S, V =  np.linalg.svd(Y, full_matrices=False)
        else:
            F = np.concatenate((np.diag(self.S), np.dot(self.U.T, Y)), axis = 1)
            U, self.S, V = np.linalg.svd(F, full_matrices=False)
            self.U = np.dot(self.U, U)
        return self.U[:, :self.k]

class SketchUpdate(Updater):
    def update(self, Y):
        if not hasattr(self, 'B'):
            D = np.empty_like(Y)
            D[:] = Y[:]
            self.ell = int(np.sqrt(Y.shape[0]))
        else:
            D = np.concatenate((self.B[:, :-1], Y), axis = 1)
 
        U,S,V = np.linalg.svd(D, full_matrices=False)
        U_ell = U[:, :self.ell]
        S_ell = S[:self.ell]
        
        self.B = np.dot(U_ell, np.diag(S_ell))
        n = np.count_nonzero(S)
        if n == 0:
            return np.zeros(Y.shape)
        elif n < self.k:
            self.k = n
        return U[:, :self.k]

class RandomizedSketchUpdate(Updater):
    def update(self, Y):
        if not hasattr(self, 'E'):
            M = np.empty_like(Y)
            M[:] = Y[:]
            self.ell = int(np.sqrt(Y.shape[0]))
        else:
            M = np.concatenate((self.E[:,:-1],Y), axis=1)

        O = np.random.normal(0., 0.1, (Y.shape[0], 100 * self.ell))
        MM = np.dot(M, M.T)
        Q, R = np.linalg.qr(np.dot(MM, O))
        S, A = np.linalg.eig(np.dot(np.dot(Q.T, MM),Q))
        order = np.argsort(S)[::-1]
        S = S[order]
        A = A[:, order]
        
        U = np.dot(Q, A)

        U_ell = U[:, :self.ell]
        S_ell = S[:sefl.ell]
       
        delta = S_ell[-1]
        S_ell = np.sqrt(S_ell - delta)
        
        self.E = np.dot(U_ell, np.diag(S_ell))

        return U[:, :self.k]

class AnomalyDetector(object):
    def __init__(self, windows):
        self.windows_len = windows
    
    def process(self, data, threshold = 0.7):
        if not hasattr(self, 'Y0'):
            self.Y0 = data
            return np.zeros(data.shape[1])
        if self.Y0.shape[1] < self.windows_len:
            self.Y0 = np.concatenate((self.Y0, data), axis = 1)
            return np.zeros(data.shape[1])
        
        if not hasattr(self, 'scorer'):
            #print(self.Y0.shape[1])
            print(self.Y0)
            rpca = RPca(self.Y0)
            self.Y0 = rpca.getL()
            print('Y0:' + str(self.Y0))
            print('rpca S:' + str(rpca.getS()))
            #print('rpca L:' + str(rpca.getL()))

            self.scorer = Scorer(self.Y0) 
            return np.zeros(data.shape[1])
        
        print(data)
        scores = self.scorer.scores(data)
        print("score:" + str(scores)) 
        if not hasattr(self, 'windows'):
            self.windows = scores
        else:
            self.windows = np.concatenate((self.windows, scores))
        
        #print(self.windows)
        if self.windows.shape[0] < self.windows_len:
            return np.zeros(data.shape[1])
        
       
        self.windows = self.windows[-self.windows_len:]
        score_wind = self.windows[-int(self.windows_len/5):]
    
        #print('windows:' + str(self.windows))
        gmean = np.mean(self.windows)
        gvar = np.var(self.windows)

        mean = np.mean(score_wind)
        print("gmean:" + str(gmean)+",gvar:" + str(gvar)+",mean:" + str(mean))
        distance  = 1 - np.exp(-1 * (mean - gmean) ** 2 / (gvar + 1e-7))
        print('distance:' + str(distance) + ",score:" + str(scores))
        if distance > threshold and scores > 0.5:
            print('detect anomaly')
        else:
            self.scorer.update(data)      


def test():
    data = pd.read_csv('./prepare/test_data')
    monitor = AnomalyDetector(24)
    for i,d in data.iterrows():
        x = np.array([d.tolist()])
        #print(x.T)
        result = monitor.process(x.T)
        #print(str(i) + ':' + str(result))

if __name__ == '__main__':
    sys.exit(test())

