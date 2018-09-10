import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
from scipy.sparse import identity
import gc

I = np.eye
sparse = csc_matrix
randn = np.random.normal
choice = np.random.choice

def Is(n):
    return identity(n,format='csc')


class InferenceModule:
    """
    Kalman Filter with a decision function
    
    Parameters
    -------
    A : matrix
        The routing matrix
        
    links : dict
        The dictionary that maps switches to their links
        Ex. { 1 : [2,6], 2 : [1,3,5] } 
        which means Switch 1 is connected to the links 2 and 6,
        and Switch 2 is connected to the links 1, 3 and 5.
        
    v : float, optional
        The standard deviation of transition density
        
    e : float, optional
        The standard deviation of measurement density

    Methods
    -------
    filter(m,M)
        Predicts the estimates and updates them with the observation.
    decide(n)
        Decides which switches are going to be queried.
    _generate()
        Generates a new state conditioned on the current state, for testing purposes only.
    _measure(M)
        Partially measures the packet count vector, for testing purposes only.

    """
    def __init__(self,A,links,v=1,e=0.001):
        
        # L is the number of links
        # H2 is the number of flows
        L, H2 = A.shape
        LH2 = L + H2
        
        # The state estimate
        x = np.zeros(LH2)
        
        # The transition matrix
        F = sparse(np.bmat([
            [I(H2),   np.zeros((H2,L))],
            [A,       I(L)            ]
        ]))
        
        # The transition covariance
        Q = sparse(v**2*np.bmat([
            [I(H2),            np.zeros((H2,L))],
            [np.zeros((L,H2)), np.zeros((L,L)) ]
        ]))
        
        # The estiamte covariance
        P = sparse(np.zeros((LH2,LH2)))
        
        # The measurement matrix that measures all packet counts 
        # Used in the decision funciton
        M = sparse(np.bmat([np.zeros((L,H2)),  I(L)]))
        
        self.L, self.H2, self.LH2 = (L,H2,LH2)
        self.x, self.F, self.Q, self.P = (x,F,Q,P)

        self.links, self.M = (links,M)
        self.M = M
        
        self.v, self.e = (v,e)
        
        # The actual state vector used for testing purposes
        self._ax = np.zeros(LH2)

        gc.collect()
    
    def filter(self,m,M):
        """
        Predicts the estimates and updates them with the observation.
        
        Parameters
        -------
        m : array
            The measured packet counts
        M : matrix
            The measurement matrix
        """
        x, Q, P, F, e = (self.x, self.Q, self.P, self.F, self.e)
        
        # Predict
        x = F.dot(x)
        P = ((F.dot(P)).dot(F.T)) + Q
        
        # Update
        y = m - (M.dot(x))
        S = e**2*Is(M.shape[0]) + ((M.dot(P)).dot(M.T))
        K = (P.dot(M.T)).dot(inv(S))  # Kalman gain
        x = x + (K.dot(y))
        P = (Is(P.shape[0]) - (K.dot(M))).dot(P)
        P[P<0] = 0  # TODO: Must be tested!
        
        self.P, self.x = (P, x)

    def decide(self,n=1):
        """
        Returns the switches which are going to be queried
        and the corresponding measurement matrix
        
        Parameters
        -------
        n : int
            The number of switches to be queried
        """
        P, H2 = (self.P, self.H2)
        
        diag_cov = self.P.diagonal()[H2:]
        
        scores = []  # The switch scores
        keys = self.links.keys()
        for s in keys:
            # Sum all the variances of the packet counts of the links 
            # which are connected to the switch
            scores.append(np.sum(diag_cov[self.links[s]]))
            
        scores = scores/np.sum(scores)
        switches = choice(list(keys),n,p=scores)
        
        links = set()
        for s in switches:
            links.update(self.links[s])
        links = sorted(list(links))
        
        return switches, self.M[links,:]
    
    def _generate(self):
        """
        Generates a new state conditioned on the current state
        """
        ax, F, H2, v = (self._ax, self.F, self.H2, self.v)
        ax = F.dot(ax)
        ax[:H2] += randn(0,v,H2)
        self._ax = ax.clip(min=0)
    
    def _measure(self,M):
        """
        Partially measures the packet count vector
        
        Parameters
        -------
        M : matrix
            The measurement matrix
        """
        ax, e = (self._ax, self.e)
        y = M.dot(ax) + randn(0,e,M.shape[0])
        return y.clip(min=0)