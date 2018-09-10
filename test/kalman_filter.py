import numpy as np
from numpy.linalg import det, inv, slogdet
from numpy.linalg import matrix_rank as rank

from scipy.sparse import csc_matrix, identity, diags
from scipy.sparse.linalg import inv as invs
from scipy.stats import multivariate_normal

from copy import deepcopy as copy
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
    
    calc_ll : boolean, optional
        The boolean value that determines if the marginal log likelihood will be calculated

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
    def __init__(self,A,links,v=200,e=5,w=2000,calc_ll=False,random=False,round_robin=False,top_cov=False):
        
        # L is the number of links
        # H2 is the number of flows
        L, H2 = A.shape
        LH2 = L + H2
        
        # The state estimate
        x = np.zeros(LH2)
        x[:H2] = w
        
        # The transition matrix
        F = sparse(np.bmat([
            [I(H2),   np.zeros((H2,L))],
            [A,       I(L)            ]
        ]))
        
        # The transition covariance
        Q = sparse(np.bmat([
            [v**2*I(H2),       np.zeros((H2,L))],
            [np.zeros((L,H2)), I(L) ]
        ]))
        
        # The estiamte covariance
        P = sparse(np.bmat([
            [w**2*I(H2),    np.zeros((H2,L))],
            [np.zeros((L,H2)), I(L) ]
        ]))
        
        # The measurement matrix that measures all packet counts 
        # Used in the decision function
        M = sparse(np.bmat([np.zeros((L,H2)),  I(L)]))
        
        self.L, self.H2, self.LH2 = (L,H2,LH2)
        self.x, self.F, self.Q, self.P = (x,F,Q,P)

        self.links, self.A, self.M = (links,A,M)
        self.calc_ll, self.ll = (calc_ll,0)
        self.random, self.round_robin, self.top_cov = (random,round_robin,top_cov)      
        self.v, self.e, self.w = (v,e,w)
        
        # The actual state vector used for testing purposes
        self._ax = np.zeros(LH2)

        gc.collect()
    
    def filter(self,m,M,calc_ll=None):
        """
        Predicts the estimates and updates them with the observation.
        
        Parameters
        -------
        m : array
            The measured packet counts
        M : matrix
            The measurement matrix
        calc_ll : boolean, optional
            The boolean value that determines if the marginal log likelihood will be calculated
        """
        x, Q, P, F, e = (self.x, self.Q, self.P, self.F, self.e)
        
        # Predict
        x = F.dot(x)
        P = F.dot(P).dot(F.T) + Q
        
        # Update
        y = m - M.dot(x)
        S = e**2*Is(M.shape[0]) + M.dot(P).dot(M.T)
        K = P.dot(M.T).dot(invs(S))  # Kalman gain
        x = x + K.dot(y)
        P = (Is(P.shape[0]) - K.dot(M)).dot(P)
        
        if (calc_ll != None and calc_ll) or (calc_ll == None and self.calc_ll):
            self.ll += multivariate_normal.logpdf(y,cov=S.todense(),allow_singular=True)
            
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
        switches = list(self.links.keys())
        if len(switches) == n:
            selected_switches = switches
        elif self.random:
            selected_switches = choice(switches,n,replace=False)
        elif self.round_robin:
            if not hasattr(self, 'index'):
                self.order = np.random.choice(len(switches),len(switches),replace=False)
                self.index = 0
            indices = np.arange(self.index, self.index+n,dtype=np.int) % len(switches)
            selected_switches = [switches[self.order[i]] for i in indices]
            self.index = indices[-1]
        elif self.top_cov:
            P, H2 = (self.P, self.H2)
            diag_cov = self.P.diagonal()[H2:]
            f = np.zeros((0,self.H2))
            links = copy(self.links)
            selected_switches = []
            for i in range(n):
                scores = [np.sum(diag_cov[self.links[s]]) for s in switches]
                selected = switches.pop(np.argmax(scores))
                selected_switches.append(selected)
                
                selected_links = links.pop(selected)
                f = np.concatenate((f,self.A[selected_links,:]))
                for s in switches:
                    new_links = []
                    for l in links[s]:
                        lv = self.A[l,:]
                        coeff = f.dot(lv.T)/lv.dot(lv.T)
                        proj = f.T.dot(coeff)
                        if np.abs(np.sum(lv.T-proj)) > 0.001:
                            new_links.append(l)
                    links[s] = new_links
        else:
            M, e, H2, LH2, links =  (self.M, self.e, self.H2, self.LH2, self.links)
            P = (self.F.dot(self.P).dot(self.F.T) + self.Q).todense()
            m = np.zeros((0,LH2))
            selected_switches = []
            
            for i in range(n):
                min_s, min_entropy = -1, np.inf
                for s in switches:
                    if s in selected_switches:
                        continue
                    m_ = np.concatenate((m,M[links[s],:].todense()))
                    S = e**2*I(m_.shape[0]) + m_.dot(P).dot(m_.T)
                    K = P.dot(m_.T).dot(inv(S))  # Kalman gain
                    P_ = (I(P.shape[0]) - K.dot(m_)).dot(P)
                    _, entropy = slogdet(P_[:H2,:H2])
                    if entropy < min_entropy:
                        min_s, min_entropy, min_m = s, entropy, m_
                m = min_m
                selected_switches.append(min_s)
            gc.collect()

        link_set = set()
        for s in selected_switches:
            link_set.update(self.links[s])

        return selected_switches, self.M[sorted(list(link_set)),:]
    
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
