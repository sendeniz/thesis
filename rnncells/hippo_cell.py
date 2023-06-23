import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import math
from scipy import linalg as la
from scipy import signal
from scipy import special as ss

class HippoLegSCell(nn.Module):
    """Hippo class utilizing legs polynomial"""

    def __init__(self, N, gbt_alpha = 0.5, maxlength = 1024):
        super(HippoLegSCell, self).__init__()
        self.N = N
        self.gbt_alpha = gbt_alpha
        self.maxlength = maxlength
        
    def compute_A(self, n, k):
        '''
        Computes the values for the HiPPO A matrix row by column 
        using the piecewise equation on p. 31 eq. 29:
                (2n+1)^{1/2} (2k+ 1)^{1/2} if n > k  
        A_{nk} = n+1                       if n = k,
                 0                         if n < k
        , where n represents the row and k the columns. 
        
        Input:
            n (int):
                nth row of a square matrix of size N
            k (int):
                kth column of a square matrix of size N
        
        Returns:
            Values (float):
            Individual values for the elements in the A matrix. 
        '''
        if n > k:
            val = np.sqrt(2 * n + 1, dtype = np.float32) * np.sqrt(2 * k + 1, dtype = np.float32)
        if n == k:
            val = n + 1 
        if n < k:
            val = 0
        return val

    def compute_B(self, n):
        '''
        Computes the values for the HiPPO B matrix row by column 
        using the piecewise equation on p. 31 eq. 29:
        B_{n} = (2n+1)^{1/2}
        
        Input:
            n (int):
                nth column of a square matrix of size N.
            
        Returns:
            Values (float):
            Individual values for the elements in the B matrix.
            The next hidden state (aka coefficients representing the function, f(t))
        '''
        val = np.sqrt(2 * n + 1, dtype = np.float32)
        return val

    def get_A_and_B(self, N):
        '''
        Creates the HiPPO A and B matrix given the size N along a single axis of 
        a square matrix.
        
        Input: 
            N (int):
            Size N of a square matrix along a single axis.
        
        Returns: 
            A (np.ndarray)
                shape: (N,N)
                the HiPPO A matrix.
            B (np.ndarray)
                shape: (N,):
                The HiPPO B matrix.
        '''
        A = np.zeros((self.N, self.N), dtype = np.float32)
        B = np.zeros((self.N, 1), dtype = np.float32)

        for n in range(A.shape[0]):
            B[n][0] = self.compute_B(n = n)
            for k in range(A.shape[1]):
                A[n, k] = self.compute_A(n = n , k = k)

        return A  * -1, B
    
    def generalized_bilinear_transform(self, A, B, t, gbt_alpha):
        '''
        Performs the generalised bilinaer transform from p. 21 eq.13:
        c(t + ∆t) − ∆tαAc(t + ∆t) = (I + ∆t(1 − α)A)c(t) + ∆tBf(t)
        c(t + ∆t) = (I − ∆tαA)^{−1} (I + ∆t(1 − α)A)c(t) + ∆t(I − ∆tαA)^{−1}Bf(t).
        on the HiPPO matrix A and B, transforming them. 
        Input:
            A (np.ndarray):
                shape: (N, N)
                the HiPPO A matrix
            B (np.ndarray):
                shape: (N,)
                the HiPPO B matrix
            Timestep t = 1/input length at t (int):
        
        Output:
            GBTA (np.array):
                shape: (N, N)
                Transformed HiPPO A matrix.
            
            GBTB (np.array):
                shape: (N,)
                Transformed HiPPO B matrix.
        '''
        I = np.eye(A.shape[0], dtype = np.float32)
        delta_t = 1 / t
        EQ13_p1 = I - (delta_t * gbt_alpha * A)
        EQ13_p2 = I + (delta_t * (1 - gbt_alpha) * A)
        EQA = np.linalg.lstsq(EQ13_p1, EQ13_p2, rcond = None)[0]
        EQB =  np.linalg.lstsq(EQ13_p1, (delta_t * B), rcond = None)[0]         
        return EQA, EQB
    
    def get_stacked_GBT(self):
        A, B = self.get_A_and_B(self.N)
        GBTA_stacked = np.empty((self.maxlength, self.N, self.N), dtype=np.float32)
        GBTB_stacked = np.empty((self.maxlength, self.N, 1), dtype=np.float32)
        
        for t in range(1, self.maxlength + 1):
            GBTA, GBTB = self.generalized_bilinear_transform(A = A, B = B, t = t, gbt_alpha = self.gbt_alpha)
            GBTA_stacked[t-1] = GBTA
            GBTB_stacked[t-1] = GBTB
            
        return GBTA_stacked, GBTB_stacked
    
    def discrete_hippo_operator(self, A, B, inputs, c_t =  None):
        '''
        Input:
            A (np.ndarray):
                shape: (N, N)
                the discretized A matrix
            B (np.ndarray):
                shape: (N, 1)
                the discretized B matrix
            c_t (np.ndarray):
                shape: (batch size, input length, N)
                the initial hidden state
            inputs (torch.tensor):
                shape: (batch size, maxlength)
                the input sequence
        Returns:
            The next hidden state (aka coefficients representing the function, f(t))
        '''
        batchsize = inputs.shape[0]
        L = inputs.shape[1]

        # Change input shape from (batch size, max length)
        # to (max length, batch size, max length)
        # note that max length can also be regarded as the length of the signal
        inputs = torch.tensor(inputs)
        inputs = inputs.T.unsqueeze(-1).unsqueeze(-1)
        f_t = inputs
        #ft sould be batchsize value at time step t
        print('f_t shape: ', f_t.shape)

        if c_t is None:
            c_t = np.zeros((batchsize, 1, self.N), dtype = np.float32)
            
        c_t = F.linear(torch.tensor(c_t).float(), torch.tensor(A[t]).float()) + np.squeeze(B[t], -1) * f_t.numpy()

        return c_t
    
    def reconstruct(self, c, B):
        vals = np.linspace(0.0, 1.0, self.maxlength)
        # If clause for supporting use of batched raw signal 
        # with batch information of shape [batchsize, maxlength] 
        if len(c.shape) == 4:
            # c shape from: [maxlength, batchsize, 1, N_coeffs]
            # 1st move to: [batchsize, maxlength, 1, N_coeffs]
            # 2nd move to: [batchsize, maxlength, N_coeffs, 1]
            c = np.moveaxis(c, 0, 1)
            c = np.moveaxis(c, 2, 3)
        
        eval_mat = (B * np.float32(ss.eval_legendre(np.expand_dims(np.arange(self.N, dtype = np.float32), -1), 2 * vals - 1))).T
        recon = eval_mat @ np.float32(c)
        return recon
    
    def forward(self, inputs):
        # 1.Compute B, GBTA and GBTA matrices
        # B is needed in 3. for the reconstruction
        # GBTA and GBTA is needed for coefficents c
        _, B = self.get_A_and_B(N = self.N)
        GBTA, GBTB = self.get_stacked_GBT()
        # 2.Compute coefficents c
        c = self.discrete_hippo_operator(A = GBTA, B = GBTB, inputs = inputs, c_t = None)
        # 3. Compute reconstruction r
        r =  self.reconstruct(c = c, B = B)
        return c, r