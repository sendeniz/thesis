class HippoLegsCell(nn.Module):
    '''
    Hippo class utilizing legs polynomial
    '''

    def __init__(self, N, gbt_alpha = 0.5, maxlength = 1024, reconst = False):
        super(HippoLegsCell, self).__init__()
        self.N = N
        self.gbt_alpha = gbt_alpha
        self.maxlength = maxlength
        A, self._B = self.get_A_and_B(N = self.N)
        GBTA, GBTB = self.get_stacked_GBT(A = A, B = self._B)
        self.A = nn.parameter.Parameter(torch.from_numpy(GBTA))
        self.B = nn.parameter.Parameter(torch.from_numpy(GBTB))
        self.reconst = reconst

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

        Output:
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

        Output:
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

        Output:
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
        EQA = np.linalg.solve(EQ13_p1, EQ13_p2)
        EQB =  np.linalg.solve(EQ13_p1, (delta_t * B))
        return EQA, EQB

    def get_stacked_GBT(self, A, B):
        GBTA_stacked = np.empty((self.maxlength, self.N, self.N), dtype=np.float32)
        GBTB_stacked = np.empty((self.maxlength, self.N, 1), dtype=np.float32)
        for t in range(1, self.maxlength + 1):
            GBTA, GBTB = self.generalized_bilinear_transform(A = A, B = B, t = t, gbt_alpha = self.gbt_alpha)
            GBTA_stacked[t-1] = GBTA
            GBTB_stacked[t-1] = GBTB
        return GBTA_stacked, GBTB_stacked


    def reconstruct(self, c, B):
        '''
        Input:
            c (np.ndarray): 2, 1, 32
                shape: (batchsize, 1, N_coeffs)
                coefficent matrix
            B (np.ndarray):
                shape: (N, 1)
                the discretized B matrix
        Returns:
            recon (np.ndarray):
                shape: (batchsize, maxlength, 1)
                Reconstruction matrix.
        '''
        with torch.no_grad():
            vals = np.linspace(0.0, 1.0, self.maxlength)
            # c shape from: [batchsize, 1, N_coeffs]
            # move to: [batchsize, N_coeffs, 1]
            c = torch.moveaxis(c, 1, 2).float()
            eval_mat = (self._B * np.float32(ss.eval_legendre(np.expand_dims(np.arange(self.N, dtype = np.float32), -1), 2 * vals - 1))).T
            # shape: B :-: (N, 1)
            # shape: eval_mat :-:  (maxlen, N)
            recon = (torch.tensor(eval_mat).to(device) @ c.to(device))
            # shape: recon :-: (batchsize, maxlength, 1
            return recon

    def forward(self, input, c_t = None, t = 0):
        '''
        Input:
            A (np.ndarray):
                shape: (N, N)
                the discretized A matrix
            B (np.ndarray):
                shape: (N, 1)
                the discretized B matrix
            c_t (np.ndarray): 2, 1, 32
                shape: (batch size, 1, N)
                the initial hidden state
            input (torch.tensor):
                shape: (batch size, 1 ,1)
                the input sequence
        Output:
            c (np.array):
                shape: (batch size, 1, N)
                coefficent matrix c.
        '''
        batchsize = input.shape[0]
        L = input.shape[1]
        if c_t is None:
            c_t = torch.zeros((batchsize, 1, self.N)).to(device)
        c_t = F.linear(c_t.float(), self.A[t]).float() + self.B[t].squeeze(-1) * input
        # shape: F.linear(torch.tensor(c_t).float(), torch.tensor(A[t])) :-: (batchsize, 1, N)
        # shape: (np.squeeze(B[t], -1) * f_t.numpy()).shape) :-: (batchsize, 1, N)
        # shape: A[t] :-: (N, N)
        # shape: torch.squeeze(B[t], -1) :-: (N, )
        # shape: f_t :-: (batchsize, 1, 1)
        # shape: c_t :-: (batchsize, 1, N)
        if self.reconst:
            # 3. Compute reconstruction r
            r =  self.reconstruct(c = c_t, B = self._B)
        else:
            r = 0
        return c_t, r
