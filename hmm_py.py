import numpy as np

class HMM:
    def __init__(self, N, M):
        self.N = N                # hidden states count
        self.M = M                # observable symbols count

        # initialize probabilities
        self.A = np.ones((N, N)) / N           # state transitions
        self.B = np.ones((N, M)) / M           # emission probabilities
        self.PI = np.ones(N) / N               # initial state prob

    # Forward Algorithm --------------------------------------------------
    def forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.N))

        # initialization
        alpha[0] = self.PI * self.B[:, obs[0]]

        # recursion
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, obs[t]]
        return alpha, np.sum(alpha[-1])

    # Backward Algorithm -------------------------------------------------
    def backward(self, obs):
        T = len(obs)
        beta = np.zeros((T, self.N))
        beta[-1] = 1  # initialization

        # recursion
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i] * self.B[:, obs[t+1]] * beta[t+1])
        return beta, np.sum(self.PI * self.B[:, obs[0]] * beta[0])

    # Viterbi (Most Likely Hidden Path) ---------------------------------
    def viterbi(self, obs):
        T = len(obs)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)

        delta[0] = self.PI * self.B[:, obs[0]]

        # recursion
        for t in range(1, T):
            for j in range(self.N):
                temp = delta[t-1] * self.A[:, j]
                psi[t, j] = np.argmax(temp)
                delta[t, j] = np.max(temp) * self.B[j, obs[t]]

        # backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

    # Baum-Welch Training -----------------------------------------------
    def train(self, obs, iterations=20):
        T = len(obs)
        for _ in range(iterations):
            alpha, prob = self.forward(obs)
            beta, _ = self.backward(obs)

            # gamma & xi expectations
            gamma = (alpha * beta) / np.sum(alpha[-1])
            xi = np.zeros((T-1, self.N, self.N))

            for t in range(T-1):
                denom = np.sum(alpha[t] * self.A * self.B[:, obs[t+1]] * beta[t+1])
                for i in range(self.N):
                    xi[t, i] = alpha[t, i] * self.A[i] * self.B[:, obs[t+1]] * beta[t+1] / denom

            # Re-estimation
            self.PI = gamma[0]
            self.A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]

            for j in range(self.M):
                mask = (np.array(obs) == j)
                self.B[:, j] = np.sum(gamma[mask], axis=0) / np.sum(gamma, axis=0)

        return prob
obs = [0,1,0,2,1,0]   # observed events (encoded as integers)
hmm = HMM(N=3, M=3)    # 3 hidden states, 3 possible observations

likelihood = hmm.train(obs, iterations=30)
print("Log-Likelihood:", likelihood)

path = hmm.viterbi(obs)
print("Most likely state path:", path)
