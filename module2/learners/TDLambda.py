

class TDLambda:

    def __init__(self, state_size, alpha=0.1, lam=0.0):
        self.alpha = alpha
        self.lam = lam
        self.V = state_size

    def update(self):
        pass