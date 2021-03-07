"""
Numerically Solve the model to estimate parameters
"""

class SEIR_Baseline(object):
    """
    SEIR model (Baseline ):
        Discrete approximations to Differential Equations which donot use past states of variables
        No Vaccination
    """
    def __init__(self,N):
        self.N = N

        self.S = [] 
        # self.V1, self.V2 = [],[] 
        self.Es = []
        self.Is = []
        self.Rs = []
        self.I, self.E = [], []
        self.D = []

    def run_predict(self, T, alpha, beta, chi, dels, rho, theta, S0, Es0, Is0):
        """
        Predict till t time stepsß
        """
        N = self.N
        # Initial conditions
        self.S = [S0] 
        # self.V1, self.V2 = [0],[0] 
        self.Es = [Es0]
        self.Is = [Is0]
        self.Rs = [0]
        self.I, self.E = [Is0+0+0], [Es0+0+0]
        self.D = [0] # Translate to 0
        #loop using the DEs
        for _ in range(T+1):
            S = self.S[-1]
            # V1, V2 = self.V1[-1], self.V2[-1]
            Es = self.Es[-1]
            Is = self.Is[-1]
            Rs = self.Rs[-1]
            I, E = self.I[-1], self.E[-1]
            D = self.D[-1]

            dS = alpha*Rs - S*I*beta/N - S*chi*E/N - rho*S
            # dV1 = rho*S + rho*Rs - V1*beta*I/N - V1*chi*E/N - phi*V1
            # dV2 = phi*V1 + phi2*R1 + (1-del2)*I2 - V2*beta*I/N - V2*chi*E/N
            dEs = S*I*beta/N + S*chi*E/N - theta*Es
            # dE1 = V1*beta*I/N + V1*chi*E/N - theta*E1
            # dE2 = V2*beta*I/N + V2*chi*E/N - theta*E2
            # dI1 = theta*E1 - I1*del1 - (1-del1)*I1
            # dI2 = theta*E2 - I2*del2 - (1-del2)*I2
            dIs = theta * Es - (1-dels)*Is - Is*dels
            dD = dels*Is
            # dR1 = (1-del1)*I1 - phi2*R1
            dRs = (1-dels)*Is - rho*Rs - alpha*Rs
            dE = dEs
            dI = dIs

            self.S.append(S+dS)
            # self.V1.append(V1+dV1)
            # self.V2.append(V2+dV2)

            self.Es.append(Es+dEs)
            # self.E1.append(E1+dE1)
            # self.E2.append(E2+dE2)

            # self.I1.append(I1+dI1)
            # self.I2.append(I2+dI2)
            self.Is.append(Is+dIs)

            self.D.append(D+dD)
            # self.R1.append(R1+dR1)
            self.Rs.append(Rs+dRs)

            self.E.append(E+dE)
            self.I.append(I+dI)
            
    def predict_Deaths(self, T, alpha, beta, chi, dels, rho, theta, S0, Es0, Is0):
        self.run_predict(T, alpha, beta, chi, dels, rho, theta, S0, Es0, Is0)
        return self.D[T]
    
    def predict_Positive(self, T, alpha, beta, chi, dels, rho, theta, S0, Es0, Is0):
        self.run_predict(T, alpha, beta, chi, dels, rho, theta, S0, Es0, Is0)
        return self.I[T]
        #TODO: I+E or I
