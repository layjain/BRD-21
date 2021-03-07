"""
Numerically Solve the model to estimate parameters
"""

class CoronaVIRES_1(object):
    """
    SERV model # 1 (_Continuous_Consant_Vaccination ):
        Discrete approximations to Differential Equations which donot use past states of variables
        Vaccination assumed at a constant rate rho, through time
    """
    def __init__(self,N):
        self.N = N

        self.S = [] 
        self.V1, self.V2 = [],[] 
        self.Es, self.E1, self.E2 = [],[],[] 
        self.I1, self.I2, self.Is = [],[],[]
        self.R1, self.Rs = [],[]
        self.I, self.E = [], []
        self.D = []

    def run_predict(self, T, alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0):
        """
        Predict till t time stepsß
        """
        N = self.N
        # Initial conditions
        self.S = [S0] 
        self.V1, self.V2 = [0],[0] 
        self.Es, self.E1, self.E2 = [Es0],[0],[0] 
        self.I1, self.I2, self.Is = [0],[0],[Is0]
        self.R1, self.Rs = [0],[0]
        self.I, self.E = [Is0+0+0], [Es0+0+0]
        self.D = [0] # Translate to 0
        #loop using the DEs
        for t in range(T+1):
            S = self.S[-1]
            V1, V2 = self.V1[-1], self.V2[-1]
            Es, E1, E2 = self.Es[-1], self.E1[-1], self.E2[-1]
            I1, I2, Is = self.I1[-1], self.I2[-1], self.Is[-1]
            R1, Rs = self.R1[-1], self.Rs[-1]
            I, E = self.I[-1], self.E[-1]
            D = self.D[-1]

            dS = alpha*Rs - S*I*beta/N - S*chi*E/N - rho*S
            dV1 = rho*S + rho*Rs - V1*beta*I/N - V1*chi*E/N - phi*V1
            dV2 = phi*V1 + phi2*R1 + (1-del2)*I2 - V2*beta*I/N - V2*chi*E/N
            dEs = S*I*beta/N + S*chi*E/N - theta*Es
            dE1 = V1*beta*I/N + V1*chi*E/N - theta*E1
            dE2 = V2*beta*I/N + V2*chi*E/N - theta*E2
            dI1 = theta*E1 - I1*del1 - (1-del1)*I1
            dI2 = theta*E2 - I2*del2 - (1-del2)*I2
            dIs = theta * Es - (1-dels)*Is - Is*dels
            dD = del1*I1+del2*I2+dels*Is
            dR1 = (1-del1)*I1 - phi2*R1
            dRs = (1-dels)*Is - rho*Rs - alpha*Rs
            dE = dE1 + dE2 + dEs
            dI = dI1 + dI2 + dIs

            self.S.append(S+dS)
            self.V1.append(V1+dV1)
            self.V2.append(V2+dV2)

            self.Es.append(Es+dEs)
            self.E1.append(E1+dE1)
            self.E2.append(E2+dE2)

            self.I1.append(I1+dI1)
            self.I2.append(I2+dI2)
            self.Is.append(Is+dIs)

            self.D.append(D+dD)
            self.R1.append(R1+dR1)
            self.Rs.append(Rs+dRs)

            self.E.append(E+dE)
            self.I.append(I+dI)
            
    def predict_Deaths(self, T, alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0):
        self.run_predict(T, alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0)
        return self.D[T]
    
    def predict_Positive(self, T, alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0):
        self.run_predict(T, alpha, beta, del1, del2, chi, dels, rho, phi, phi2, theta, S0, Es0, Is0)
        return self.I[T]
        #TODO: I+E or I

    def fit_model(self, Deaths_observed, Infected_Observed, plot=False, plot_title="CoronaVIRES1", weights=None):
        pass