import numpy as np

paulis = {}
paulis['X'] = np.array([[0, 1], [1, 0]], dtype=complex)
paulis['Y'] = np.array([[0, -1.j], [1.j, 0]], dtype=complex)
paulis['Z'] = np.array([[1, 0], [0, -1]], dtype=complex)
paulis['I'] = np.array([[1, 0], [0, 1]], dtype=complex)
paulis['SP'] = np.array([[0, 1], [0, 0]], dtype=complex)
paulis['SM'] = np.array([[0, 0], [1, 0]], dtype=complex)
paulis['N'] = np.array([[1, 0], [0, 0]], dtype=complex)

class spin_models:

    def __init__(self, N, model):

        self.N = N
        self.model = model

    def hamiltonian(self, coeff):

        match self.model:

            case "LR":

                pauliops = []
                coeffs = []

            
                for isite in range(self.N):
                    # BX
                    oplist = ['I'] * self.N
                    oplist[isite] = 'X'

                    pauliops.append("".join(oplist))
                    coeffs.append(-coeff["h_t"])

        
                    for jsite in range(isite + 1, self.N):

                        dist = abs(isite - jsite)

                        if abs(isite - jsite) > self.N / 2:
                            dist = (self.N - abs(isite - jsite))

                        # ZZ
                        oplist = ['I'] * self.N
                        oplist[isite] = 'Z'
                        oplist[jsite] = 'Z'
                        pauliops.append("".join(oplist))
                        coeffs.append(-coeff['J'] / (dist ** coeff['a']))

                return pauliops, coeffs

            case "NN":

                pauliops = []
                coeffs = []

                for isite in range(self.N):
                    jsite = (isite + 1) % self.N

                    # spin_x

                    oplist = ['I'] * self.N
                    oplist[isite] = 'X'
                    oplist[jsite] = 'X'

                    pauliops.append("".join(oplist))
                    coeffs.append(-coeff['J'])

                    # spin_y

                    oplist = ['I'] * self.N
                    oplist[isite] = 'Y'
                    oplist[jsite] = 'Y'

                    pauliops.append("".join(oplist))
                    coeffs.append(-coeff['J'])

                    # spin_z

                    oplist = ['I'] * self.N
                    oplist[isite] = 'Z'
                    oplist[jsite] = 'Z'

                    pauliops.append("".join(oplist))
                    coeffs.append(-coeff['J'])

                return pauliops, coeffs

            case "LF":

                pauliops = []
                coeffs = []

                for isite in range(self.N):
                    # transverse_field
                    oplist = ['I'] * self.N
                    oplist[isite] = 'X'

                    pauliops.append("".join(oplist))
                    coeffs.append(-coeff['J'] * coeff['h_t'])

                    # longitudinal_field
                    oplist = ['I'] * self.N
                    oplist[isite] = 'Z'

                    pauliops.append("".join(oplist))
                    coeffs.append(-coeff['J'] * coeff['h_l'])

                    jsite = np.mod(isite + 1, self.N)

                    # spin_x
                    oplist = ['I'] * self.N
                    oplist[isite] = 'Z'
                    oplist[jsite] = 'Z'

                    pauliops.append("".join(oplist))
                    coeffs.append(-coeff['J'])

                return pauliops, coeffs

            case "H":

                pauliops = []
                coeffs = []
                a = coeff['t']
                b = coeff['t']  ## for the pi-shift (add a minus sign infront for it to work.)

                u_prime = coeff['U'] / 4
                mu_prime = -coeff['U'] / 4 + coeff['m'] / 2 ## accounting for the transformation from n to z.

                for isite in range(self.N):

                    ksite = (isite + int(self.N / 2)) % self.N

                    if ksite >= isite:
                        oplist = ['I'] * self.N
                        oplist[isite] = 'Z'
                        oplist[ksite] = 'Z'

                        pauliops.append("".join(oplist))
                        coeffs.append(u_prime)

                for isite in range(self.N - 1):

                    jsite = (isite + 1) % self.N

                    if jsite != self.N / 2:
                        oplist = ['I'] * self.N
                        oplist[isite] = 'X'
                        oplist[jsite] = 'X'

                        pauliops.append("".join(oplist))
                        coeffs.append(a)

                        oplist = ['I'] * self.N
                        oplist[isite] = 'Y'
                        oplist[jsite] = 'Y'

                        pauliops.append("".join(oplist))
                        coeffs.append(a)

                    if isite == int(self.N / 2) - 1:
                        oplist = ['I'] * self.N
                        zlist = ['Z'] * int(self.N / 2)
                        oplist[0:int(self.N / 2)] = zlist
                        oplist[0] = 'X'
                        oplist[isite] = 'X'

                        pauliops.append("".join(oplist))
                        coeffs.append(b)

                        oplist = ['I'] * self.N
                        zlist = ['Z'] * int(self.N / 2)
                        oplist[0:int(self.N / 2)] = zlist
                        oplist[0] = 'Y'
                        oplist[isite] = 'Y'

                        pauliops.append("".join(oplist))
                        coeffs.append(b)

                    if isite == int(self.N / 2):
                        oplist = ['I'] * self.N
                        zlist = ['Z'] * int(self.N / 2)
                        oplist[int(self.N / 2):] = zlist
                        oplist[isite] = 'X'
                        oplist[self.N - 1] = 'X'

                        pauliops.append("".join(oplist))
                        coeffs.append(b)

                        oplist = ['I'] * self.N
                        zlist = ['Z'] * int(self.N / 2)
                        oplist[int(self.N / 2):] = zlist
                        oplist[isite] = 'Y'
                        oplist[self.N - 1] = 'Y'

                        pauliops.append("".join(oplist))
                        coeffs.append(b)

                for i in range(self.N):
                    oplist = ['I'] * self.N
                    oplist[i] = 'Z'

                    pauliops.append("".join(oplist))
                    coeffs.append(-mu_prime)

                return pauliops, coeffs
        
            case "H-B":

                '''
                This spin Hamiltonian utilizes the binary mapping of bosons to qubits for the Holstein Model. For a system
                of N Fermions and N_B as an upper limit for the number of bosons in the system (the cut-off, one would need N * log_2(N_B+1) 
                qubits/spins to fully simulate the Hilbert space. 


                '''
                pauliops = []
                coeffs = []

                # Will need to modify parameters here:

                # Mapping to the free fermionic part:


                for isite in range(self.N):
                    jsite = (isite + 1) % self.N

                    # spin_x

                    oplist = ['I']*(self.N+coeff['N_B'])
                    oplist[isite] = 'X'
                    oplist[jsite] = 'X'

                    pauliops.append("".join(oplist))
                    coeffs.append(-coeff['t'])

                    # spin_y

                    oplist = ['I'] * (self.N*(coeff['N_B']+1))
                    oplist[isite] = 'Y'
                    oplist[jsite] = 'Y'

                    pauliops.append("".join(oplist))
                    coeffs.append(-coeff['t'])

                # Mapping to the free bosonic part:
                # Sum (Sum 1/2(1+Z_j))_i
                    
                for isite in range(self.N,  self.N*(coeff['N_B']+1)):

                    oplist[isite] = 'Z'

                    pauliops.append("".join(oplist))
                    coeffs.append(-coeff['w'])


                # Mapping to the interaction term:
                # Sum 1/2(1+Z_i) Sum (X_i)_j
                        
                for isite in range(self.N):

                    oplist = ['I'] *  (self.N*(coeff['N_B']+1))
                    oplist[isite] = 'Z'
                    oplist[isite + self.N] = 'X'

                    pauliops.append("".join(oplist))
                    coeffs.append(-coeff['g'])

                return pauliops, coeffs


                










