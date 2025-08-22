import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
import scipy.sparse.linalg as scilasp
import scipy.linalg as scila
import matplotlib.cm as cmap
from spin_models import spin_models
from probes_f import probes_f


def sparse_mat(N, coeffs, model):
    # creates the hamiltonian matrix in sparse matrix form
    h = spin_models(N, model)
    hterms, hcoeffs = h.hamiltonian(coeffs)

    sparseop = SparsePauliOp(hterms, np.array(hcoeffs, dtype=complex))
    hmat = sparseop.to_matrix(sparse=True)

    return hmat


def op_sparse(N, opp):
    PauliOp = []
    PauliSparse = []

    tensor_list = ['I'] * N
    for i in range(N):
        tensor_list[i] = opp
        paulis_str = "".join(tensor_list)
        PauliOp.append(paulis_str)
        tensor_list[i] = 'I'

        PauliSparse.append(SparsePauliOp(PauliOp[i]).to_matrix(sparse=True))

    return PauliSparse

def par_relax(initial_par, final_par, t, t0, tau):

    '''
    
    Aim: 
        come up with a relaxing dynamics for a given Hamiltonian Parameter and a certain dimension depending on given parameters

    Input: 
        min_par: minimun value you wish to start or end with
        max_par: maximum value you wish to start or end with
        t: time axis as a numpy array
        t0: sets when the quench occur
        tau: exponential factor (speed of decay)

    Output:
        NdArray of parameter evolving over time-axis with preferred direction
    
    '''

    if final_par > 0:

        if (initial_par-final_par) < 0:

            return np.heaviside(t0-t, t) * initial_par + np.heaviside(t-t0, 0) * ((final_par-initial_par) * np.exp(-((t-t0))/(tau)) +initial_par)
        
        else:

            return np.heaviside(t-t0, 0)* (-initial_par * np.exp(-(t-t0)/tau)+ initial_par) + np.heaviside(t0-t, 0) * initial_par
    else:

        return np.heaviside(t-t0, 0)* (final_par * np.exp(-(t-t0)/tau)+ initial_par) + np.heaviside(t0-t, 0) * initial_par



def fd_evolve(N, time_pars, h_pars, c1, probe_pars, site, TV = True, TD = True):

    """
    
    returns the desired operator response function A(t) for a perturbed time-dependent Hamiltonian H0 + h(t0)B. 

    Parameters:
    -----------
    N: int
        Number of sites 

    time_pars : Dictionary
        Time parameters for the evolution. Comes in format: time_pars = {'tax': tax, 'dt': dt, 'nt': nt, 't0': t0, 'tau': tau}
    
    h_pars : Dictionary
        Time-dependent Hamiltonian parameters. Comes in format: c1 = {'par1_name': par1_value, 'par2_name': par2_value, 'par3_name': par3_value}

    probe_pars : Dictionary
        Probe Parameters. Comes in format: probe_pars = {'t0': t0, 'sigma': sigma', 'mapping': mapping}

    Returns:
    --------

    response_array : NDArray
        The time-domain array of the expectation value of the desired operator after perturbing at tp.

    """

    PauliSparse = op_sparse(N, probe_pars['opp'])
    hl_max = c1['h_l']
    ct_l = par_relax(h_pars['h_l0'], c1['h_l'], time_pars['tax'], time_pars['t0'], time_pars['tau'])

    js = np.zeros(probe_pars['j_points']+1)
    js[0] = 5
    js[1:]=np.linspace(15,50,probe_pars['j_points'])#np.geomspace(time_pars['t0'], time_pars['tmax']-5, probe_pars['j_points'])
    H1 = sparse_mat(N=N, coeffs=c1, model=h_pars['model'])
    
    #evals, psi0_p = scilasp.eigsh(H1,1)

    psi0_p = np.zeros(2 ** N, dtype=complex)
    psi0_p[0] = 1

    tax = time_pars['tax']

    response_array = np.ones((N, len(js), time_pars['nt']), dtype= 'complex')
    hit_arr = np.zeros((len(js), time_pars['nt']), dtype = complex)
    psi = np.zeros((nt, 2**N), dtype = 'complex')
    psi[0,:] = psi0_p

     
  
    for j in range(len(js)):

        d_h = probes_f(probe_pars['sigma'], js[j], probe_pars['amp'], probe_pars['w'], tax)
        
        hit_arr[j,:] = d_h.probe(probe_pars['mapping'])
        hit_arr[0,:] = 0

        psi[0,:] = psi0_p
          
        for i in range(time_pars['nt']-1):

            if TD == True:
                c1['h_l'] = ct_l[i]
           
                H1 = sparse_mat(N=N, coeffs=c1, model=h_pars['model'])    
            
            for n in range(N):
                
                response_array[n, j, i+1] = np.conj(psi[i,:]) @ PauliSparse[n] @  psi[i,:]     

            psi[i+1,:] = scilasp.expm_multiply(-1.j * (H1 + hit_arr[j,i] * PauliSparse[site]) * time_pars['dt'], psi[i,:])

    if TV == True:
       
        np.savez(f'response_FD_TV_{hl_max}.npz', arr1 = response_array, arr2 = js, arr3 = hit_arr, arr4 = ct_l)
    
    else:
        
        np.savez(f'response_FD_FV_S{hl_max}_sigma{probe_pars['sigma']}.npz', arr1 = response_array, arr2 = js, arr3 = hit_arr, arr4 = ct_l)
    
    return response_array, js, hit_arr, ct_l

TV = True
N = 12
nt = 2000
tmax = 80
tax = np.linspace(0, tmax, nt)
dt = tax[1]-tax[0]
t0 = 7

tau = 35 
sigma = 0.2
amp = 0.01

w = 2.2
h_l0 = 0
h_l = -0.4
nj = 100
site = 5


mapping = 'GAUSSIAN'
time_pars = {'tax': tax, 'dt': dt, 'nt': nt, 't0': t0, 'tau': tau, 'tmax': tmax}
probe_pars = {'j_points': nj,'sigma': sigma, 'amp': amp, 'w': w, 'mapping': mapping, 'opp': 'Z', 'site': 5}
h_pars = {'model':'LF', 'h_l0': h_l0}
c1 = {'J':1, 'h_t':0.25, 'h_l':h_l}

if h_l < 0:
    TV = False

A, js, hit_arr,ct_l = fd_evolve(N, time_pars, h_pars, c1, probe_pars, site,TV)




def f_derivatives(response, j_points, N, tax, hit_arr, damping_power):
    
    """
    
    this function takes raw response functions in time and position spaces, takes their functional derivative to first order 
    and transforms them into frequency and momentum spaces:

    Parameters:
    -----------
    response: NDArray
        The time-domain array of the expectation value of the desired operator after perturbing at tp.

    j-points : NDArray
        time points where a perturbation were performed for the functional derivative approach
    
    N : int
       number of sites

    tax : NDArray
        time array
    
    hit_arr : NDArray
        the temporal part of the hit/perturbation used to compute responses.
    
    damping power: int
        damping parameter to multiply response with

    Returns:
    --------

    wax : NDArray
        frequency axis transformed from corresponding tax
    
    a_ft : NDArray
        fourier transformed signal/response after damping

    A_damped: NDArray
        time-domain response after damping
    
    corr_FD: NDArray
        desired signal/correlation function in frequency domain
    
    hofw: NDArray
        frequency support of the hit/perturbation
    """

    wax = np.fft.fftshift(np.fft.fftfreq(len(tax),d=tax[1]-tax[0]))*2*np.pi
    A_ft = np.zeros((N, len(j_points), len(wax)), dtype=complex)
    ratio = np.zeros((N, len(j_points),len(wax)),dtype=complex)
    hofw = np.zeros((len(j_points), len(wax)), dtype=complex)
    #response_avg = np.zeros((N, len(j_points),len(wax)),dtype=complex)
    A_damped = np.zeros((N, len(j_points), len(wax)), dtype=complex)

    # damper = np.zeros(len(j_points), dtype = complex)

    for j in range(len(j_points)):

        damper = np.exp(-(tax-j_points[j])/damping_power)

        for n in range(N):

            # response_avg[n,j,:] = response[n,j,:]-np.average(response[n,j,:], axis = 0)
            A_damped[n,j,:] = damper*(response[n,j,:]-response[n,0,:])
            A_ft[n,j,:] = np.fft.fftshift(np.fft.fft(A_damped[n,j,:], norm='ortho')) 


    
    for j in range(1,len(j_points)):
        
        hofw[j, :] = np.fft.fftshift(np.fft.fft(hit_arr[j, :], norm='ortho'))
        hofw[j, :] /= max(abs(hofw[j, :]))
  

        for n in range(N):

            for iw in range(len(wax)):

                if np.abs(hofw[j,iw])**2 > 1e-2:

                    ratio[n, j, iw] = (A_ft[n, j, iw])/hofw[j, iw]
                    
    # q = np.zeros(N)
    # for iq in range(N):
    #     q[iq] = (iq)*2*np.pi/N
    
    
    # for j in range(len(j_points)):
    #     for kindex in range(N):
    #         result = 0
    #         for n in range(N):
    #             result = result + ratio[j,:,n] * np.exp(-n*q[kindex]*1j)
    #         signal_k[j, kindex, :] = result
            
            
    return wax, A_ft, A_damped, ratio, hofw  

wax, a_FT, A_damped, corr_FD, hofw = f_derivatives(A, js, N, tax, hit_arr, 40)


if mapping == 'SELECTIVE':
    np.savez(f'correlations_S_FD_sigma{probe_pars['sigma']}.npz', arr1 = a_FT, arr2 = corr_FD, arr3 = wax, arr4 = hofw, arr5 = A_damped)
else:
    np.savez('correlations_FD.npz', arr1 = a_FT, arr2 = corr_FD, arr3 = wax, arr4 = hofw, arr5 = A_damped)