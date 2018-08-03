import numpy as np
import scipy, scipy.integrate
import pyscf, pyscf.gto, pyscf.scf, pyscf.ao2mo, pyscf.mp

def get_emp2_t(eigs, nocc, eri, t):
    """
    Laplace point for MP2
    """
    gpos = get_g0mat_t(eigs, nocc, t)
    gneg = get_g0mat_t(eigs, nocc, -t)

    emp2J = 2*np.einsum("aA,Ii,bB,Jj,aibj,AIBJ",gpos,gneg,gpos,gneg,eri,np.conj(eri))
    emp2K = -np.einsum("aA,Ij,bB,Ji,aibj,AIBJ",gpos,gneg,gpos,gneg,eri,np.conj(eri))
    return emp2J+emp2K

def lt7_int(fn):
    """
    7 pt Laplace transform quadrature (from Narbe)
    """
    times=[-0.0089206000,-0.0611884000,-0.2313584000,-0.7165678000,-1.9685146000,-4.9561668000,-11.6625886000]
    wts=[0.0243048000,0.0915096000,0.2796534000,0.7618910000,1.8956444000,4.3955808000,9.6441228000]
    return sum(w*fn(t) for w,t in zip(wts, times))

def get_g0mat_t(eigs, nocc, t):
    """
    OK time-domain Matsubara GF
    eq. A.4 in edoc.ub.uni-muenchen.de/18937/1/Wolf_Fabian_A.pdf
    """
    nmo = len(eigs)
    dtype = type(t)
    g0mat_t = np.zeros([nmo, nmo], dtype=dtype)
    if t > 0:
        for a in range(nocc, nmo):
            g0mat_t[a,a] = -np.exp(-eigs[a]*t)
    else:
        for i in range(nocc):
            g0mat_t[i,i] = np.exp(-eigs[i]*t)
    return g0mat_t

def get_ft_g0mat_t(eigs, beta, mu, t):
    """
    finite temperature time-domain Matsubara GF
    e.g Eq. 23.30, Fetter-Walecka
    """
    nmo = len(eigs)
    dtype = type(t)
    g0mat_t = np.zeros([nmo, nmo], dtype=dtype)

    if t > 0:
        for p in range(nmo):
            g0mat_t[p,p] = -np.exp(-(eigs[p]-mu)*t) * (1 - fermi_dirac(eigs[p], beta, mu))
    else:
        for p in range(nmo):
            g0mat_t[p,p] = np.exp(-(eigs[p]-mu)*t) * (fermi_dirac(eigs[p], beta, mu))
    return g0mat_t

def fermi_dirac(eig, beta, mu):
    if beta*(eig-mu) > 20:
        return 0.
    elif beta*(eig-mu) < -20:
        return 1.
    return 1./(np.exp(beta * (eig - mu)) + 1.)

def get_sigma_t(gpos, gneg, eri):
    sigmaJ = 2*np.einsum("aAt,Iit,bBt,aibj,AIBJ->jJt",gpos,gneg,gpos,eri,np.conj(eri))
    sigmaK = -np.einsum("aAt,Ijt,bBt,aibj,AIBJ->iJt",gpos,gneg,gpos,eri,np.conj(eri))
    return sigmaJ + sigmaK

inv = np.linalg.inv

def get_ft_sigma_w_scf(eigs, nocc, mu, eri, beta, ndiv, niter=50, tol=1.e-8):
    """
    self-consistent GF2
    """
    nt = 2**ndiv+1
    times = np.linspace(-beta, beta, nt)
    n = len(eigs)

    g0_t = np.zeros([n,n,len(times)])
    sigma_t = np.zeros_like(g0_t)
    
    for it, t in enumerate(times):
        g0_t[:,:,it] = get_ft_g0mat_t(eigs, beta, mu, t)

    g0_w = np.fft.fft(g0_t, norm="ortho") # only odd Matsubara frequencies should be non-zero
    gnew_w = np.zeros_like(g0_w)
    sigma_w = np.zeros_like(g0_w)

    gold_w = g0_w

    for iter in range(niter):
        inv = np.linalg.inv

        gpos_t = np.fft.ifft(gold_w, norm="ortho")
        gneg_t = np.flip(gpos_t, -1)
        sigma_t = get_sigma_t(gpos_t, gneg_t, eri)
        sigma_w = np.fft.fft(sigma_t, norm="ortho")


        for iw in range(nt):
            gnew_w[:,:,iw] = inv(inv(g0_w[:,:,iw]) + mu*np.eye(n) - sigma_w[:,:,iw])

        energy = (1./nt)*beta*np.einsum("ijw,jiw", gold_w, sigma_w).real # use gold so iter=0 is MP2
        print "Energy", energy
        nelec = 2*np.trace(gpos_t[:,:,0])
        print "nelec", nelec
        if np.linalg.norm(gold_w - gnew_w) < tol:
            break

        gold_w = gnew_w.copy()        

    return energy, nelec

def test():
    mol = pyscf.gto.M(
        verbose = 0,
        atom = 'H 0 0 0; H 0 0 1.2',
        basis = 'sto-3g',
        spin = 0)

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    mp = pyscf.mp.RMP2(mf)
    mp.kernel()
    # Conventional MP2
    print "MP2 energy", mp.emp2
    
    mo_energy = mf.mo_energy
    nmo = len(mo_energy)
    nocc = int(sum(mf.mo_occ)/2)

    eri = pyscf.ao2mo.kernel(mol, mf.mo_coeff, compact=False)
    eri = np.reshape(eri, [nmo]*4)

    # Laplace transform MP2
    def _emp2t(t):
        return get_emp2_t(mo_energy, nocc, eri, t)
    print "7 pt integration", lt7_int(_emp2t)

    # GF2
    tmax = 10; ndiv = 10
    #niter = 20
    mu = 0.
    print mo_energy

    # outerloop deltan
    def deltan(mu):
        # G(-ibeta) = G(0-)
        print "Mu value", mu
        egf2, ngf2 = get_ft_sigma_w_scf(mo_energy, nocc, mu, eri, tmax, ndiv)
        dn = ngf2 - 2*nocc
        return dn

    mu = scipy.optimize.brentq(deltan, -1, 1.)

    
    print "GF2 energy", egf2
