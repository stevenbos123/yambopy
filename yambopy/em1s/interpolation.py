#!/miniconda3/envs/yambopy/bin/python3.11

from yambopy.dbs.latticedb import *
from yambopy.dbs.em1sdb import *
from yambopy.em1s.em1s_rotate import *
from yambopy.lattice import point_matching
from itertools import product
import math
import bisect
import time
from netCDF4 import Dataset

#Re-implement Paleari's algorithm to fold the static screening of the primitive cell to the smaller supercell 1BZ'
# WARNING: Please use cartesian coordinates
from scipy.interpolate import griddata, NearestNDInterpolator
from sklearn.model_selection import KFold
from typing import Tuple, Literal
import numpy.typing as npt



import scipy.interpolate as interpolate



def lanczos_global_parallel(epsilon, psi, m):
        """
        Parallel implementation of Lanczos basis construction using MPI.

        Parameters
        ----------
        epsilon : np.ndarray
            Dielectric operator, shape (N_q, N_gvecs, N_gvecs).
        psi : np.ndarray
            Wavefunctions, shape (N_q, N_bands, N_gvecs).
        m : int
            Number of Lanczos steps (subspace dimension).

        Returns
        -------
        Q : np.ndarray
            Lanczos basis, shape (N_q, N_bands, N_gvecs, m).
        alpha : np.ndarray
            Diagonal elements of the tridiagonal matrix, shape (m,).
        beta : np.ndarray
            Off-diagonal elements of the tridiagonal matrix, shape (m-1,).
        """
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        N_q, N_gvecs, _ = epsilon.shape
        N_bands = psi.shape[1]

        # Divide q-points among processes
        q_start = rank * (N_q // size)
        q_end = (rank + 1) * (N_q // size) if rank != size - 1 else N_q
        epsilon_local = epsilon[q_start:q_end]
        psi_local = psi[q_start:q_end]

        # Initialize local Lanczos basis arrays
        local_Q = np.zeros((q_end - q_start, N_bands, N_gvecs, m), dtype=complex)
        alpha = np.zeros(m, dtype=float)
        beta = np.zeros(m-1, dtype=float)

        # Normalize initial wavefunction
        psi_global = psi_local.reshape(-1)
        psi_norm = np.linalg.norm(psi_global)
        Q_global = np.zeros((N_q * N_bands * N_gvecs, m), dtype=complex)
        if rank == 0:
            Q_global[:, 0] = psi_global / psi_norm

        # Broadcast the initial Lanczos vector to all processes
        comm.Bcast(Q_global[:, 0], root=0)

        # Perform Lanczos iterations
        for n in range(m):
            w_local = np.zeros((q_end - q_start, N_bands, N_gvecs), dtype=complex)

            for q in range(q_end - q_start):
                w_q = np.einsum('ij,bj->bi', epsilon_local[q], Q_global[q_start + q].reshape(N_bands, N_gvecs))
                w_local[q] = w_q

            # Gather `w` from all processes
            w_global = np.zeros_like(Q_global[:, n])
            comm.Allgather([w_local.flatten(), MPI.COMPLEX], [w_global, MPI.COMPLEX])

            # Compute alpha and update w
            if rank == 0:
                alpha[n] = np.real(np.vdot(Q_global[:, n], w_global))
            alpha = comm.bcast(alpha, root=0)

            w_global -= alpha[n] * Q_global[:, n]

            if n > 0:
                w_global -= beta[n-1] * Q_global[:, n-1]

            # Reorthogonalize
            if rank == 0:
                coeffs = np.dot(Q_global[:, :n+1].conj().T, w_global)
                w_global -= np.dot(Q_global[:, :n+1], coeffs)

            # Broadcast orthogonalized `w`
            comm.Bcast(w_global, root=0)

            # Normalize and compute beta
            beta_norm = np.linalg.norm(w_global)
            beta = comm.bcast(beta, root=0)
            if beta_norm > 1e-14 and n < m - 1:
                if rank == 0:
                    Q_global[:, n+1] = w_global / beta_norm
                    beta[n] = beta_norm

        # Gather final Q from all processes
        comm.Allgather([local_Q.flatten(), MPI.COMPLEX], [Q_global, MPI.COMPLEX])

        return Q_global.reshape(N_q, N_bands, N_gvecs, m), alpha, beta



def fourier_interpolate(original_data, original_freq, target_grid, method):
    """
    Interpolate data using Fourier transform method

    Parameters:
    -----------
    original_data : array_like
    The original data to be interpolated
    original_freq : array_like
    The frequency grid of the original data
    target_grid : array_like
    The target grid for interpolation

    Returns:
    --------
    interpolated_data : ndarray
    Interpolated data on the target grid
    """
    # Perform FFT on the original data
    fft_data = np.fft.fft(original_data)

    # Create interpolation function for FFT magnitude and phase
    magnitude_interp = interpolate.interp1d(
    original_freq, 
    np.abs(fft_data), 
    kind=method, 
    fill_value='extrapolate'
    )
    phase_interp = interpolate.interp1d(
    original_freq, 
    np.angle(fft_data), 
    kind=method, 
    fill_value='extrapolate'
    )

    # Interpolate magnitude and phase on the target frequency grid
    target_magnitude = magnitude_interp(np.fft.fftfreq(len(target_grid), d=(target_grid[1]-target_grid[0])))
    target_phase = phase_interp(np.fft.fftfreq(len(target_grid), d=(target_grid[1]-target_grid[0])))

    # Reconstruct complex FFT data
    target_fft = target_magnitude * np.exp(1j * target_phase)

    # Perform inverse FFT to get interpolated data
    interpolated_data = np.real(np.fft.ifft(target_fft))

    return interpolated_data

def direct_interpolate(original_data, original_grid, target_grid, method):
    """
    Directly interpolate data in the original spatial domain

    Parameters:
    -----------
    original_data : array_like
    The original data to be interpolated
    original_grid : array_like
    The original grid points
    target_grid : array_like
    The target grid for interpolation
    method : str, optional
    Interpolation method. Defaults to 'linear'.
    Options include 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'

    Returns:
    --------
    interpolated_data : ndarray
    Interpolated data on the target grid
    """
    # Create interpolation function
    interpolator = interpolate.interp1d(
    original_grid, 
    original_data, 
    kind=method, 
    fill_value='extrapolate'
    )

    # Interpolate data
    interpolated_data = interpolator(target_grid)

    return interpolated_data
    





class UCtoSCInterpolator():
    """
    Interpolator for obtaining SC screening matrix from UC screening matrix.
    Handles the transformation from X_{G,G'}(Q) in UC to SC coordinates.
    """
    def __init__(self, UcLatticePath, UcScreeningPath, ScLatticePath, ScScreeningPath, 
                 UcRotatedXOutPath, ScRotatedXOutPath, NewScRotatedXOutPath, 
                 UcLattDB ="ns.db1",UcEm1sDB = "ndb.em1s" ,ScLattDB = "ns.db1" ,ScEm1sDB = "ndb.em1s", 
                 ExpandUc=True, ExpandSc=False,
                 method: Literal['linear', 'cubic'] = 'linear',
                 n_fold_cv: int = 5):
        self.method = method
        self.n_fold_cv = n_fold_cv
        self.interpolator = None
        self.nearest_interpolator = None
        self.error_estimate = None
        self.relative_error = None
 
        #MAKE ACCESSIBLE TO SUBROUTINES SEVERAL INPUT VARIABLES
        self.ScScreeningPath = ScScreeningPath 
        self.NewScRotatedXOutPath = NewScRotatedXOutPath
        self.ScEm1sDB = ScEm1sDB
        
        
        #READ YAMBO DATABASES
        #read uc lattice points and screening database
        UcLattice = YamboLatticeDB.from_db_file(filename = UcLatticePath+UcLattDB, Expand=True)
        UcStatScreen = YamboStaticScreeningDB(save = UcLatticePath, em1s = UcScreeningPath )
        
        self.UcLattice = UcLattice
        self.UcStatScreen = UcStatScreen
        
        #read sc lattice points and screening database. Do not care if Sc computation is converged or correct. We just need the correct set of q points and g vectors
        ScLattice = YamboLatticeDB.from_db_file(filename = ScLatticePath+ScLattDB, Expand=True)
        ScStatScreen = YamboStaticScreeningDB(save = ScLatticePath, em1s = ScScreeningPath )

        self.ScLattice = ScLattice
        self.ScStatScreen = ScStatScreen
        

        # if screening databases are not already expanded from IBZ to the full 1BZ make the rotation
        print("Expand unit cell Q points")
        if ExpandUc == True :
            #expand the Uc static screening from IBZ to the full 1BZ by mean of a rotation
            UcFull1BZStaticScreening = YamboEm1sRotate(UcStatScreen, save_path = UcLatticePath , path_output_DBs = UcRotatedXOutPath)
        else :
            UcFull1BZStaticScreening = UcStatScreen
        
        print("Expand supercell q points")  
        if ExpandSc == True :
            #expand the Sc static screening from IBZ to the full 1BZ by mean of a rotation. Might give a problem with fixed dimensions to allocate via netcdf libraries
            ScFull1BZStaticScreening = YamboEm1sRotate(ScStatScreen, save_path = ScLatticePath , path_output_DBs = ScRotatedXOutPath)
        else : 
            ScFull1BZStaticScreening = ScStatScreen

        #EXTRACT NUMBER OF G, g, Q, q FROM DATABASES     
        # make available the G and Q arrays for the Uc epsilon 
        self.Gvectors = UcFull1BZStaticScreening.gvectors   # G vectors in cartesian coordinates 
        self.NGvectors = UcFull1BZStaticScreening.ngvectors
        if ExpandUc == True :   #if Uc calculation is with symmetries use the Q point list after rotation
            self.Qpts = UcFull1BZStaticScreening.qpoints   # in cartesian coordinates
        elif ExpandUc == False:  #if Uc calculation is without symmetries use the Q point list read from lattice database
            self.Qpts = UcFull1BZStaticScreening.car_qpoints
        self.NQpts = UcFull1BZStaticScreening.nqpoints
        self.UcX = UcFull1BZStaticScreening.X
        # At this point I have X_{G,G'}(Q) for the Uc with a database fragment for each Q

        # make available the g and q arrays for the Sc epsilon 
        self.gvectors = ScFull1BZStaticScreening.gvectors   # g vectors in cartesian coordinates
        self.Ngvectors = ScFull1BZStaticScreening.ngvectors
        if ExpandSc == True :   #if Sc calculation is with symmetries use the Q point list after rotation
            self.qpts = ScFull1BZStaticScreening.qpoints
        elif ExpandSc == False:   #if Sc calculation is without symmetries use the Q point list read from lattice database
            self.qpts = ScFull1BZStaticScreening.car_qpoints
        self.Nqpts = ScFull1BZStaticScreening.nqpoints
        self.ScX = ScFull1BZStaticScreening.X
        # At this point I have the number of g vectors and q points for the Sc


        
        #print Q and q in cartesian coordinates
        print("****Q points processed by YamboEm1sRotate (after rotation)*************")
        print("Number of Q points: ", len(self.Qpts))
        print("Q points (cart.)")
        print(self.Qpts)
        
        print("****q points read from YamboStaticScreenings***************************")
        print("Number of q points: ", len(self.qpts))
        print("q points (cart.)")
        print(self.qpts)
        
        #print G and g 
        print("****G vectors*****")
        print("number of G vectors : ",  len(self.Gvectors))
        print("G vectors (cart)")
        print(self.Gvectors)
        
        print("****g vectors*****")
        print("number of g vectors : ",  len(self.gvectors))
        print("g vectors (cart)")
        print(self.gvectors)
    
        
    def fit_uc_data(self) -> None:
        """
        Fit the interpolator to the UC screening data.
        
        Parameters:
        -----------
        uc_qpoints : ndarray
            UC Q-points in cartesian coordinates (nq, 3)
        uc_gvectors : ndarray
            UC G-vectors in cartesian coordinates (ng, 3)
        uc_X : ndarray
            UC Screening matrix X(q,G,G') with shape (nq, ng, ng)
        """

        # Prepare full coordinate grid from UC data
        self.points, self.values = self._prepare_uc_grid()
        print(f"Points shape: {self.points.shape}")
        print(f"Values shape: {self.values.shape}")
        
        # Create nearest neighbor interpolator for points outside convex hull
        self.nearest_interpolator = NearestNDInterpolator(self.points, self.values)
        
        # Estimate interpolation error
        self._estimate_error()
    
    def _prepare_uc_grid(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Prepare the full coordinate grid from UC data.
        """
        nq, ng = self.NQpts, self.NGvectors

        # Create meshgrid of all UC combinations
        q_idx, g1_idx, g2_idx = np.meshgrid(
            np.arange(nq),
            np.arange(ng),
            np.arange(ng),
            indexing='ij'       # double check this
        )

        # Get the cartesian coordinates
        q_coords = self.Qpts[q_idx]
        g1_coords = self.Gvectors[g1_idx]
        g2_coords = self.Gvectors[g2_idx]

        # Compute the 3D vector sum of q + g1 + g2
        points = (q_coords + g1_coords + g2_coords).reshape(-1, 3)

        values = self.UcX.flatten()

        return points, values
    
    def _prepare_sc_grid(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Prepare the full coordinate grid from SC data.
        """
        nq, ng = self.Nqpts, self.Ngvectors

        # Create meshgrid of all SC combinations
        q_idx, g1_idx, g2_idx = np.meshgrid(
            np.arange(nq),
            np.arange(ng),
            np.arange(ng),
            indexing='ij'
        )

        # Get the cartesian coordinates
        q_coords = self.qpts[q_idx]
        g1_coords = self.gvectors[g1_idx]
        g2_coords = self.gvectors[g2_idx]

        # Compute the 3D vector sum of q + g1 + g2
        points = (q_coords + g1_coords + g2_coords).reshape(-1, 3)


        return points
    
    def _estimate_error(self) -> None:
        """
        Estimate interpolation error using k-fold cross-validation.
        """
        kf = KFold(n_splits=self.n_fold_cv, shuffle=True, random_state=42)
        errors = []
        
        for train_idx, test_idx in kf.split(self.points):
            points_train = self.points[train_idx]
            values_train = self.values[train_idx]
            points_test = self.points[test_idx]
            values_test = self.values[test_idx]
            
            values_pred = griddata(
                points_train, values_train, points_test,
                method=self.method, fill_value=np.nan
            )
            
            nan_mask = np.isnan(values_pred)
            if nan_mask.any():
                nearest = NearestNDInterpolator(points_train, values_train)
                values_pred[nan_mask] = nearest(points_test[nan_mask])
            
            errors.extend(np.abs(values_pred - values_test))
        
        self.error_estimate = np.mean(errors)
        self.relative_error = self.error_estimate / np.mean(np.abs(self.values))
    
    def interpolate_sc(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Interpolate to get SC screening matrix.
        
        Parameters:
        -----------
        sc_qpoints : ndarray
            SC Q-points to interpolate at
        sc_gvectors : ndarray
            SC G-vectors to interpolate at
            
        Returns:
        --------
        sc_X : ndarray
            Interpolated SC screening matrix
        uncertainties : ndarray
            Estimated uncertainties for interpolated values
        """
        nq_sc = self.Nqpts
        ng_sc = self.Ngvectors
        sc_qpoints = self.qpts
        sc_gvectors = self.gvectors
        # Initialize output arrays
        sc_X = np.zeros((nq_sc, ng_sc, ng_sc), dtype=np.complex128)
        uncertainties = np.zeros((nq_sc, ng_sc, ng_sc))
        
        # Prepare full coordinate grid from SC data
        points_new = self._prepare_sc_grid()
        
        # Interpolate
        values = griddata(
            self.points, self.values, points_new,
            method=self.method, fill_value=np.nan
        )
        
        # Handle points outside convex hull
        nan_mask = np.isnan(values)
        if nan_mask.any():
            values[nan_mask] = self.nearest_interpolator(points_new[nan_mask])
        
        # Estimate uncertainties
        # uncertainties_flat = np.full_like(values, self.error_estimate, dtype=float)
        # uncertainties_flat[nan_mask] *= 2  # Double uncertainty for extrapolated points
        
        # Reshape results
        sc_X = values.reshape(nq_sc, ng_sc, ng_sc)
        # uncertainties = uncertainties_flat.reshape(nq_sc, ng_sc, ng_sc)
        
        return sc_X#, uncertainties

    def SaveNewXDB(self, InputX, Db2BeOverWritten ,OutputPath) :
            """
            Save the database, constructed from saveDBS in em1sdb
            """
            if os.path.isdir(OutputPath): shutil.rmtree(OutputPath)
            os.mkdir(OutputPath)

            # recast X in array of X[q] of the different q
    #       self.new_x = np.array([ self.X[q] for q in range(self.sqq) if self.in_sIBZ[q]==1 ])     
    #        NewX = np.array([ InputX[q] for q in range(self.Nqpts) if self.in_sIBZ[q]==1 ])

            # copy ndb.em1s
            shutil.copyfile("%s/%s"%(Db2BeOverWritten,self.ScEm1sDB),"%s/%s"%(OutputPath,self.ScEm1sDB))
            # copy em1s fragments, one per q point
            for Indexq in range(self.Nqpts):
                FragmentName = "%s_fragment_%d"%(self.ScEm1sDB,Indexq+1)
                shutil.copyfile("%s/%s"%(Db2BeOverWritten,FragmentName),"%s/%s"%(OutputPath,FragmentName))

            #overwrite new X in the copied databases
            for Indexq in range(self.Nqpts):
                FragmentName = "%s_fragment_%d"%(self.ScEm1sDB,Indexq+1)
                database = Dataset("%s/%s"%(OutputPath,FragmentName),'r+')
                database.variables['X_Q_%d'%(Indexq+1)][0,:,:,0] = InputX[Indexq].real
                database.variables['X_Q_%d'%(Indexq+1)][0,:,:,1] = InputX[Indexq].imag
                database.close()




    def gmod(self,g):
        """
        Calculate the modulus of the g-vector(s)
        """
        g = np.atleast_2d(g)  # Ensure g is at least 2D
        gbar = g * 2 * np.pi
        g_mod = np.einsum('ij,ij->i', gbar, np.abs(gbar))
        
        return np.sign(g_mod) * np.sqrt(np.abs(g_mod))
    
    def prepare_Q_grid(self):
        Qdist = np.abs(self.gmod(self.Qpts))
        Qsorted_argsort = np.argsort(Qdist)
        self.Qsorted = np.sort(Qdist)
        self.xQsorted = self.UcX[Qsorted_argsort,:,:]
        print(f"Shape of X UC after sorting: {self.xQsorted.shape}")


    def prepare_q_grid(self):
        qdist = np.abs(self.gmod(self.qpts))
        self.qsorted_argsort = np.argsort(qdist)
        self.qsorted = np.sort(qdist)
        

    def interpolate_q(self, method):
        self.prepare_Q_grid()
        self.prepare_q_grid()

        num_g = len(self.Gvectors)
        interpolated_data = np.zeros((len(self.qsorted), num_g, num_g), dtype=complex)
        
        for gi in range(num_g):
            for gj in range(gi, num_g):  # Only compute for gj >= gi to leverage symmetry
                if gi//10==1 or gj//10==1:
                    print(gi, gj)
                real_data, imag_data = np.real(self.xQsorted[:, gi, gj]), np.imag(self.xQsorted[:, gi, gj])
                fft_real_freq = np.fft.fftfreq(len(real_data), d=(self.Qsorted[1] - self.Qsorted[0]))

                interpolated_imag_fourier = fourier_interpolate(imag_data, fft_real_freq, self.qsorted, method=method)
                interpolated_real_fourier = fourier_interpolate(real_data, fft_real_freq, self.qsorted, method=method)

                interpolated_complex_fourier = interpolated_real_fourier + 1j * interpolated_imag_fourier
                
                # Store the result using symmetry
                interpolated_data[:, gi, gj] = interpolated_complex_fourier
                if gi != gj:
                    interpolated_data[:, gj, gi] = interpolated_complex_fourier
        
        inverse_qsort_indices = np.argsort(self.qsorted_argsort)
        self.ScX = interpolated_data[inverse_qsort_indices]




from yambopy.dbs.wfdb import *

class dielectricModel():
    def __init__(self, UcLatticePath, UcScreeningPath, 
                 UcLattDB ="ns.db1",UcEm1sDB = "ndb.em1s", 
                 ExpandUc=True) -> None:
        
        x_uc= YamboStaticScreeningDB(save=UcLatticePath, em1s=UcScreeningPath)
        x_uc_rotated = YamboEm1sRotate(x_uc)
        wf_uc= YamboWFDB(save=UcLatticePath)
        self.psi, self.vq, self.e = self.lanczos_basis(x_uc, wf_uc)
        self.epsilon = x_uc.X

        self.car_qpoints = x_uc.car_qpoints


    def lanczos_basis(self,x_uc, wf_uc):
        NGvecs = x_uc.ngvectors
        e = wf_uc.wf[:,:,:,:NGvecs]
        vq = x_uc.sqrt_V
        # Step 1: Compute conjugate and product
        e_conj = np.conj(e)
        product = e_conj * e  # Element-wise multiplication

        # Step 2: Apply dielectric function
        v_sqrt = vq  # Element-wise square root
        psi_intermediate = v_sqrt[:, np.newaxis, np.newaxis, :] * product
        psi = psi_intermediate.reshape(x_uc.nqpoints, x_uc.nbands, x_uc.ngvectors)

        return psi, vq, e
        # Step 3: Apply transformation matrix C
        # Assuming C is defined and has compatible dimensions for multiplication
        psi_final = psi_intermediate


        # Compute (eps - I)|psi> for each q-point
        result = eps - np.eye(x_uc.ngvectors)[np.newaxis, :, :]
        final_result = np.einsum('qij,qbaj->qbai', result, psi_final).reshape(x_uc.nqpoints, x_uc.nbands, x_uc.ngvectors)


    def lanczos_global(self, m):
        """
        Construct a global orthonormal Lanczos basis across q-points and bands.

        Parameters
        ----------
        epsilon : np.ndarray
            Dielectric operator, shape (N_q, N_gvecs, N_gvecs).
        psi : np.ndarray
            Wavefunctions, shape (N_q, N_bands, N_gvecs).
        m : int
            Number of Lanczos steps (subspace dimension).

        Returns
        -------
        Q : np.ndarray
            Orthonormal global Lanczos basis, shape (N_q, N_bands, N_gvecs, m).
        alpha : np.ndarray
            Diagonal elements of the tridiagonal matrix, shape (m,).
        beta : np.ndarray
            Off-diagonal elements of the tridiagonal matrix, shape (m-1,).
        """
        epsilon, psi = self.epsilon, self.psi
        N_q, N_gvecs, _ = epsilon.shape
        _, N_bands, _ = psi.shape

        # Flatten psi for global processing
        psi_global = psi.reshape(-1)  # Flatten to 1D array

        # Initialize arrays for Lanczos iteration
        total_dim = N_q * N_bands * N_gvecs
        Q = np.zeros((total_dim, m), dtype=complex)
        alpha = np.zeros(m, dtype=float)
        beta = np.zeros(m-1, dtype=float)

        # Normalize initial wavefunction
        psi_norm = np.linalg.norm(psi_global)
        Q[:, 0] = psi_global / psi_norm

        for n in range(m):
            # Apply block-diagonal epsilon to the current vector
            print("iteration: ",m)
            w = np.zeros_like(Q[:, n], dtype=complex)

            # Process all q-points in parallel
            Q_flat = Q[:, n].reshape(N_q, N_bands, N_gvecs)  # Reshape for q-point processing
            w_flat = np.einsum('qij,qbj->qbi', epsilon, Q_flat)  # Batched matrix-vector multiplication
            w = w_flat.reshape(-1)  # Flatten back to global space

            # Compute alpha (diagonal element)
            alpha[n] = np.real(np.vdot(Q[:, n], w))
            w -= alpha[n] * Q[:, n]

            # Remove component along the previous Lanczos vector
            if n > 0:
                w -= beta[n-1] * Q[:, n-1]

            # Reorthogonalize with all previous vectors
            coeffs = np.dot(Q[:, :n+1].conj().T, w)  # Projection onto all previous vectors
            w -= np.dot(Q[:, :n+1], coeffs)  # Subtract contributions

            # Normalize w and compute beta
            beta_norm = np.linalg.norm(w)
            if beta_norm > 1e-14:
                if n < m - 1:
                    Q[:, n+1] = w / beta_norm
                    beta[n] = beta_norm
            else:
                break

        self.Q = Q.reshape(N_q, N_bands, N_gvecs, m)
        self.alpha = alpha
        self.beta = beta


        # return self.Q, self.alpha, self.beta


    def _tridiagonal_matrix(self, alpha, beta):
        """
        Construct the tridiagonal matrix from Lanczos coefficients.

        Parameters
        ----------
        alpha : np.ndarray
            Diagonal elements, shape (m,).
        beta : np.ndarray
            Off-diagonal elements, shape (m-1,).

        Returns
        -------
        T : np.ndarray
            Tridiagonal matrix, shape (m, m).
        """
        m = len(alpha)
        T = np.zeros((m, m), dtype=alpha.dtype)

        # Fill diagonal
        np.fill_diagonal(T, alpha)

        # Fill off-diagonal
        np.fill_diagonal(T[:-1, 1:], beta)
        np.fill_diagonal(T[1:, :-1], beta)

        return T


    def build_tridiagonal_matrix(self):
        # Example usage (assuming alpha and beta are computed from Lanczos):
        # T_q_band is the tridiagonal representation of epsilon - I for each q-point and band.

        N_q, N_bands, _, m = self.Q.shape
        T_q_band = np.zeros((N_q, N_bands, m, m), dtype=self.alpha.dtype)

        for q in range(N_q):
            for b in range(N_bands):
                T_q_band[q, b] = self._tridiagonal_matrix(self.alpha, self.beta)

        self.T = T_q_band

    def reconstruct_epsilon(self):
        """
        Reconstruct the original dielectric matrix from the Lanczos basis and tridiagonal representation.

        Parameters
        ----------
        Q : np.ndarray
            Lanczos basis, shape (N_q, N_bands, N_gvecs, m).
        T : np.ndarray
            Tridiagonal representation of epsilon - I in the Lanczos basis, shape (N_q, N_bands, m, m).

        Returns
        -------
        epsilon_reconstructed : np.ndarray
            Reconstructed dielectric matrix, shape (N_q, N_gvecs, N_gvecs).
        """

        N_q, N_bands, N_gvecs, _ = self.Q.shape
        epsilon_reconstructed = np.zeros((N_q, N_gvecs, N_gvecs), dtype=complex)

        for q in range(N_q):
            for b in range(N_bands):
                # Extract Q and T for this q-point and band
                Q_qb = self.Q[q, b]  # Shape: (N_gvecs, m)
                T_qb = self.T[q, b]  # Shape: (m, m)

                # Project T back to the full G-space: epsilon_reconstructed = Q T Q^dagger + I
                epsilon_reconstructed[q] += Q_qb @ T_qb @ Q_qb.conjugate().T

        # Add the identity matrix back to reconstruct epsilon
        epsilon_reconstructed += np.eye(N_gvecs, dtype=complex)
        self.eps_prime = epsilon_reconstructed
        # return epsilon_reconstructed

    from yambopy.plot.plotting import add_fig_kwargs

    @add_fig_kwargs
    def compare_eps_prime(self,save_path='./epsilon_comparison.png', **kwargs):
        fig, ax = plt.subplots(1,1, **kwargs)


        indices = np.arange(len(self.car_qpoints))

        # Vectorized computation of x
        modelX = self.vq[:, :, np.newaxis] * self.eps_prime/ self.vq[:, np.newaxis, :] /np.pi
        trueX = self.vq[:, :, np.newaxis] * self.epsilon / self.vq[:, np.newaxis, :] /np.pi

        x = np.linalg.norm(self.car_qpoints[indices], axis=1)

        # Vectorized extraction of y
        y_true = trueX[:, 0, 0][indices]
        y_model = modelX[:, 0, 0][indices]
        # Sort using numpy
        sort_indices = np.argsort(x)
        x = x[sort_indices]
        y_true = y_true[sort_indices]
        y_model = y_model[sort_indices]
  
        ax.plot(x,(1+y_model).real, label='eps_prime')
        ax.plot(x,(1+y_true).real, label='eps')

        ax.set_xlabel('$|q|$')
        ax.set_ylabel('$\epsilon^{-1}_{%d%d}(\omega=0)$'%(0,0))
        ax.legend()
        fig.savefig(fname=save_path)
        return fig, ax

