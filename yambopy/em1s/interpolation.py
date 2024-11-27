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
            self.qpts = UcFull1BZStaticScreening.car_qpoints
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
