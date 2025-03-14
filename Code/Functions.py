import numpy as np
from scipy.optimize import least_squares
import os

def EstimateFundamentalMatrix(pts1, pts2):
    """
    Estimate fundamental matrix F using 8-point algo
    pts1, pts2 : Nx2 aarrays of correspoinding points (x,y)

    Returns:
    F: 3x3 fundamental matrix with rank 2
    """
    pts1 = pts1.astype(np.float64)
    pts2 = pts2.astype(np.float64)

    # Normalize for numerical stability
    # below uses a simple scale-shift normalization
    def normalize_points(pts):
        mean = np.mean(pts, axis=0) #cols
        std = np.std(pts, axis=0)
        T = np.array([
        [1/std[0],          0,      -mean[0]/std[0]],
        [0,         1/std[1],        -mean/std[1]],
        [0,              0,                  1]
        ])
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_norm = np.matmul(T, pts_h.T).T # transformation matrix T is designed to operate on column vectors
        return pts_norm[:, :2], T
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    N = pts1_norm.shape[0]
    A = np.zeros((N,9))
    for i in range(0,N):
        x1, y1 = pts1_norm[i, 0], pts1_norm[i, 1]
        x2, y2 = pts2_norm[i, 0], pts2_norm[i, 1]
        A[i] = [x1*x2, x2*y1, x2,
                y2*x1, y2*y1, y2,
                x1, y1, 1]
        
    #solve for Af = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    F_approx = Vt[-1].reshape(3,3)

    U_f, S_f, Vt_f = np.linalg.svd(F_approx)
    S_f[-1] = 0
    F_approx = U_f @ np.diag(S_f) @ Vt_f

    #de-normalize
    F = T2.T @ F_approx @ T1

    #scaling the F
    if F[2,2] > 1e-8:
        F = F/F[2,2]

    return F

def GetInlierRANSAC(pts1, pts2, threshold=0.001, max_iters=2000):
    """
    Estimate F and inliners via RANSAC
    Args:
        pts1, pts2 : Nx2 arrays of corresponding points
        threshold: inliner threshold for Sampson epipolar distance
        max_iters: number of RANSAC iterations
    Returns:
        F_best: 3x3 fundamental matric with rank=2
        inliers1, inliers2: The subset of inlier correspondences
    """
    pts1_h = np.hstack([pts1, np.ones(pts1.shape[0], 1)])
    pts2_h = np.hstack([pts2, np.ones(pts2.shape[0], 1)])

    best_inliers_count = 0
    F_best = None
    inlier_mask_best = None

    np.random.seed(42)

    for _ in range(max_iters):
        #Randomly sample 8 points
        sample_indices = np.random.choice(pts1.shape[0], 8, replace=False)
        sample_pts1 = pts1[sample_indices]
        sample_pts2 = pts2[sample_indices]

        #Estimate F from these 8 points
        F_candidate = EstimateFundamentalMatrix(sample_pts1, sample_pts2)

        # Compute errors (Sampson distance)
        # d = (x'^T F x)^2 / ( (F x)_0^2 + (F x)_1^2 + (F^T x')_0^2 + (F^T x')_1^2 )
        Fx1 = (F_candidate @ pts1_h.T).T
        Ftx2 = (F_candidate.T @ pts2_h.T).T
        x2tFx1 = np.sum(pts2_h * Fx1, axis=1)

        denom = Fx1[:,0]**2 + Fx1[:,1]**2 + Ftx2[:,0]**2 + Ftx2[:,1]**2
        denom[denom < 1e-12] = 1e-12
        dist = (x2tFx1**2) / denom

        inlier_mask = dist < threshold
        inlier_count = np.sum(inlier_mask)      

        if inlier_count > best_inliers_count:
            best_inliers_count = inlier_count
            F_best = F_candidate
            inlier_mask_best = inlier_mask
            
    #Final inliers
    inliers1 = pts1[inlier_mask_best]
    inliers2 = pts2[inlier_mask_best]

    # Re-estimate F using all inliers
    F_best = EstimateFundamentalMatrix(inliers1, inliers2)

    return F_best, inliers1, inliers2

def EssentialMatrixFromFundamentalMatrix(F, K):
    """
    Compute E from F, i.e. E = K^T * F * K, then
    enforce the singular values of E to be (1,1,0)

    Args:
        F: 3x3 fundamental matrix
        K: 3x3 camera intrinsic matrix
    Returns:
        E: 3x3 essential matrix
    """
    E_approx = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E_approx)

    avg = (S[0] + S[1])/2.0
    E = U @ np.diag([avg, avg, 0]) @ Vt

    return E

def ExtractCameraPose(E):
    """
    Decompose E into the four possible (R,C)
    Return a list of possible (C, R) solutions.
    """
    U, S, Vt = np.linalg.svd(E)

    #Ensuring proper rotation (det(R)==1)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0],
                  [1, 0, 0]
                  [0, 0, 1]])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    C1 = U[:,2]
    C2 = -U[:,2]

    possible_poses = [
        (C1, R1),
        (C2, R1),
        (C1, R2),
        (C2, R2)
    ]


    # Fix orientation if det(R) < 0
    out_poses = []
    for (C, R) in possible_poses:
        if np.linalg.det(R) < 0:
            R = -R
            C = -C
        out_poses.append((C, R))

    return out_poses


def LinearTriangulation(pts1, pts2, P1, P2):
    """
    Linear triangulation for each pair of points

    pts1, pts2: Nx2 arrays
    P1, P2: 3x4 camera projection matrices (product of Intrinsic and Extrinsic)
    Returns:
        X: Nx3 array of triangulated points in 3D(not homogeneous)
    """
    X_3d = []
    for i in range(pts1.shape[0]):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.vstack([
            x1 * P1[2,:] - P1[0,:],
            y1 * P1[2,:] - P1[1,:],
            x2 * P2[2,:] - P2[0,:],
            y2 * P2[2,:] - P2[1,:]
        ])
        
        # solve for X in AX = 0 using SVD
        _, _, Vt = np.linalg.svd(A)
        X_hom = Vt[-1]
        X_hom /= X_hom[-1]
        X_3d.append(X_hom[:3])

    return np.array(X_3d)

def DisambiguateCameraPose(possible_poses, pts1, pts2, K):
    """
    Among the 4 possible poses, pick the (C,R) that yields
    the largest no of points in front of both cameras.

    pts1, pts2: Nx2 arrays (in pixel coordinates)
    K: 3x3 intrinstic matrix

    Returns:
        C, R, X: the winning camera center, rotation, and 3D points(Nx3)
    """
    # The first camera (ref) is assumed at [0 0 0], R = I
    # so projection matrix is P1 = K[I|0]
    I = np.eye(3)
    zero = np.zeros((3,1))
    P1 = K @ np.hstack((I, zero))

    best_count = 0
    best_pose = None
    best_3d = None

    for (C,R) in possible_poses:
        # Build projection matrix for second camera
        # P2 = K [R | -R*C]
        C_col = C.reshape(3,1)
        t = -R @ C_col
        P2 = K @ np.hstack((R, t))

        # Triangulate
        X = LinearTriangulation(pts1, pts2, P1, P2)

        # Cheirality check
        X_cam2 = (R @ (X - C).T).T
        valid_1 = (X[:,2] > 0)
        valid_2 = (X_cam2[:,2] > 0)
        count_in_front = np.sum(valid_1 & valid_2)

        if count_in_front > best_count:
            best_count = count_in_front
            best_pose = (C, R)
            best_3d = X

    return best_pose[0], best_pose[1], best_3d

def NonLinearTriangulation(X_init, pts1, pts2, P1, P2, max_iter=50):
    """
    Perform per-point nonlinear refinement to minimize reprojection error
    X_init : Nx3 initial guesses of 3D points
    pts1, pts2 : Nx2 image points
    P1, P2 : 3x4 projection matrices
    Returns:
        X_refined: Nx3 refined 3D points
    """

    def reprojection_residual(X, x1, x2, P1, P2):
        # X is [X, Y, Z] in 3D
        X_h  = np.append(X, 1.0)

        # Reproject to image 1
        proj1 = P1 @ X_h
        proj1 /= proj1[2]
        err1 = proj1[:2] - x1

        #Reproject to image 2
        proj2 = P2 @ X_h
        proj2 /= proj2[2]
        err2 = proj2[:2] - x2

        return np.concatenate([err1, err2])
    
    X_refined = []
    for i in range(X_init.shape[0]):
        x1 = pts1[i]
        x2 = pts2[i]
        X0 = X_init[i]

        #Optimize
        res = least_squares(
            fun=reprojection_residual,
            x0=X0,
            args=(x1,x2,P1,P2),
            max_nfev=max_iter
        )
        X_refined.append(res.x)

    return np.array(X_refined)

def LinearPnP(X_3d, x_2d, K):
    """ 
    Solve for camera pose (C,R) from 2D-3D correspondences using 
    a simple linear PnP
    X_3d: Nx3
    x_2d: Nx2
    K: 3x3
    Returns:
        C: 3-vector (camera center)
        R: 3x3 rotation
    """
    # Convert to normalized coordinates
    # x_norm = inv(K) * x_2d_h
    ones = np.ones((x_2d.shape[0], 1))
    x_2d_h = np.hstack([x_2d, ones]).T # 3xN
    x_norm = np.linalg.inv(K) @ x_2d_h # 3xN

    #Build M in the equation M * [r1^T r2^T r3^T t]^T = 0
    # or use the standard linear system approach
    N = X_3d.shape[0]
    M = []
    for i in range(N):
        X, Y, Z = X_3d[i]
        u, v, w = x_norm[:,i] # w should be 1 after normalisation
        M.append([X,Y,Z,1,0,0,0,0,-u*X,-u*Y,-u*Z,-u])
        M.append([0,0,0,0,X,Y,Z,1, -v*X, -v*Y, -v*Z, -v])
    M = np.array(M) #2N x12

    _,_, Vt = np.linalg.svd(M)
    P = Vt[-1].reshape(3,4)

    R_approx = P[:,:3]
    t_approx = P[:, 3]

    U, S, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
        t_approx = -t_approx

    # s is a scale factor, get scale frm S
    # scale can be taken average of S or S[0] etc
    # typically, t = (1/S[0]) * t_approx if S is diagonal

    scale = np.mean(S)
    t = t_approx/scale

    C = -R.T @ t

    return C, R

def compute_pnp_inliers(X_3d, x_2d, C, R, K, threshold):
    """ 
    Compute which 2D-3D correspondences are inliers
    by projecting X_3d into the camera (C,R) and
    checking distance to x_2d in pixels
    """
    N = X_3d.shape[0]
    X_3d_h = np.hstack([X_3d, np.ones((N, 1))])

    # Build projection P = K[R | -R*C]
    C_col = C.reshape(3,1)
    t = -R @ C_col
    P = K @ np.hstack((R, t))

    #Project
    proj = (P @ X_3d_h.T).T
    proj[:,0] /= proj[:2]
    proj[:,1] /= proj[:2]

    # Compare with x_2d
    diff = proj[:,:2] - x_2d
    errors = np.linalg.norm(diff, axis=1)
    inliers_mask = errors < threshold
    
    return inliers_mask

def PnPRANSAC(X_3d, x_2d, K, threshold=3.0, max_iters=1000):
    """ 
    RANSAC-based robust PnP from 2D-3D correspondences.
    X_3d: Nx3
    x_2d: Nx2
    K: 3x3
    threshold: pixel reprojection eror threshold
    max_iters: RANSAC iterations
    returns:
        best_C, best_R, inliers mask
    """
    N = X_3d.shape[0]
    best_inliers_count = 0
    best_C, best_R = None, None
    best_mask = None

    np.random.seed(42)
    for _ in range(max_iters):
        sample_indices = np.random.choice(N, 6, replace=False)
        X_sample = X_3d[sample_indices]
        x_sample = x_2d[sample_indices]

        try:
            C_candidate, R_candidate = LinearPnP(X_sample, x_sample, K)
        except:
            continue

        # count inliers
        # project all X_3d to see how close to x_2d
        inliers_mask = compute_pnp_inliers(X_3d, x_2d, C_candidate, R_candidate, K, threshold)
        inliers_count = np.sum(inliers_mask)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_C = C_candidate
            best_R = R_candidate
            best_mask = inliers_mask

    # Re-estimate using all inliers
    if best_mask is not None and np.sum(best_mask) >=0:
        X_in = X_3d[best_mask]
        x_in = x_2d[best_mask]
        best_C, best_R = LinearPnP(X_in, x_in, K)

    return best_C, best_R, best_mask 

def NonLinearPnP(X_3d, x_2d, C_init, R_init, K, max_iter=50):
    """ 
    Refine camera pose (C, R) that minimize reprojection error, given an initial guess.
    For simplicity, er param by (rx, ry, rz) and (tx, ty, tz), i.e 6 parameters
    A better approach is to param rotation as a quaterion.

    X_3d: Nx3
    x_2d: Nx2
    C_init, R_init: initial guess
    K: 3x3 
    """
    # Convert R_init to axis-angle or euler angles
    # For simplicity, lets do naive Euler from R
    def rot_to_euler(R):
        sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
        if sy > 1e-6:
            rx = np.arctan2(R[2,1], R[2,2])
            ry = np.arctan2(-R[2,0], sy)
            rz = np.arctan2(R[1,0], R[0,0])
        else:
            rx = np.arctan2(-R[1,2], R[1,1])
            ry = np.arctan2(-R[2,0], sy)
            rz = 0
        return np.array([rx, ry, rz])
    
    def euler_to_rot(rx, ry, rz):
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx
        return R
    
    #initial
    eul_init = rot_to_euler(R_init)
    c_init = C_init
    param0 = np.hstack([c_init, eul_init])
    
    def projection_error(X_3d, x_2d, C, R, K):
        N = X_3d.shape[0]
        X_3d_h = np.hstack([X_3d, np.ones((N, 1))])

        #Build P
        C_col = C.reshape(3,1)
        t = -R @ C_col
        P = K @ np.hstack((R, t))

        proj = (P @ X_3d_h.T).T
        proj[:, 0] /= proj[:2]
        proj[:, 1] /= proj[:2]
        residuals = proj[:, :2] - x_2d
        return residuals.ravel() # shape= 2N

    def residual_func(params):
        c = params[0:3]
        rx, ry,rz = params[3:6]
        R = euler_to_rot(rx, ry, rz)
        return projection_error(X_3d, x_2d, c, R, K)
    
    # Optimize
    res = least_squares(residual_func, param0, max_nfev=max_iter)
    c_est = res.x[0:3]
    rx, ry, rz = res.x[3:6]
    R_est = euler_to_rot(rx, ry, rz)
    return c_est, R_est

def BuildVisibilityMatrix(matches_list, num_images, num_3d_points):
    """ 
    matches_list: data structure that indicates for each image i,
        which 3D points are matched ( and the 2D location).
        You can design your own structure or parse from your logic.
    num_images: I
    num_3d_points: J
    Returns:
        V: IxJ binary
    """
    V = np.zeros((num_images, num_3d_points), dtype=int)

    #Suppose we have a structure like:
    # matches_list[i] = list of (point3d_id, x, y) for that image i
    # Then we mark visibility
    for i in range(num_images):
        for (pid, x, y) in matches_list[i]:
            V[i, pid] = 1
    return V

def BundleAdjustment(Cset, Rset, X3d, K, matches, V, max_iter=50):
    """ 
    A simplified global BA: we pack all camera poses and 3D points
    into a single parameter vector, and solve with least_squares.

    Cset: list of camera centers for i=0...I-1
    Rset: list of rotations for i=0...I-1
    X3d: Nx3 of 3D points
    K: 3x3
    matches: a structure describing the 2D observations for each (i,j)
            e.g. matches[i][j] = (x, y) if V[i,j] = 1
    V: IxN visibility matrix, V[i, j] in {0,1}
    """
    I = len(Cset)
    N = X3d.shape[0]

    #Flatten parameters
    def rot_to_euler(R):
        sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
        if sy > 1e-6:
            rx = np.arctan2(R[2,1], R[2,2])
            ry = np.arctan2(-R[2,0], sy)
            rz = np.arctan2(R[1,0], R[0,0])
        else:
            rx = np.arctan2(-R[1,2], R[1,1])
            ry = np.arctan2(-R[2,0], sy)
            rz = 0
        return np.array([rx, ry, rz])
    
    param_cameras = []
    for i in range(I):
        C = Cset[i]
        R = Rset[i]
        eul = rot_to_euler[R]
        param_cameras.append(np.hstack([C, eul]))
    param_cameras = np.concatenate(param_cameras)
    param_points = X3d.ravel()

    params0 = np.concatenate([param_cameras, param_points])

    def unpack_params(params):
        #cameras
        cam_params = params[0:6*I]
        pt_params = params[6*I:]
        #reshape camera params
        cameras = []
        for i in range(I):
            c = cam_params[6*i : 6*i + 3]
            eul = cam_params[6*i + 3 : 6*i + 6]
            cameras.append((c, eul))
        #reshape 3d
        points_3d = pt_params.reshape((N,3))
        return cameras, points_3d
    
    def euler_to_rot(rx, ry, rz):
        Rx = np.array([
            [1, 0,      0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [ np.cos(ry), 0, np.sin(ry)],
            [ 0,    1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz),  np.cos(rz), 0],
            [0,           0,          1]
        ])
        return Rz @ Ry @ Rx
    
    def residual_function(params):
        cameras, points_3d = unpack_params(params)
        residuals = []
        for i in range(I):
            (c, eul) = cameras[i]
            rx, ry, rz = eul
            R = euler_to_rot(rx, ry, rz)
            C_col = c.reshape(3,1)
            t = -R @ C_col
            P = K @ np.hstack((R,t))

            for j in range(N):
                if V[i, j] == 1:
                    x_2d = matches[i][j]
                    X = np.append(points_3d[j], 1)
                    proj = P @ X
                    proj /= proj[2]
                    residuals.append(proj[0] - x_2d[0])
                    residuals.append(proj[1] - x_2d[1])
        return np.array(residuals)

    # Run Optimization
    result = least_squares(residual_function, params0, max_nfex= max_iter)

    # Unpaack final results
    cameras_opt, points_opt = unpack_params(result.x)
    Cset_opt = []
    Rset_opt = []
    for i in range(I):
        c, eul = cameras_opt[i]
        rx, ry, rz = eul
        R = euler_to_rot(rx, ry, rz)
        Cset_opt.append(c)
        Rset_opt.append(R)

    X3d_opt = points_opt

    return Cset_opt, Rset_opt, X3d_opt


def run_sfm_pipeline():
    """
    End-to-end pipeline to reconstruct a scene from 5 images of Unity Hall,
    using classical SfM steps:
      1) Load camera intrinsics from 'calibration.txt'
      2) Read pairwise matches from matching1..4.txt
      3) Bootstrap with images 1 & 2:
         - RANSAC => F
         - E = K^T F K
         - Extract 4 camera poses from E
         - Disambiguate by triangulation => pick best (C2, R2)
         - LinearTriangulation => initial 3D points
         - NonlinearTriangulation => refine 3D points
      4) For each new image (3,4,5):
         - Find 2D-3D correspondences
         - Solve with PnPRANSAC => get (C, R)
         - NonLinearPnP => refine (C, R)
         - Triangulate new 3D points if available
         - NonlinearTriangulation for newly added points
      5) Build Visibility Matrix
      6) Bundle Adjustment => refine all cameras and 3D points
      7) Save or visualize final results
    """

    # ------------------------------------------------
    # 1) Load intrinsics
    # ------------------------------------------------
    K = np.loadtxt("Data/calibration.txt")  # shape should be (3,3)

    # ------------------------------------------------
    # 2) Read pairwise matches from matching files
    # ------------------------------------------------
    # We have matching1.txt, matching2.txt, matching3.txt, matching4.txt
    # matching1 => I1->I2, I1->I3, I1->I4, I1->I5
    # matching2 => I2->I3, I2->I4, I2->I5
    # matching3 => I3->I4, I3->I5
    # matching4 => I4->I5
    #
    # We'll parse them into a dictionary: matches[(i,j)] = (pts_i, pts_j)
    # NOTE: The actual file format is more complex. Here we show a *schematic* parser.
    #
    matches = {}
    def parse_matching_file(filename, base_image_id):
        """
        A schematic parser that reads matchingX.txt
        and extracts correspondences from the 'base_image_id' to subsequent images.
        We return a dictionary of the form:
             {
               (base_image_id, next_img_id): (pts_base, pts_next),
               ...
             }
        """
        local_dict = {}
        # Example reading logic (pseudo-code):
        if not os.path.exists(filename):
            return local_dict

        with open(filename, "r") as f:
            lines = f.readlines()

        # lines[0] might be: "nFeatures: 3930"
        # subsequent lines describe features in base_image_id that match other images
        # YOU need to parse carefully. We'll show a naive approach.
        idx = 0
        # skip first line if it starts with 'nFeatures:' 
        if lines[idx].startswith('nFeatures'):
            idx += 1

        while idx < len(lines):
            row = lines[idx].strip().split()
            idx += 1
            if len(row) < 7:
                continue

            n_matches = int(row[0])
            # row[1:4] => R,G,B (can ignore or store)
            # row[4], row[5] => (u, v) in base_image
            u_base = float(row[4])
            v_base = float(row[5])

            # Then pairs: (image_id, u, v) repeated n_matches times
            # row has length = 6 + 3*n_matches
            # so we read from row[6..]
            sub_idx = 6
            for _ in range(n_matches):
                im_id = int(row[sub_idx])
                x_match = float(row[sub_idx + 1])
                y_match = float(row[sub_idx + 2])
                sub_idx += 3

                # We'll store the match base_image_id <-> im_id
                # We'll accumulate them in a local structure so we can combine them
                pair_key = (base_image_id, im_id)
                if pair_key not in local_dict:
                    local_dict[pair_key] = [[], []]  # [pts_base, pts_other]

                local_dict[pair_key][0].append([u_base, v_base])
                local_dict[pair_key][1].append([x_match, y_match])
        
        # convert lists to np.array
        out_dict = {}
        for k in local_dict.keys():
            arr_base = np.array(local_dict[k][0], dtype=float)
            arr_other = np.array(local_dict[k][1], dtype=float)
            out_dict[k] = (arr_base, arr_other)

        return out_dict

    # parse each matching file
    dict1 = parse_matching_file("Data/matching1.txt", base_image_id=1)
    dict2 = parse_matching_file("Data/matching2.txt", base_image_id=2)
    dict3 = parse_matching_file("Data/matching3.txt", base_image_id=3)
    dict4 = parse_matching_file("Data/matching4.txt", base_image_id=4)

    # merge them all into 'matches'
    for d in [dict1, dict2, dict3, dict4]:
        for key in d.keys():
            matches[key] = d[key]  # (pts_i, pts_j)

    # Now we have matches[(1,2)], matches[(1,3)], etc. up to matches[(4,5)].

    # ------------------------------------------------
    # 3) Pick images 1 & 2 for bootstrapping
    # ------------------------------------------------

    # Retrieve the correspondences between images 1 and 2
    if (1, 2) not in matches:
        raise ValueError("No matches found for (1,2). Cannot bootstrap!")
    pts1, pts2 = matches[(1,2)]  # Nx2, Nx2

    # 3a) Estimate F (via RANSAC)
    F_12, inliers1, inliers2 = GetInlierRANSAC(pts1, pts2, threshold=0.001, max_iters=2000)

    # 3b) Compute E
    E_12 = EssentialMatrixFromFundamentalMatrix(F_12, K)

    # 3c) Extract 4 possible camera poses
    possible_poses = ExtractCameraPose(E_12)

    # 3d) Disambiguate by triangulation + cheirality
    C2, R2, X_3d_lin = DisambiguateCameraPose(possible_poses, inliers1, inliers2, K)

    # 3e) Nonlinear triangulation to refine 3D
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    t2 = -R2 @ C2.reshape(3,1)
    P2 = K @ np.hstack((R2, t2))
    X_3d = NonLinearTriangulation(X_3d_lin, inliers1, inliers2, P1, P2)

    # Initialize camera poses:
    #   The first camera is at the origin (C1=[0,0,0], R1=I).
    Cset = [np.zeros(3), C2]  # for images 1 and 2
    Rset = [np.eye(3), R2]

    # We'll store the 3D points from inliers of (1,2). We also need some indexing
    # for these points. For simplicity, let's store them in a big array.
    X_global = X_3d.copy()  # Nx3

    # We'll also track which 2D points correspond to which row in X_global
    # For example, 'pt_index_12[i]' = index in X_global for inliers1[i], inliers2[i].
    pt_index_12 = np.arange(X_global.shape[0])  # 0..N-1

    # We also keep track of which images have these points. For example:
    #   image 1 sees them at inliers1, image 2 sees them at inliers2.
    # In a robust pipeline, you'd store a big table of 2D observations of each 3D point.
    # We'll keep it minimal here.
    # Let's define a dictionary storing the 2D observations:
    # obs[ (img_id, point_id) ] = (u, v)
    obs = {}
    for i in range(X_global.shape[0]):
        obs[(1, i)] = inliers1[i]  # in image #1
        obs[(2, i)] = inliers2[i]  # in image #2

    # ------------------------------------------------
    # 4) For each new image i = 3, 4, 5:
    #    - Identify the 2D-3D correspondences
    #    - PnPRANSAC => (C_i, R_i)
    #    - NonLinearPnP => refine (C_i, R_i)
    #    - Triangulate new points that appear in i with previously registered cameras
    #    - NonlinearTriangulation of newly added points
    # ------------------------------------------------

    current_num_points = X_global.shape[0]
    for new_img_id in [3, 4, 5]:
        # 4a) Identify which matches exist: we have matches with cameras already in {1,2}
        #     We collect 2D-3D correspondences from any existing 3D points
        #     i.e. if (1, new_img_id) is in matches, we find all correspondences that match
        #     a known 3D point. We do the same for (2, new_img_id).
        known_3d = []
        known_2d = []

        for reg_img_id in range(1, new_img_id):
            if (reg_img_id, new_img_id) in matches:
                base_pts, new_pts = matches[(reg_img_id, new_img_id)]
                # We want to see if each base_pts is among the observations from reg_img_id
                # We do a small approximate matching: for each base_pt, we see if it matches
                # something we stored in obs for (reg_img_id, ?).
                # In a real system, you’d likely keep track of a feature index or do direct dictionary lookup.

                for match_i, (bx, by) in enumerate(base_pts):
                    # we check each 3D point that image reg_img_id sees. 
                    # We'll do a naive "closest point" approach as an example:
                    # (In real code, you'd keep a robust index from the prior pipeline.)
                    best_3d_id = None
                    best_dist = 1e10
                    for (k_img, k_ptid) in obs.keys():
                        if k_img == reg_img_id:
                            # Compare with obs[(reg_img_id, k_ptid)]
                            (ox, oy) = obs[(k_img, k_ptid)]
                            dist = np.hypot(ox - bx, oy - by)
                            if dist < best_dist:
                                best_dist = dist
                                best_3d_id = k_ptid

                    # if the best_dist is small => we consider them the same 2D feature
                    if best_dist < 3.0:  # some threshold in pixels
                        # that means we have a known 3D point:
                        known_3d.append(X_global[best_3d_id])
                        # the new image's 2D coordinate is new_pts[match_i]
                        (ux, uy) = new_pts[match_i]
                        known_2d.append([ux, uy])

            # Similarly, if (new_img_id, reg_img_id) is in matches, parse them
            elif (new_img_id, reg_img_id) in matches:
                new_pts, base_pts = matches[(new_img_id, reg_img_id)]
                # similar logic as above, reversed roles
                for match_i, (nx, ny) in enumerate(new_pts):
                    best_3d_id = None
                    best_dist = 1e10
                    for (k_img, k_ptid) in obs.keys():
                        if k_img == reg_img_id:
                            (ox, oy) = obs[(k_img, k_ptid)]
                            dist = np.hypot(ox - nx, oy - ny)
                            if dist < best_dist:
                                best_dist = dist
                                best_3d_id = k_ptid

                    if best_dist < 3.0:
                        known_3d.append(X_global[best_3d_id])
                        (ux, uy) = base_pts[match_i]
                        known_2d.append([ux, uy])

        known_3d = np.array(known_3d)
        known_2d = np.array(known_2d)
        if known_3d.shape[0] < 6:
            # We need at least 6 to run linear PnP
            print(f"Not enough 2D-3D correspondences for image {new_img_id}. Skipping it.")
            # We'll store dummy pose
            Cset.append(np.zeros(3))
            Rset.append(np.eye(3))
            continue

        # 4b) Solve PnPRANSAC => (C_i, R_i)
        C_init, R_init, inliers_mask = PnPRANSAC(known_3d, known_2d, K, threshold=3.0, max_iters=1000)

        # 4c) Nonlinear PnP => refine
        X_in = known_3d[inliers_mask]
        x_in = known_2d[inliers_mask]
        C_ref, R_ref = NonLinearPnP(X_in, x_in, C_init, R_init, K, max_iter=50)

        # Save camera pose
        Cset.append(C_ref)
        Rset.append(R_ref)

        # 4d) Triangulate new points that appear in new_img_id with old cameras
        # We'll loop over all reg_img_id in [1..(new_img_id-1)], take matches,
        # and for each match that is not yet in 3D, we do a 2-view triangulation
        # from (reg_img_id, new_img_id).
        # Then we optionally refine those new 3D points with NonlinearTriangulation.
        # For brevity, let's just do it with the FIRST camera we already have, e.g. camera 1.

        if (1, new_img_id) in matches:
            old_pts, new_pts = matches[(1, new_img_id)]
            P_old = K @ np.hstack((np.eye(3), np.zeros((3,1))))  # for camera 1
            t_new = -R_ref @ C_ref.reshape(3,1)
            P_new = K @ np.hstack((R_ref, t_new))
            # Triangulate
            X_new_lin = LinearTriangulation(old_pts, new_pts, P_old, P_new)
            # Nonlinear refine
            X_new = NonLinearTriangulation(X_new_lin, old_pts, new_pts, P_old, P_new, max_iter=50)
            # Append to X_global
            base_idx_start = X_global.shape[0]
            X_global = np.vstack([X_global, X_new])
            # Update obs to store 2D observation in image1 and new_img_id
            for irow in range(X_new.shape[0]):
                ptid = base_idx_start + irow
                obs[(1, ptid)] = old_pts[irow]
                obs[(new_img_id, ptid)] = new_pts[irow]

        elif (new_img_id, 1) in matches:
            new_pts, old_pts = matches[(new_img_id, 1)]
            P_old = K @ np.hstack((np.eye(3), np.zeros((3,1))))  # camera 1
            t_new = -R_ref @ C_ref.reshape(3,1)
            P_new = K @ np.hstack((R_ref, t_new))
            X_new_lin = LinearTriangulation(old_pts, new_pts, P_old, P_new)
            X_new = NonLinearTriangulation(X_new_lin, old_pts, new_pts, P_old, P_new, max_iter=50)
            base_idx_start = X_global.shape[0]
            X_global = np.vstack([X_global, X_new])
            for irow in range(X_new.shape[0]):
                ptid = base_idx_start + irow
                obs[(1, ptid)] = old_pts[irow]
                obs[(new_img_id, ptid)] = new_pts[irow]

        # In a real pipeline, you'd do the above for all previously registered cameras, not just camera 1.

    # ------------------------------------------------
    # 5) Build Visibility Matrix
    # ------------------------------------------------
    # We have I=5 cameras and N=X_global.shape[0] points
    I = 5
    N = X_global.shape[0]
    # We'll build 'matches_list' as required by BuildVisibilityMatrix
    # matches_list[i] should be a list of (pt_id, u, v) for each 3D point observed by camera i+1
    # (Because we used i+1 as camera ID in the code above)
    matches_list = [[] for _ in range(I)]  # i=0..4 => camera 1..5
    for (img_id, pt_id) in obs.keys():
        # store into matches_list[img_id - 1]
        (ux, vy) = obs[(img_id, pt_id)]
        matches_list[img_id - 1].append((pt_id, ux, vy))

    V = BuildVisibilityMatrix(matches_list, num_images=I, num_3d_points=N)

    # ------------------------------------------------
    # 6) Bundle Adjustment
    # ------------------------------------------------
    # Cset, Rset so far => lists of length 5
    # X_global => Nx3
    # We'll need 'matches' argument that has, for each camera i, for each point j in V[i,j]==1, the 2D coords
    # We'll build 'matches_for_ba' as a list-of-lists: matches_for_ba[i][j] = (u, v) or None
    # Or we can do something simpler if your existing BundleAdjustment code expects
    # matches[i][j] to be 2D coords.

    # We'll build matches_for_ba as a list of length I, each of size N, storing (u, v)
    # or dummy if not visible.
    matches_for_ba = []
    for i_cam in range(I):
        row = []
        for j_pt in range(N):
            if V[i_cam, j_pt] == 1:
                # find the obs
                row.append(obs[(i_cam+1, j_pt)])
            else:
                row.append(None)
        matches_for_ba.append(row)

    # The user’s BundleAdjustment function might want them as row[j] = (u, v) or something.
    # We'll convert None -> (0,0) if code does not handle None:
    # (Adjust as needed for your function's signature)
    for i_cam in range(I):
        for j_pt in range(N):
            if matches_for_ba[i_cam][j_pt] is None:
                matches_for_ba[i_cam][j_pt] = (0., 0.)  # or dummy

    # Now run BA
    Cset_opt, Rset_opt, X_opt = BundleAdjustment(Cset, Rset, X_global, K,
                                                matches_for_ba, V, max_iter=50)

    # ------------------------------------------------
    # 7) Save or visualize
    # ------------------------------------------------
    print("==== Final Results ====")
    for i_cam in range(I):
        print(f"Camera {i_cam+1} center = {Cset_opt[i_cam]}")
        print(f"Camera {i_cam+1} rotation = \n{Rset_opt[i_cam]}")
    print("Sample of final 3D points:\n", X_opt[:5])

    # Done. We have refined Cset_opt, Rset_opt, X_opt for the 5 images.


if __name__ == "__main__":
    run_sfm_pipeline()