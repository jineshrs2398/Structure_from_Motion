import numpy as np
from scipy.optimize import least_squares

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
            [1/std[0], 0, -mean[0]/std[0]],
            [0, 1/std[1], -mean/std[1]],
            [0, 0, 1]
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
    R2 = U @ W @ Vt
    R3 = U @ W.T @ Vt
    R4 = U @ W.T @ Vt

    C1 = U[:,2]
    C2 = -U[:,2]
    C3 = U[:,2]
    C4 = -U[:,2]

    if np.linalg.det(R1) < 0:
        R1 = -R1
        C1 = -C1
    if np.linalg.det(R2) < 0:
        R2 = -R2
        C2 = -C2
    if np.linalg.det(R3) < 0:
        R3 = -R3
        C3 = -C3
    if np.linalg.det(R4) < 0:
        R4 = -R4
        C4 = -C4

    possible_poses = [
        (C1, R1),
        (C2, R2),
        (C3, R3),
        (C4, R4)
    ]

    return possible_poses

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

        A = np.vstack[x1*P1[:,2] - P1[:,0], 
                      y1*P1[:,2] - P1[:,0],
                      x2*P2[:,2] - P2[:,1],
                      y2*P2[:,2] - P2[:,1]]
        
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
        err2 = proj2[2] - x2

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
    P = Vt[-1]
    P = P.reshape(3,4)

    R_approx = P[:,:3]
    t_approx = P[:, 3]

    U, S, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
        t_approx = -t_approx

    scale = np.mean(S)
    t = t_approx/scale

    C = -R.T @t

    return C, R

    