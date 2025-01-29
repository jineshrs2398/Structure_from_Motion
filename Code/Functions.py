import numpy as np

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





