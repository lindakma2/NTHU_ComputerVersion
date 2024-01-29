###
### This homework is modified from CS231.
###


import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
計算兩個camera之間的變換矩陣
'''
def estimate_initial_RT(E):
    
    # TODO: Implement this method!
    U,D,VT=np.linalg.svd(E)
    Z = np.mat([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 0]])
    W = np.mat([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]])
    M=U*Z*U.transpose()
    Q1=U*W*VT
    Q2=U*W.transpose()*VT
    R1=np.linalg.det(Q1)*Q1
    R2=np.linalg.det(Q2)*Q2
    T1 = U[:, 2].reshape(-1,1)
    T2 = (-1)* T1
    
    RT1 = np.concatenate((R1,T1), axis=1)
    RT2 = np.concatenate((R1,T2), axis=1)
    RT3 = np.concatenate((R2,T1), axis=1)
    RT4 = np.concatenate((R2,T2), axis=1)
    
    RT_set = []
    RT_set.append(RT1)
    RT_set.append(RT2)
    RT_set.append(RT3)
    RT_set.append(RT4)
    RT_set=np.array(RT_set)
    
    return RT_set
    raise Exception('Not Implemented Error')

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
計算3D的最佳線性估計
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    N = image_points.shape[0]   #兩個相機矩陣
    A = np.zeros((2*N, 4))  
    A_set = []

    for i in range(N):
        A_set.append(image_points[i,0] * camera_matrices[i,2,:] - camera_matrices[i,0,:])
        A_set.append(image_points[i,1] * camera_matrices[i,2,:] - camera_matrices[i,1,:])
        
    A_set = np.array(A_set)
    u, s, vt = np.linalg.svd(A_set)
    
    min_ans=vt.shape[0]-1 #vt的最後一列是min eigen value
    p = vt[min_ans]
    norm=np.shape(p)[0]-1 #要把ATA norm
    p = p / p[norm]  
    p = p[:3]  
    return p

    raise Exception('Not Implemented Error')

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2M reprojection error vector
    給3D point在影像中的對應平面，計算重新投影的誤差向量
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    
    N = image_points.shape[0]
    p=np.append(point_3d,[1])
    y=camera_matrices.dot(p)
    
    error_set=[]
       
    for i in range(N):
        pi_prime = 1.0 / y[i,2] * y[i,:2]      #重新投影成2D點
        error = pi_prime - image_points[i]
        error_set.append(error[0])        
        error_set.append(error[1])        
    error_set = np.array(error_set)
    return error_set
    raise Exception('Not Implemented Error')

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
計算重新投影的誤差向量的jacobian行列式
'''
def jacobian(point_3d, camera_matrices):
    
    N = camera_matrices.shape[0]
    J = np.zeros((2 * N, 3))        

    
    p=np.append(point_3d,[1])
    y=camera_matrices.dot(p)
    jb = np.zeros((2*N, 3))
    for i in range(N):
        Mi = camera_matrices[i]
        jb[2*i, 0] = (Mi[0, 0] * y[i,2] - y[i,0] * Mi[2, 0]) / y[i,2] ** 2
        jb[2*i, 1] = (Mi[0, 1] * y[i,2] - y[i,0] * Mi[2, 1]) / y[i,2] ** 2
        jb[2*i, 2] = (Mi[0, 2] * y[i,2] - y[i,0] * Mi[2, 2]) / y[i,2] ** 2
        jb[2*i+1, 0] = (Mi[1, 0] * y[i,2] - y[i,1] * Mi[2, 0]) / y[i,2] ** 2
        jb[2*i+1, 1] = (Mi[1, 1] * y[i,2] - y[i,1] * Mi[2, 1]) / y[i,2] ** 2
        jb[2*i+1, 2] = (Mi[1, 2] * y[i,2] - y[i,1] * Mi[2, 2]) / y[i,2] ** 2

    return jb

    raise Exception('Not Implemented Error')

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
計算迭代更新點的3D點
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    P=linear_estimate_3d_point(image_points, camera_matrices)
    for i in range(10):
        J = jacobian(P, camera_matrices)
        e = reprojection_error(P, image_points, camera_matrices)
        P=P-np.linalg.inv(J.T.dot(J)).dot(J.T).dot(e)
    return P
    raise Exception('Not Implemented Error')

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras 兩個相機之間的轉換矩陣
    image_points - N measured points in each of the M images (NxMx2 matrix) M張照片每張有N個測量點
    K - the intrinsic camera matrix 原有的相機矩陣
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
計算兩個camera之間的RT
'''
def estimate_RT_from_E(E, image_points, K):
    # TODO: Implement this method!
    # TODO: Implement this method!
    RT = estimate_initial_RT(E)
    N = image_points.shape[0]
    I = np.array([[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0]])     #原相機初始值

    M1 = K.dot(I)     #第一個相機的變換矩陣
    for i in range(RT.shape[0]): 
        count = 0
        max_points = 0
        rt = RT[i]
        M2 = K.dot(rt) #第二個相機的變換矩陣
        camera_matrices = np.array([M1, M2]) 
        for j in range(N):   #估計nonlinear的3D點
            estimated_point = nonlinear_estimate_3d_point(image_points[j], camera_matrices)
            if (estimated_point[2]) > 0:
                count += 1
        if count > max_points:
            max_points = count
            correct_RT = RT[i]
    return correct_RT
    raise Exception('Not Implemented Error')

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)


    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    np.save('results.npy', dense_structure)
    print ('Save results to results.npy!')
