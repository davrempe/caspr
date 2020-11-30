import numpy as np
import open3d as o3d

import torch

AXIS_MAP = {'x' : np.array([1.0, 0.0, 0.0]), 
            'y' : np.array([0.0, 1.0, 0.0]),
            'z' : np.array([0.0, 0.0, 1.0])}

def random_rotation():
    '''
    Returns a uniformly sampled rotation matrix.
    Uses method at http://planning.cs.uiuc.edu/node198.html to generate
    a random uniform quaternion which is then converted
    '''
    u = np.random.uniform(size=3)
    coeff1 = 2.0 * np.pi * u[1]
    coeff2 = 2.0 * np.pi * u[2]
    w = np.sqrt(1.0 - u[0]) * np.sin(coeff1)
    x = np.sqrt(1.0 - u[0]) * np.cos(coeff1)
    y = np.sqrt(u[0]) * np.sin(coeff2)
    z = np.sqrt(u[0]) * np.cos(coeff2)

    R = o3d.geometry.get_rotation_matrix_from_quaternion(np.array([w, x, y, z]))

    return R

def rotation_axis(axis, angle):
    '''
    Return rot matrix around the given axis [x,y,or z] and angle.
    '''
    axis /= np.linalg.norm(axis)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis*angle)

def random_rotation_axis(axis):
    '''
    Returns a random rotation matrix around a given
    principle axis (x, y, or z)
    '''
    if axis not in AXIS_MAP:
        print('Axis must be x, y, or z!')
        return None
    rot_axis = AXIS_MAP[axis]
    rot_angle = np.random.uniform(low=0.0, high=2.0*np.pi)

    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis*rot_angle)
    
    return R

def random_sphere_point():
    '''
    Returns a random point on the surface of a unit sphere.
    Using method from: http://mathworld.wolfram.com/SpherePointPicking.html.
    '''
    u = np.random.uniform(low=-1.0, high=1.0)
    theta = np.random.uniform(low=0, high=2.0*np.pi)
    coeff = np.sqrt(1.0 - u**2)
    x = coeff * np.cos(theta)
    y = coeff * np.sin(theta)
    z = u
    return np.array([x,y,z])

def random_sphere_points(num_points, radius=0.5):
    '''
    Returns random points inside a sphere of the given radius
    Using method from: https://stackoverflow.com/a/5408843.
    '''
    costheta = np.random.uniform(low=-1.0, high=1.0, size=num_points)
    phi = np.random.uniform(low=0, high=2.0*np.pi, size=num_points)
    u = np.random.uniform(low=0, high=1.0, size=num_points)
    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)

    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )
    vol_points = np.array([x, y, z]).T
    return vol_points

def sphere_surface_points(num_points, radius=0.5):
    # sample uniformly
    uniform_cube = np.random.uniform(low=-1.0, high=1.0, size=(num_points,3))
    norm_uniform = uniform_cube / np.linalg.norm(uniform_cube, axis=1).reshape((-1,1))
    random_surface = norm_uniform*radius
    return random_surface

def normals_to_angles(normals):
    '''
    Given a batch of normals at each time step, converts to a 2D angle representation.
    normals : B x T x N x 3
    '''
    x2y2 = torch.norm(normals[:,:,:,:2], dim=3)
    theta = torch.atan(x2y2 / normals[:,:,:,2]).unsqueeze(3) # norm(x, y) / z
    theta[theta < 0] += np.pi # make angles from 0 to pi rather than discontinously from pi/2 to -pi/2
    phi = torch.atan2(normals[:,:,:,1], normals[:,:,:,0]).unsqueeze(3) # y / x
    phi[phi < 0] += 2.0*np.pi # make angles from 0 to 2pi rather -pi to pi
    angles = torch.cat([theta, phi], dim=3)
    return angles

def angles_to_normals(angles):
    '''
    Given a batch of angles representating points on the unit sphere, 
    converts them to the 3D euclidean represenation.

    angles : B x T x N x 2
    '''
    theta = angles[:,:,:,0]
    # theta[theta < 0] += np.pi
    phi = angles[:,:,:,1]
    x = torch.sin(theta)*torch.cos(phi)
    y = torch.sin(theta)*torch.sin(phi)
    z = torch.cos(theta)
    normals = torch.stack([x, y, z], dim=3)
    return normals

if __name__=='__main__':
    import matplotlib.pyplot as plt
    num_samples = 1000

    samples = np.zeros((num_samples, 3))
    vec = np.array([1.0, 0.0, 0.0])
    for i in range(num_samples):
        R = random_rotation()
        rvec = np.dot(R, vec)
        samples[i] = rvec

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-1.0, 1.0)
    ax.set_zlim3d(-1.0, 1.0)
    ax.set_ylim3d(-1.0, 1.0)
    plt.plot(samples[:,0], samples[:,1], samples[:,2], 'o')
    plt.show()