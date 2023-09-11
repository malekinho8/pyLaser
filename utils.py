import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D

# use latex for rendering in matplotlib
plt.rc('text', usetex=True)

def run(laser_azimuth, laser_polar, mirror_roll, mirror_pitch, mirror_yaw, intersection_plane_roll, intersection_plane_pitch, intersection_plane_yaw, intersection_plane_point):
    incident_vector = calculate_incident_vector(laser_azimuth, laser_polar)
    plane_normal = calculate_plane_normal(mirror_roll, mirror_pitch, mirror_yaw)
    reflected_vector = calculate_reflected_vector(incident_vector, plane_normal)
    azimuth, polar = convert_to_spherical_coordinates(reflected_vector)
    intersection_plane_normal = calculate_plane_normal(intersection_plane_roll, intersection_plane_pitch, intersection_plane_yaw)
    intersection_point = compute_intersection(intersection_plane_point, intersection_plane_normal, np.array([0,0,0]) ,reflected_vector)
    visualize_beam_and_surface_with_plotly(incident_vector, reflected_vector, plane_normal)
    print(f'Reflected Vector Azimuth: {azimuth} degrees')
    print(f'Reflected Vector Polar: {polar} degrees')
    print(f'Photodiode Intersection with Reflected Vector Point: {np.round(intersection_point, 3)}')

def calculate_incident_vector(azimuth, polar, r=1):
    """
    Function to calculate the direction vector of the incident laser beam from its azimuth and polar angles.

    Parameters
    ----------
    azimuth : float
        Azimuth angle of the incident beam in degrees.
    polar : float
        Polar angle of the incident beam in degrees.
    
    Returns
    -------
    incident_vector : numpy.ndarray
        Direction vector of the incident beam in Cartesian coordinates (x, y, z).
    """
    # convert the azimuth and polar angles to radians
    azimuth_rad = np.deg2rad(azimuth)
    polar_rad = np.deg2rad(polar)

    # calculate the x, y, and z components of the incident vector
    x = np.cos(azimuth_rad) * np.sin(polar_rad)
    y = np.sin(azimuth_rad) * np.sin(polar_rad)
    z = np.cos(polar_rad)

    # convert the x, y, and z components to a unit vector
    incident_vector = np.array([x, y, z])
    incident_vector = incident_vector / np.linalg.norm(incident_vector)

    # scale the unit vector by the radius of the sphere
    incident_vector = incident_vector * r
    
    return incident_vector 

def calculate_plane_normal(roll, pitch, yaw):
    """
    Function to calculate the normal vector to the reflecting plane from its roll, pitch, and yaw.

    Parameters
    ----------
    roll : float
        Roll angle of the reflecting plane in degrees.
    pitch : float
        Pitch angle of the reflecting plane in degrees.
    yaw : float
        Yaw angle of the reflecting plane in degrees.
    
    Returns
    -------
    plane_normal : numpy.ndarray
        Normal vector to the reflecting plane in Cartesian coordinates (x, y, z).
    """
    # converting degrees to radians
    roll = np.deg2rad(roll)      
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    plane_normal = R[:,2]

    # convert to a unit vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    return plane_normal

def calculate_reflected_vector(incident_vector, plane_normal, r=1):
    """
    Function to calculate the direction vector of the reflected beam given the incident vector and the normal vector.

    Parameters
    ----------
    incident_vector : numpy.ndarray
        Direction vector of the incident beam in Cartesian coordinates (x, y, z).
    plane_normal : numpy.ndarray
        Normal vector to the reflecting plane in Cartesian coordinates (x, y, z).
    
    Returns
    -------
    reflected_vector : numpy.ndarray
        Direction vector of the reflected beam in Cartesian coordinates (x, y, z).
    """
    # calculate the reflected vector
    reflected_vector = incident_vector - 2 * np.dot(incident_vector, plane_normal) * plane_normal

    # normalize the reflected vector
    reflected_vector = reflected_vector / np.linalg.norm(reflected_vector)

    # scale the reflected vector by the radius of the sphere
    reflected_vector = reflected_vector * r

    return reflected_vector

def convert_to_spherical_coordinates(reflected_vector):
    """
    Function to convert the direction vector of the reflected beam back to azimuth and polar angles.

    Parameters
    ----------
    reflected_vector : numpy.ndarray
        Direction vector of the reflected beam in Cartesian coordinates (x, y, z).
    
    Returns
    -------
    azimuth : float
        Azimuth angle of the reflected beam in degrees.
    polar : float
        Polar angle of the reflected beam in degrees.
    """
    # calculate the azimuth and polar angles
    azimuth = np.rad2deg(np.arctan2(reflected_vector[1], reflected_vector[0]))
    polar = np.rad2deg(np.arccos(reflected_vector[2]))

    # return the azimuth and polar angles
    return azimuth, polar

def visualize_beam_and_surface(incident_vector, reflected_vector):
    """
    Function to generate a 3D matplotlib visualization of the incident and reflected beams and the reflecting surface.

    Parameters
    ----------
    incident_vector : numpy.ndarray
        Direction vector of the incident beam in Cartesian coordinates (x, y, z).
    reflected_vector : numpy.ndarray
        Direction vector of the reflected beam in Cartesian coordinates (x, y, z).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add the incident beam to the figure
    ax.plot([-incident_vector[0],0], [-incident_vector[1],0], [-incident_vector[2],0], color='blue', label='Incident Beam')

    # Add the reflected beam to the figure
    ax.plot([0,reflected_vector[0]], [0,reflected_vector[1]], [0,reflected_vector[2]], color='red', label='Reflected Beam')

    # Set the axis labels and title with LaTeX
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')

    ax.legend()

    # Determine maximum absolute value across all vector components
    max_value = np.max(np.abs([incident_vector, reflected_vector]))

    # Set symmetric axis limits
    ax.set_xlim([-max_value, max_value])
    ax.set_ylim([-max_value, max_value])
    ax.set_zlim([-max_value, max_value])

    plt.title(r'Incident and Reflected Beams')

    # Make the plot a "tight fit"
    # plt.tight_layout()

    # Show the figure
    plt.show()

def compute_intersection(plane_point, plane_normal, line_point, reflected_vector):
    """
    Function to calculate the intersection of a line and a plane. The line is defined by a point and a direction 
    vector, and the plane is defined by a point and a normal vector.
    """
    # Normalize vectors to ensure their magnitudes are 1
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    reflected_vector = reflected_vector / np.linalg.norm(reflected_vector)

    # Calculate the value t
    t = np.dot(plane_point - line_point, plane_normal) / np.dot(reflected_vector, plane_normal)

    # Calculate the intersection point
    intersection_point = line_point + t * reflected_vector

    return intersection_point

def visualize_beam_and_surface_with_plotly(incident_vector, reflected_vector, normal_vector=None):
    """
    Function to generate a 3D plotly visualization of the incident and reflected beams and the reflecting surface.

    Parameters
    ----------
    incident_vector : numpy.ndarray
        Direction vector of the incident beam in Cartesian coordinates (x, y, z).
    reflected_vector : numpy.ndarray
        Direction vector of the reflected beam in Cartesian coordinates (x, y, z).
    """
    # initialize the plotly figure
    fig = go.Figure()

    # add the incident beam to the figure
    fig.add_trace(go.Scatter3d(
        x=[-incident_vector[0],0], # 0 indicates the starting value
        y=[-incident_vector[1],0],
        z=[-incident_vector[2],0],
        name='Incident Beam',
        mode='lines',
        line=dict(
            color='blue',
            width=5
        )
    ))

    # add the reflected beam to the figure
    fig.add_trace(go.Scatter3d(
        x=[0, reflected_vector[0]],
        y=[0, reflected_vector[1]],
        z=[0, reflected_vector[2]],
        name='Reflected Beam',
        mode='lines',
        line=dict(
            color='red',
            width=5
        )
    ))

    # add the normal vector to the figure
    if normal_vector is not None:
        fig.add_trace(go.Scatter3d(
            x=[0, normal_vector[0]],
            y=[0, normal_vector[1]],
            z=[0, normal_vector[2]],
            name='Normal Vector',
            mode='lines',
            line=dict(
                color='black',
                width=5
            )
        ))

    # set the axes labels and title
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[-1, 1],
                title='X'
            ),
            yaxis=dict(
                range=[-1, 1],
                title='Y'
            ),
            zaxis=dict(
                range=[-1, 1],
                title='Z'
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(
                    x=0,
                    y=-2,
                    z=0.2
                )
            )
        ),
        title='Incident and Reflected Beams'
    )

    # show the figure
    fig.show()