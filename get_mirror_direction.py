import numpy as np
import matplotlib.pyplot as plt

# use latex serif font for rendering in matplotlib
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

def reflect(n, m):
    """Reflects vector n over a surface with normal m.
       Returns the reflected vector in 3D.
    """
    return n - 2 * np.dot(n, m) * m

# Define 3D vectors
n = np.array([8, -8, -1])  # Incoming beam direction
n = n / np.linalg.norm(n)  # Normalize to unit vector

h_expected = np.array([1, 1, -5])  # Expected reflected beam direction
h_expected = h_expected / np.linalg.norm(h_expected)  # Normalize to unit vector

# Mirror's normal as bisector of the angle between n and -h_expected
m = (h_expected - n)
m = m / np.linalg.norm(m)  # Normalize the vector

h_calculated = reflect(n, m)

# Verify
if np.allclose(h_expected, h_calculated):
    print("The calculated reflection is correct!")
else:
    print("There seems to be an error in the calculation.")

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the incoming beam so it appears to be traveling towards the origin
ax.quiver(-n[0], -n[1], -n[2], n[0], n[1], n[2], color='blue', label='Incoming beam (n)')

ax.quiver(0, 0, 0, h_expected[0], h_expected[1], h_expected[2], color='green', label='Expected reflected beam (h)')
ax.quiver(0, 0, 0, m[0], m[1], m[2], color='red', label='Mirror normal (m)')

# find the component of the mirror normal vector with the largest absolute value
max_component = np.argmax(np.abs(m))

d = 0  # The plane passes through the origin

# set limits of three axes
xmin, xmax = -2, 2
ymin, ymax = -2, 2
zmin, zmax = -2, 2

if max_component == 0:  # If x has the largest magnitude
    # use limits of y and z axes
    yy, zz = np.meshgrid(range(ymin, ymax), range(zmin, zmax))
    xx = (-m[1] * yy - m[2] * zz - d) / m[0]
elif max_component == 1:  # If y has the largest magnitude
    # use limits of x and z axes
    xx, zz = np.meshgrid(range(xmin, xmax), range(zmin, zmax))
    yy = (-m[0] * xx - m[2] * zz - d) / m[1]
else:  # If z has the largest magnitude
    # use limits of x and y axes
    xx, yy = np.meshgrid(range(xmin, xmax), range(ymin, ymax))
    zz = (-m[0] * xx - m[1] * yy - d) / m[2]

# assert that the plane equation holds for all points
assert np.allclose(m[0] * xx + m[1] * yy + m[2] * zz + d, 0)

# Define the point (the origin in this case)
point = np.array([0, 0, 0])

# Plot the plane
ax.plot_surface(xx, yy, zz, color='cyan', alpha=0.5)


ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)
ax.legend()
plt.title("3D Vector Reflection with Plane")
plt.show()
