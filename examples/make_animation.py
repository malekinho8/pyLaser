import sys; sys.path.append('./')
import numpy as np
from utils import *
from manim import *

# This code basically takes in the laser and mirror input orientation and outputs the reflected laser orientation at each time step, and forms an animation.
# To view the animation, run the following command in the terminal:
# manim -pql examples/make_animation.py LaserReflection3D -r 1440,1080

class LaserReflection3D(ThreeDScene):
    def construct(self):
        # set camera perspective
        # self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        # set camera perspective to be looking down towards the xy plane from the -y direction towards the +y direction
        # self.set_camera_orientation(phi=20 * DEGREES, theta=-90 * DEGREES, distance=20)
        # self.set_camera_orientation(phi=90 * DEGREES, theta=-5 * DEGREES, distance=20)
        # self.set_camera_orientation(phi=90 * DEGREES, theta=-88 * DEGREES, distance=20)
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES, distance=20)

        camera_rotation_rate = 0.2

        # Specify the laser and mirror parameters
        laser_length = 3
        reflected_length = 3
        mirror_size = 2
        degrees_per_second = 4

        # Specify the orientation of the mirror using roll, pitch, and yaw
        mirror_roll = 10
        mirror_pitch = 0
        mirror_yaw = 0

        # Define the end pitch of the mirror
        mirror_roll_end = 0

        if mirror_roll_end < mirror_roll:
            dt_sign = -1
        else:
            dt_sign = 1

        # Set the incident laser vector
        azimuth_angle = 45  # Sample value, you can change this
        polar_angle = 155    # Sample value, you can change this
        incident_vector = calculate_incident_vector(azimuth_angle, polar_angle)
        incident_point = ORIGIN
        incident_start = incident_point - laser_length * np.array(incident_vector)

        # Create the laser
        laser = Line(incident_start, incident_point, color=RED)

        # Define the mirror and its normal
        mirror = Square(side_length=mirror_size).set_fill(GOLD, opacity=0.2)
        mirror.set_stroke(width=0.5, color=GOLD_A)

        # Set the orientation of the mirror based on the roll, pitch, and yaw
        mirror.rotate(mirror_roll * DEGREES, axis=RIGHT)
        mirror.rotate(mirror_pitch * DEGREES, axis=OUT)
        mirror.rotate(mirror_yaw * DEGREES, axis=UP)

        # calculate the normal vector of the mirror
        mirror_normal = calculate_plane_normal(mirror_roll, mirror_pitch, mirror_yaw)

        # Reflect the laser off the mirror as an Arrow3D object
        reflected_direction = calculate_reflected_vector(incident_vector, mirror_normal)
        reflected_end = reflected_length * np.array(reflected_direction)
        # reflected_laser = Arrow3D(ORIGIN, reflected_end, color=RED)
        reflected_laser = Line(ORIGIN, reflected_end, color=RED)

        # Define a dot at the end of the reflected laser
        reflected_dot = Dot3D(reflected_end, color=YELLOW)

        # Define a traced path for the reflected laser
        reflected_path = VMobject(color=YELLOW)
        reflected_path.set_points_as_corners([reflected_end, reflected_end])

        # define a ValueTracker to keep track of the angle of rotation
        angle_tracker = ValueTracker(mirror_roll)

        # define a tracker to keep track of the reflected laser vector components
        reflected_laser_x = ValueTracker(reflected_direction[0])
        reflected_laser_y = ValueTracker(reflected_direction[1])
        reflected_laser_z = ValueTracker(reflected_direction[2])

        # Define a label to display the reflected vector components
        reflected_laser_x_label = MathTex(r"R_x = {:05.2f}".format(reflected_direction[0])).to_corner(UL)
        reflected_laser_y_label = MathTex(r"R_y = {:05.2f}".format(reflected_direction[1])).next_to(reflected_laser_x_label, DOWN)
        reflected_laser_z_label = MathTex(r"R_z = {:05.2f}".format(reflected_direction[2])).next_to(reflected_laser_y_label, DOWN)
        
        # Define a label to display the angle of rotation to sit in the top right corner
        angle_label = MathTex(r"\theta = {:03}^\circ".format(mirror_roll)).to_corner(UR)        

        # Add the labels to the scene
        self.add_fixed_in_frame_mobjects(angle_label, reflected_laser_x_label, reflected_laser_y_label, reflected_laser_z_label)

        # Create 3D Axes
        axes = ThreeDAxes()
        
        # Create labels for the axes
        x_label = MathTex("X").next_to(axes.x_axis, RIGHT)
        y_label = MathTex("Y").next_to(axes.y_axis, UP)
        z_label = MathTex("Z").next_to(axes.z_axis, DOWN)

        # Add everything to the scene
        self.add(axes, x_label, y_label, z_label)

        # Define a function to update the reflected laser path
        def update_reflected_path(path):
            previous_path = path.copy()
            previous_last_point = previous_path.points[-1]
            new_point = reflected_dot.get_center()
            if np.linalg.norm(new_point - previous_last_point) > 0.1:  # this is to prevent too many points that are too close together
                path.add_smooth_curve_to(new_point)

        # Define an updater to rotate the mirror at a constant rate and update a label showing the value of rotation
        def update_mirror_rotation(mob, dt):
            d_angle = degrees_per_second * dt * dt_sign
            mob.rotate(d_angle * DEGREES, axis=RIGHT)
            angle_tracker.increment_value(d_angle)

        def update_reflected_laser(mob):
            current_normal = calculate_plane_normal(angle_tracker.get_value(), mirror_pitch, mirror_yaw)
            new_reflected_dir = calculate_reflected_vector(incident_vector, current_normal)
            reflected_laser_x.set_value(new_reflected_dir[0])
            reflected_laser_y.set_value(new_reflected_dir[1])
            reflected_laser_z.set_value(new_reflected_dir[2])
            mob.become(Line(ORIGIN, reflected_length * np.array(new_reflected_dir), color=RED))
        
        # define an updater function to keep the angle label updated
        def update_angle_label(mob):
            formatted_angle = "{:03}".format(int(angle_tracker.get_value()))
            new_label = MathTex(r"\theta = " + formatted_angle + r"^\circ").to_corner(UR)
            mob.become(new_label)
        
        # define an updater function to keep the reflected vector components updated
        def update_reflected_laser_x(mob):
            new_label = MathTex(r"R_x = {:05.2f}".format(reflected_laser_x.get_value())).to_corner(UL)
            mob.become(new_label)
        
        def update_reflected_laser_y(mob):
            new_label = MathTex(r"R_y = {:05.2f}".format(reflected_laser_y.get_value())).next_to(reflected_laser_x_label, DOWN)
            mob.become(new_label)
        
        def update_reflected_laser_z(mob):
            new_label = MathTex(r"R_z = {:05.2f}".format(reflected_laser_z.get_value())).next_to(reflected_laser_y_label, DOWN)
            mob.become(new_label)

        # Rotate the scene to show the 3D effect
        self.begin_ambient_camera_rotation(rate=camera_rotation_rate)

        # Play the creation of the mirror.
        self.play(Create(mirror), run_time=0.25)

        # Play the first laser coming in.
        self.play(Create(laser), run_time=0.25)

        # Show the reflection.
        self.play(Create(reflected_laser), run_time=0.25)

        self.wait(10)

        # Now, add the updaters for mirror rotation and other necessary updaters after the objects have been played/loaded.
        angle_label.add_updater(update_angle_label)
        mirror.add_updater(update_mirror_rotation)
        reflected_laser.add_updater(update_reflected_laser)
        reflected_laser_x_label.add_updater(update_reflected_laser_x)
        reflected_laser_y_label.add_updater(update_reflected_laser_y)
        reflected_laser_z_label.add_updater(update_reflected_laser_z)
        reflected_path.add_updater(update_reflected_path)
        reflected_dot.add_updater(lambda m: m.move_to(reflected_laser.get_end()))

        self.add(reflected_path, reflected_dot)

        # Rotate the mirror by x degrees.
        dT = np.abs((mirror_roll_end - mirror_roll)) / degrees_per_second
        self.wait(dT)

        # stop the rotation
        mirror.clear_updaters()
        reflected_laser.clear_updaters()
        angle_label.clear_updaters()
        reflected_laser_x_label.clear_updaters()
        reflected_laser_y_label.clear_updaters()
        reflected_laser_z_label.clear_updaters()

        self.wait(20)

        # stop the camera rotation
        self.stop_ambient_camera_rotation()