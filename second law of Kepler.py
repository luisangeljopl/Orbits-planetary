#Import Statements
from math import sin, cos, atan2, sqrt, pi
import numpy as np

#matplotlib inline
import matplotlib.pyplot as plt

# The following two parameters are get us the strength of gravity at various heights:
g = 0.00981 
radius_of_earth = 6371.0 

# The following two variables define the cannonball launch conditions:
initial_position = [0.0, radius_of_earth + 500.0]  
# The value of 8.982 km/s was tweaked to make exactly one elliptical orbit in 200 minutes.
initial_velocity = [8.982, 0.0]                     

delta_t = 240        # simulation time step in seconds ****(120.0) (240)
simulation_time_steps = 50 # 200 minutes worth for a delta_t of 120.0 ****(100) (50)
# computes horizontal and vertical components of a vector and returns them as a tuple
def vector_from_length_and_angle(length: float, angle: float) -> np.ndarray:
    # we are working in degrees -- python's are expecting radians -- convert degrees to radian
    angle_in_radians = angle * pi / 180.0
    x_component = length * cos(angle_in_radians)
    y_component = length * sin(angle_in_radians)
    return np.array([x_component, y_component])

# get angle from components using atan2 version of arctangent function -- result is converted to degrees
def angle_from_vector(vector: np.ndarray) -> float:
    # use the arctangent function
    angle_in_radians = atan2(vector[1], vector[0])  
    # we are working in degrees -- python's functions return radians -- convert radians to degrees
    angle = angle_in_radians * 180.0 / pi
    # return the result
    return angle

# get length from components using Pythagorean theorem
def length_from_vector(vector: np.ndarray) -> float:
    length_squared = np.sum(vector**2)
    return sqrt(length_squared)
def strength_of_gravity(position):
    # this function encodes the strength of gravity as a function of distance from the center of the Earth
    radius = length_from_vector(position)
    strength = g * radius_of_earth**2 / radius**2
    return strength

def direction_of_gravity(position):
    # this function encodes the direction of gravity (the angle)
    # gravity is attractive -- it always points toward the center of the Earth
    direction = angle_from_vector(position) + 180.0
    return direction

def acceleration_of_gravity(position):
    # using the strength and direction functions you have just implemented compute and
    # return a 2x1 array for the acceleration of gravity
    strength = strength_of_gravity(position)
    direction = direction_of_gravity(position)
    acceleration = vector_from_length_and_angle(strength, direction)
    return acceleration
#computes the area of a triangle given two points in the orbit
def triangle_area(before_position, after_position):
    #gets length of two vectors from center point and averages their length
    r1 = length_from_vector(before_position)
    r2 = length_from_vector(after_position)
    height = (r1 + r2) / 2
    
    #gets angle from two vectors and uses it to compute the length of the new base of triangle
    delta_theta = -(angle_from_vector(after_position) - angle_from_vector(before_position))
    base = height * delta_theta * (2 * pi / 360)
    
    #returns standard area of triangle equation
    area = 0.5 * base * height
    return area
# Initialize the x and y velocities
velocities = np.zeros((simulation_time_steps, 2))
velocities[0] = initial_velocity

# Initialize the x and y positions
positions = np.zeros((simulation_time_steps, 2))
positions[0] = initial_position

# Initialize the array to hold the every triangle's area
triangle_areas = np.zeros((simulation_time_steps - 1,))

# Initialize the times
times = np.zeros((simulation_time_steps,))

for i in range(1, simulation_time_steps):

    #define before velocity for each step of for loop
    before_velocity = velocities[i - 1]
    before_position = positions[i - 1]
    before_time = times[i - 1]
    
    # fundamental change for 2nd-order Runge-Kutta -- first estimate mid_position!!
    mid_position = before_position + (0.5 * before_velocity * delta_t)
    
    # calculate mid_acceleration -- using the estimated mid_position!
    mid_acceleration = acceleration_of_gravity(mid_position)
   
    # calculate after_velocity from before_velocity and mid_acceleration
    after_velocity = mid_acceleration * delta_t + before_velocity 
    
    # calculate after_position using the democratic combination of after_velocity and before_velocity
    after_position = before_position + (0.5 * (before_velocity + after_velocity)) * delta_t
    
    #find area of the triangle from the before_position vector and after_positions vector
    area = triangle_area(before_position, after_position)
    triangle_areas[i - 1] = area
    
    # update time
    after_time = before_time + delta_t
    
    # assign the after values into their lists
    velocities[i] = after_velocity
    positions[i] = after_position
    times[i] = after_time


#for loop that checks for outlier negative value and replaces with subsequent value
for i in range(1,simulation_time_steps - 1):
    if triangle_areas[i] < 0:
        triangle_areas[i] = triangle_areas[i + 1]

def plot_triangle(first, second):
    #plots a line between center point and the first vector
    x1_plots = [0, x_positions[first]]
    y1_plots = [0, y_positions[first]]
    plt.plot(x1_plots, y1_plots, 'go--', linewidth=2, markersize=12, color='r')

    #plots a line between center point and the second vector
    x2_plots = [0, x_positions[second]]
    y2_plots = [0, y_positions[second]]
    plt.plot(x2_plots, y2_plots, 'go--', linewidth=2, markersize=12, color='r')

    #plots a line between the end of the first and the end of the second vector
    x3_plots = [x_positions[first], x_positions[second]]
    y3_plots = [y_positions[first], y_positions[second]]
    plt.plot(x3_plots, y3_plots, 'go--', linewidth=2, markersize=12, color='r')

#plots and formats the annotation of the area to each triangle on the graph 
def plot_area(first, second):
    plt.annotate(xy=[x_positions[first],y_positions[first]], s="   Area = " + str(int(triangle_areas[first])))

#rounds the array to a whole number and prints the array
triangle_areas = np.round_(triangle_areas, decimals = 0, out = None)
print(triangle_areas)

#create figure for eliptical orbit plot
plt.figure(figsize=(15, 15))

#plots x and y positions as a scatter plot
x_positions = positions[:, 0]
y_positions = positions[:, 1]
plt.scatter(x_positions, y_positions)

#labels the x and y axis
plt.xlabel("x position (km)")
plt.ylabel("y position (km)")

# ***** Choosing what points you want traingles from *****
point_1 = 1
point_2 = 28

#calls function to plot triangle using prevoiusly defined points
plot_triangle(point_1, point_1+1)
plot_triangle(point_2, point_2+1)

#plots the area of the same triangles drawn
plot_area(point_1, point_1+1)
plot_area(point_2, point_2+1)

# Draw a big blue circle to represent the earth
earth = plt.Circle((0, 0), radius_of_earth, color='b')
plt.gcf().gca().add_artist(earth)

# Make the plot big enough to show elliptical orbits
plot_limit = 8000
plt.xlim(-1.8 * plot_limit, 1.8 * plot_limit)
plt.ylim(-2.4 * plot_limit, 1.2 * plot_limit)

plt.show()

#create figure for eliptical orbit plot
plt.figure(figsize=(15, 15))

#plots x and y positions as a scatter plot
x_positions = positions[:, 0]
y_positions = positions[:, 1]
plt.scatter(x_positions, y_positions)

#labels the x and y axis
plt.xlabel("x position (km)")
plt.ylabel("y position (km)")

# Loop that prints every triangle and its area 
for i in range(0, len(triangle_areas)):
    plot_triangle(i, i+1)
    plot_area(i,i+1)

# Plots the last triangle in the series (from the last point back to the first)
plot_triangle(len(triangle_areas), 0)

# Draw a big blue circle to represent the earth
earth = plt.Circle((0, 0), radius_of_earth, color='b')
plt.gcf().gca().add_artist(earth)

# Make the plot big enough to show elliptical orbits
plot_limit = 8000
plt.xlim(-1.8 * plot_limit, 1.8 * plot_limit)
plt.ylim(-2.4 * plot_limit, 1.2 * plot_limit)

plt.show()

#Creates an array to represent our simulation timesteps for plotting purposes
x_axis = np.arange(0, simulation_time_steps -1)
#Sets Y-Axis as our array of triangle areas
y_axis = triangle_areas

plt.ylabel('Area (km^2)')
plt.xlabel('Timesteps')
plt.title('Areas by times step')
plt.grid()
plt.scatter(x_axis, y_axis)

#Creates and plots a straight red line at the most accurate area to show error below it 
x_line_plots = [0, simulation_time_steps - 1]
y_line_plots = [triangle_areas[simulation_time_steps//2],triangle_areas[simulation_time_steps//2]]
plt.plot(x_line_plots, y_line_plots, 'go--', linewidth=2, markersize=1, color='r')
plt.show()

#Creates an array to represent our simulation timesteps for plotting purposes
graph_time_steps = np.arange(0, simulation_time_steps-1)

plt.bar(graph_time_steps, triangle_areas)
plt.ylabel('Area (km^2)')
plt.xlabel('Timesteps')
plt.title('Areas by times step')
plt.show()
