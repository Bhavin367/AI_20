import numpy as np

# constants from coursework 
n_legs = 8  
joints_per_leg = 3 
segment_length = [1.2,0.7,1.0] # length of coxa, femur and timia or sm like that  
coxa_elevation = np.deg2rad(30)  
A,B = 1.5,1.0  # A  = half length , B = half width 

leg_names = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
joint_names = ["coxa","femur","tibia"]

# base mounting points, legs direction would be base angle + evolved angle 
left_leg_angles = np.deg2rad([45, 75, 105, 135])
right_leg_angles = np.deg2rad([-135, -105, -75, -45])
base_angles = [left_leg_angles, right_leg_angles] # yep its copy pasted straight from given file 

 #          90° (left side)
 #   135°  L4   L3 L2  L1  45°
 # 180° ---------------------- 0°  (head)
 #  -135°  R4   R3 R2  R1  -45°
 #          -90° (right side)
 #

# Design choices we gonna make 

# [45.83662361 68.75493542 85.94366927]
low = np.tile([-0.8,-1.2,-1.5],n_legs) 
high = np.tile([0.8,1.2,1.5],n_legs) 

# giving coxa shorter range of motion and timbia higher since its connected to feet


ground_z = -0.7
# assuming body of the spider is at z = 0 ground must be somewhere around below that
# and total length of the leg is 2.9

# gait
n_frames = 4          # frames in one gait cycle (must be even)
lift_height = 0.2     # swinging feet must be at least this far above the ground
max_lift = 0.5        # ...but no higher than this (no waving legs over the body)
stride = 0.5          # each foot must push back at least this far per cycle


# weights 

w_limit , w_sym , w_ground = 1.0,1.0,2.0 # staying on ground is more imp 
w_lift , w_stride , w_slip = 1.0,1.0,1.0



# parameters 

population_size = 100  
n_genes = n_frames * n_legs * joints_per_leg  # 4 * 8 * 3 = 96
generations = 200 
