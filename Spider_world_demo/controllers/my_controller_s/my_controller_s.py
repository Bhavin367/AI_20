import math
from controller import Robot

# =========================================================
# INITIALISE ROBOT
# =========================================================

robot = Robot()

timestep = int(robot.getBasicTimeStep())


# =========================================================
# LEGS
# =========================================================

right_legs = ["R1", "R2", "R3", "R4"]
left_legs  = ["L1", "L2", "L3", "L4"]

all_legs = right_legs + left_legs


# =========================================================
# TARGET STANDING POSITION
# =========================================================

TARGET_ANGLES = {

    "R": {
        "hip_joint": 0.0,
        "thigh_joint": 1.07,
        "knee_joint": -0.58
    },

    "L": {
        "hip_joint": 0.0,
        "thigh_joint": -1.07,
        "knee_joint": 0.58
    }
}


# =========================================================
# GPS
# =========================================================

gps = robot.getDevice("gps")

if gps is None:
    print("ERROR: GPS not found!")
    exit()

gps.enable(timestep)


# =========================================================
# MOTOR SETTINGS
# =========================================================

MOTOR_VELOCITY = 1.5

# Small movement made each cycle.
#
# 0.5 degrees is deliberately small so that we can
# observe the effect on the actual robot height.
#
LOWER_STEP = math.radians(0.5)


# =========================================================
# GET MOTORS
# =========================================================

motors = {}

for side, leg_list in [
    ("R", right_legs),
    ("L", left_legs)
]:

    for leg in leg_list:

        for joint_type in [
            "hip_joint",
            "thigh_joint",
            "knee_joint"
        ]:

            motor_name = f"{leg}_{joint_type}"

            motor = robot.getDevice(motor_name)

            if motor:

                motors[motor_name] = motor
                motor.setVelocity(MOTOR_VELOCITY)

            else:

                print(
                    f"WARNING: Motor '{motor_name}' not found!"
                )


# =========================================================
# SET INITIAL STANDING POSITION
# =========================================================

print("")
print("==============================================")
print("SETTING INITIAL STANDING POSITION")
print("==============================================")


for side, leg_list in [
    ("R", right_legs),
    ("L", left_legs)
]:

    angles = TARGET_ANGLES[side]

    for leg in leg_list:

        motors[
            f"{leg}_hip_joint"
        ].setPosition(
            angles["hip_joint"]
        )

        motors[
            f"{leg}_thigh_joint"
        ].setPosition(
            angles["thigh_joint"]
        )

        motors[
            f"{leg}_knee_joint"
        ].setPosition(
            angles["knee_joint"]
        )


# =========================================================
# WAIT FOR STANDING POSITION
# =========================================================

print("Waiting for standing position...")

for _ in range(
    int(3.0 * 1000 / timestep)
):

    if robot.step(timestep) == -1:
        exit()


# =========================================================
# READ INITIAL GPS HEIGHT
# =========================================================

position = gps.getValues()

current_z = position[2]

print("")
print("==============================================")
print("INITIAL GPS POSITION")
print("==============================================")
print(
    f"X = {position[0]:.4f} m"
)
print(
    f"Y = {position[1]:.4f} m"
)
print(
    f"Z = {position[2]:.4f} m"
)

print("")
print(
    f"Target Z = 0.3000 m"
)
print(
    f"Current Z = {current_z:.4f} m"
)

# =========================================================
# CURRENT JOINT TARGETS
# =========================================================

right_thigh = TARGET_ANGLES["R"]["thigh_joint"]
right_knee = TARGET_ANGLES["R"]["knee_joint"]

left_thigh = TARGET_ANGLES["L"]["thigh_joint"]
left_knee = TARGET_ANGLES["L"]["knee_joint"]


# =========================================================
# LOWER ROBOT
# =========================================================

print("")
print("==============================================")
print("LOWERING ROBOT")
print("==============================================")
print("")


TARGET_Z = 0.300


while robot.step(timestep) != -1:

    # -----------------------------------------------------
    # READ GPS
    # -----------------------------------------------------

    position = gps.getValues()

    current_z = position[2]


    # -----------------------------------------------------
    # CHECK WHETHER TARGET HEIGHT HAS BEEN REACHED
    # -----------------------------------------------------

    if current_z >= TARGET_Z:

        print("")
        print("==============================================")
        print("TARGET HEIGHT REACHED")
        print("==============================================")

        print(
            f"GPS Z = {current_z:.4f} m"
        )

        break


    # -----------------------------------------------------
    # LOWER THE LEGS
    # -----------------------------------------------------
    #
    # Right side:
    #   thigh becomes more positive
    #   knee becomes more negative
    #
    # Left side:
    #   thigh becomes more negative
    #   knee becomes more positive
    #
    # This increases the absolute movement symmetrically.
    #
    # -----------------------------------------------------

    right_thigh += LOWER_STEP
    right_knee -= LOWER_STEP

    left_thigh -= LOWER_STEP
    left_knee += LOWER_STEP


    # -----------------------------------------------------
    # COMMAND RIGHT LEGS
    # -----------------------------------------------------

    for leg in right_legs:

        motors[
            f"{leg}_thigh_joint"
        ].setPosition(
            right_thigh
        )

        motors[
            f"{leg}_knee_joint"
        ].setPosition(
            right_knee
        )


    # -----------------------------------------------------
    # COMMAND LEFT LEGS
    # -----------------------------------------------------

    for leg in left_legs:

        motors[
            f"{leg}_thigh_joint"
        ].setPosition(
            left_thigh
        )

        motors[
            f"{leg}_knee_joint"
        ].setPosition(
            left_knee
        )


    # -----------------------------------------------------
    # REPORT POSITION
    # -----------------------------------------------------

    print(
        f"Z = {current_z:.4f} m | "
        f"R thigh = {math.degrees(right_thigh):.2f}° | "
        f"R knee = {math.degrees(right_knee):.2f}° | "
        f"L thigh = {math.degrees(left_thigh):.2f}° | "
        f"L knee = {math.degrees(left_knee):.2f}°"
    )


# =========================================================
# HOLD POSITION
# =========================================================

print("")
print("Robot is holding its position.")

while robot.step(timestep) != -1:

    pass
