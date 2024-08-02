import numpy as np
import math
from geometry_msgs.msg import PoseStamped, Quaternion
import rospy
import utm

#Class for general functions
class utils:
    def __init__(self):
        self.queue_size = 0

    def rmap(self, value, from_min, from_max, to_min, to_max):
        # Calculate the range of the input value
        from_range = from_max - from_min

        # Calculate the range of the output value
        to_range = to_max - to_min

        # Scale the input value to the output range
        mapped_value = (value - from_min) * (to_range / from_range) + to_min


        return mapped_value

    # Convert quaternion to yaw angle (in radians)
    def quaternion_to_yaw(self, quaternion):
        quaternion_norm = math.sqrt(quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2)
        if (quaternion_norm == 0):
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        yaw = math.atan2(2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
                         1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2))

        return yaw
    
    # Convert quaternion to roll angle (in radians)
    def quaternion_to_roll(self, quaternion):
        quaternion_norm = math.sqrt(quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2)
        if quaternion_norm == 0:
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        roll = math.atan2(2.0 * (quaternion.y * quaternion.z + quaternion.w * quaternion.x),
                        1.0 - 2.0 * (quaternion.x**2 + quaternion.y**2))

        return roll

    # Convert quaternion to pitch angle (in radians)
    def quaternion_to_pitch(self, quaternion):
        quaternion_norm = math.sqrt(quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2)
        if quaternion_norm == 0:
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        pitch = math.asin(2.0 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x))

        return pitch

    # Convert yaw angle (in radians) to quaternion
    def yaw_to_quaternion(self, yaw):
        quaternion = Quaternion()
        quaternion.x = 0.0
        quaternion.y = 0.0
        quaternion.z = math.sin(yaw / 2.0)
        quaternion.w = math.cos(yaw / 2.0)

        return quaternion
    
    # Convert quaternion to roll, pitch, and yaw angles
    def quaternion_to_rpy(self, quaternion):
        qw = quaternion.w
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    # Convert roll, pitch, and yaw angles to quaternion
    def rpy_to_quaternion(self, roll, pitch, yaw):
        quaternion = Quaternion()
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        quaternion.x = sr * cp * cy - cr * sp * sy
        quaternion.y = cr * sp * cy + sr * cp * sy
        quaternion.z = cr * cp * sy - sr * sp * cy
        quaternion.w = cr * cp * cy + sr * sp * sy

        return quaternion

    def get_dist(self, start_pose, goal_pose):
        return math.sqrt((goal_pose.position.x - start_pose.position.x)**2 + (goal_pose.position.y - start_pose.position.y)**2)

    # Create a PoseStamped message from a Pose message
    def create_pose_stamped(self, pose):
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = 'odom'  # Replace 'world' with your desired frame ID

        return pose_stamped

    def map_value(self, value, from_min, from_max, to_min, to_max):
        # Calculate the range of the input value
        from_range = from_max - from_min

        # Calculate the range of the output value
        to_range = to_max - to_min

        # Scale the input value to the output range
        mapped_value = (value - from_min) * (to_range / from_range) + to_min

        return mapped_value
    
    #convert lat and long msg from gps to utm co-ordinates
    def to_utm(self, lat, long):        
        return list(utm.from_latlon(lat, long))
    
    #convert the gps anges which start from due north to start from due east 
    def convert_gps_to_xy_angles(self, angle_np):
        angle_np = 360 - angle_np
        angle_np = angle_np + 90
        angle_np[angle_np > 360] = angle_np[angle_np > 360] - 360
        angle_np[angle_np < 0] = angle_np[angle_np < 0] + 360
        return np.radians(angle_np)

    #transformation to the robot frame using numpy SE2
    def non_batch_2d_transform_to_robot(self, p1, p2):
        tf = np.zeros(3)
        T1 = np.array([[np.cos(p1[2]), -np.sin(p1[2]), p1[0]],
                    [np.sin(p1[2]),  np.cos(p1[2]), p1[1]],
                    [0,               0,               1]])

        T2 = np.array([[np.cos(p2[2]), -np.sin(p2[2]), p2[0]],
                    [np.sin(p2[2]),  np.cos(p2[2]), p2[1]],
                    [0,               0,               1]])

        T1_inv = np.linalg.inv(T1)
        # print("x and y coordinates: ", T1.dot(T2.dot(np.array([0, 0, 1])))[:2] )
        tf2_mat = T2 @ T1_inv

        tf[:2] = T1_inv.dot(T2.dot(np.array([0, 0, 1])))[:2]
        tf[2] = np.arctan2(tf2_mat[1,0], tf2_mat[0,0])

        # print("theta: ", np.arctan2(tf2_mat[1,0], tf2_mat[0,0]))
        return tf
    