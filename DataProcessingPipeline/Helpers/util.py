import math
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion

class CacheROSMessage:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = []
        self.time = []

    def add_element(self, data):
        if len(self.cache) >= self.max_size:
            self.time.pop(0)
            self.cache.pop(0)

        self.cache.append(data)
        self.time.append(data.header.stamp.secs)
    
    def get_all(self):
        return self.cache
    
    def clear_cache(self):
        self.cache = []
        self.time = []
    
    def get_index(self, index):
        return self.cache[index]

    def get_element_from_time(self, time):
        if time not in self.time:
            return None

        index = self.time.index(time)
        return self.cache[index]
       
    def get_oldest_element(self):
        return self.cache[0]

class CacheHeaderlessROSMessage:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = []

    #add element in the cache
    def add_element(self, data):
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)

        self.cache.append(data)

    #get all the elements in the cache    
    def get_all(self):
        return self.cache
    
    #clear the cache
    def clear_cache(self):
        self.cache = []
        self.time = []
    
    #get element at a particular index
    def get_index(self, index):
        return self.cache[index]

    #get the oldest element in the cache       
    def get_oldest_element(self):
        return self.cache[0]

    def get_last_element(self):
        if len(self.cache) == 0:
            return None
        
        return self.cache[-1]

    #get the latest n elements in the cache, if n > cache size, return None
    def get_last_n_elements(self, n):
        if n > len(self.cache):
            return self.get_all()

        return self.cache[-n:]
    
    def get_length(self):
        return len(self.cache)

class ImuConversions:
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