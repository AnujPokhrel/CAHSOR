import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
import rospy
from ublox_msgs.msg import NavPVT
import Helpers.utils as utils
import DataProcessingPipeline.Helpers.image_processing as image_processing

class FindCurrentPatch:
    def __init__(self):
        self.odom_cached = utils.CacheROSMessage(200)
        self.image_cached = utils.CacheHeaderlessROSMessage(200)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/rgb_publisher/color/image/compressed', CompressedImage, self.image_callback)
        rospy.Subscriber('/f9p_rover/navpvt', NavPVT, self.cmd_vel_callback)
        self.latest_odom = None
        self.patch_underneath = None
        self.odom_initialized = 0
        self.camera_initialized = 0

    def odom_callback(self, msg):
        if self.odom_initialized == 0:
            self.odom_initialized = 1
            print("odom initialized")

        self.latest_odom = msg

    def image_callback(self, msg):
        if self.camera_initialized == 0:
            self.camera_initialized= 1
            print("camera initialized")

        self.odom_cached.add_element(self.latest_odom) 

        bev_image = image_processing.bev_image(msg)
        self.image_cached.add_element(bev_image) 
 
    #sync image patch extraction to the msg from the gps
    def cmd_vel_callback(self, msg):
        last_15_odom_msgs = self.odom_cached.get_last_n_elements(15)
        last_15_image_msgs = self.image_cached.get_last_n_elements(15)

        patch = None
        patch_black_count = []
        patches = []
        if len(last_15_image_msgs) > 0 and len(last_15_odom_msgs) == len(last_15_image_msgs):
            for j in range(0, len(last_15_odom_msgs)):
                prev_image = last_15_image_msgs[j]
                current_image = last_15_image_msgs[-1]
                prev_odom = last_15_odom_msgs[j]
                current_odom = last_15_odom_msgs[-1]
                patch, patch_black_pct, curr_img, vis_img = image_processing.get_patch_from_odom_delta(
                    current_odom.pose.pose, prev_odom.pose.pose, current_odom.twist.twist, prev_odom.twist.twist,
                    prev_image, current_image)
                if patch is not None:
                    self.patch_underneath = patch 
                    patches.append(patch)
                    patch_black_count.append(np.sum(patch == 0))
             
            self.patch_underneath = patches[np.argmin(patch_black_count)]

            if patch is None:
                self.patch_underneath = last_15_image_msgs[-1] 
        