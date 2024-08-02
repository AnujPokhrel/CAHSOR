import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

PATCH_SIZE = 256
PATCH_EPSILON = 0.7 * PATCH_SIZE * PATCH_SIZE
ACTUATION_LATENCY = 0.15
HOMOGRAPHY_MATRIX = np.array([[ 2.10928225e-01, -1.13894127e+00,  4.44216119e+02],
       [-1.26548983e-16, -1.84371290e+00,  1.29715449e+03],
       [-9.85515129e-20, -1.76433885e-03,  1.00000000e+00]]) 

#process all the bev images and patches from the data, also find the current patch underneath the robot
def process_bev_image_and_patches(msg_data):
    processed_data = {'image':[]}
    msg_data['patches'] = {}
    msg_data['patches_found'] = {}
    msg_data['patches_path'] = []
    found_patch_number = 0

    for i in range(len(msg_data['image_msg'])):
        show_img = False 
        if msg_data['image_msg'][i] is None:
            processed_data['image'].append(None)
            msg_data['patches'][i] = None
            msg_data['patches_found'][i] = False
            continue
        bevimage = bev_image(msg_data['image_msg'][i])
        processed_data['image'].append(bevimage)
        curr_odom = msg_data['odom_msg'][i]

        found_patch = False

        patches_black_count = []
        for j in range(i, max(i-30, 0), -2):
            prev_image = processed_data['image'][j]
            if prev_image is None: continue
            prev_odom = msg_data['odom_msg'][j]

            patch, patch_black_pct, curr_img, vis_img = get_patch_from_odom_delta(
                curr_odom.pose.pose, prev_odom.pose.pose, curr_odom.twist.twist,
                prev_odom.twist.twist, prev_image, processed_data['image'][i])
            if patch is not None:
                found_patch = True
                if i not in msg_data['patches']:
                    msg_data['patches'][i] = []
                msg_data['patches'][i].append(patch)
                patches_black_count.append(np.sum(patch == 0))
            show_img = False
          
        if not found_patch:
            print("Unable to find patch for idx: ", i)
            msg_data['patches'][i] = [processed_data['image'][i][0:256, 0:256]]
        else:
            found_patch_number += 1
            msg_data['patches'][i] = [msg_data['patches'][i][np.argmin(patches_black_count)]]
        if i > 30:
            processed_data['image'][i-30] = None

        # was the patch found or no ?
        if found_patch: msg_data['patches_found'][i] = True
        else: msg_data['patches_found'][i] = False

    print(f"no of patches found is {found_patch_number}")
    return msg_data['patches'], msg_data['patches_found']

#being used the function the process_bev_image_and_patches
def get_patch_from_odom_delta(curr_pos, prev_pos, curr_vel, prev_vel, prev_image, curr_image):
    curr_pos_np = np.array([curr_pos.position.x, curr_pos.position.y, 1])
    prev_pos_transform = np.zeros((3, 3))
    z_angle = R.from_quat([prev_pos.orientation.x, prev_pos.orientation.y, prev_pos.orientation.z, prev_pos.orientation.w]).as_euler('xyz', degrees=False)[2]
    prev_pos_transform[:2, :2] = R.from_euler('xyz', [0, 0, z_angle]).as_matrix()[:2,:2] # figure this out
    prev_pos_transform[:, 2] = np.array([prev_pos.position.x, prev_pos.position.y, 1]).reshape((3))

    inv_pos_transform = np.linalg.inv(prev_pos_transform)
    curr_z_angle = R.from_quat([curr_pos.orientation.x, curr_pos.orientation.y, curr_pos.orientation.z, curr_pos.orientation.w]).as_euler('xyz', degrees=False)[2]
    curr_z_rotation = R.from_euler('xyz', [0, 0, curr_z_angle]).as_matrix()
    projected_loc_np  = curr_pos_np + ACTUATION_LATENCY * (curr_z_rotation @ np.array([curr_vel.linear.x, curr_vel.linear.y, 0]))

    patch_corners = [
        projected_loc_np + curr_z_rotation @ np.array([0.5, 0.33, 0]),
        projected_loc_np + curr_z_rotation @ np.array([0.5, -0.33, 0]),
        projected_loc_np + curr_z_rotation @ np.array([-0.5, -0.33, 0]),
        projected_loc_np + curr_z_rotation @ np.array([-0.5, 0.33, 0])
    ]
    patch_corners_prev_frame = [
        inv_pos_transform @ patch_corners[0],
        inv_pos_transform @ patch_corners[1],
        inv_pos_transform @ patch_corners[2],
        inv_pos_transform @ patch_corners[3],
    ]
    scaled_patch_corners = [
        (patch_corners_prev_frame[0] * 306).astype(np.int64),
        (patch_corners_prev_frame[1] * 306).astype(np.int64),
        (patch_corners_prev_frame[2] * 306).astype(np.int64),
        (patch_corners_prev_frame[3] * 306).astype(np.int64),
    ]
  
    CENTER = np.array((640, 670))
    patch_corners_image_frame = [
        CENTER + np.array((-scaled_patch_corners[0][1], -scaled_patch_corners[0][0])),
        CENTER + np.array((-scaled_patch_corners[1][1], -scaled_patch_corners[1][0])),
        CENTER + np.array((-scaled_patch_corners[2][1], -scaled_patch_corners[2][0])),
        CENTER + np.array((-scaled_patch_corners[3][1], -scaled_patch_corners[3][0]))
    ]
    vis_img = prev_image.copy()
    
    persp = cv2.getPerspectiveTransform(np.float32(patch_corners_image_frame), np.float32([[0, 0], [255, 0], [255, 255], [0, 255]]))

    patch = cv2.warpPerspective(prev_image, persp, (256, 256))
    zero_count = np.logical_and(np.logical_and(patch[:, :, 0] == 0, patch[:, :, 1] == 0), patch[:, :, 2] == 0)

    if np.sum(zero_count) > PATCH_EPSILON:
        return None, 1.0, None, None

    return patch, (np.sum(zero_count) / (256. * 256.)), curr_image, vis_img

def process_odom_vel_data(data):
    odoms = []
    for i in range(len(data['odom_msg'])-1):
        odom_now = data['odom_msg'][i]
        odom_now = np.array([odom_now.twist.twist.linear.x, odom_now.twist.twist.linear.y, odom_now.twist.twist.angular.z])
        if i>len(data['odom_msg'])-6:
            odom_next = data['odom_msg'][i+1]
        else:
            odom_next = data['odom_msg'][i+5] # assuming a delay of 0.2s
        odom_next = np.array([odom_next.twist.twist.linear.x, odom_next.twist.twist.linear.y, odom_next.twist.twist.angular.z])
        odoms.append(np.hstack((odom_now, odom_next)))
    return odoms

#create the bev image
def bev_image(image):
    img = np.fromstring(image.data, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    
    output = cv2.warpPerspective(img, HOMOGRAPHY_MATRIX, (1280, 720))
    output = cv2.flip(output, 1)
    
    return output 
