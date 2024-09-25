#!/usr/bin/env python3

"""
Software License Agreement (BSD 3-Clause License)

Copyright (c) 2024, Simon Janzon
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import rospy
import time
import numpy as np
import torch
from cv_bridge import CvBridge
from pathlib import Path
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import mseg.utils.names_utils as names_utils

from inference_task import InferenceTask
from inference_task import Arguments

# TODO: use parameters to decide if overlaid image is published as well


class MsegNode():
    
    def __init__(self) -> None:
        rospy.init_node('ros_mseg', anonymous=True)
        
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback, queue_size=1)

        self.segmented_image_pub = rospy.Publisher('/scene_graph/mseg_segmented_image', Image, queue_size=10)
        self.overlaid_image_pub = rospy.Publisher('/scene_graph/overlaid_image', Image, queue_size=10)
        
        self.inference_task = self.create_inference_task()
        self.cv_bridge = CvBridge()
        
        
    def rgb_callback(self, msg):
        
        if self.inference_task is None:
            return
        
        start_time = time.time()
        
        segmented_image, overlaid_image = self.inference_task.execute(self.cv_bridge.imgmsg_to_cv2(msg))
        
        # self.inference_task = self.create_inference_task()
        
        segmented_image = self.create_message(segmented_image)
        
        torch.cuda.empty_cache()
        
        # DEBUG
        end_time = time.time()
        print(end_time - start_time)
        
        
        self.segmented_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(segmented_image, 'bgr8'))
        self.overlaid_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(overlaid_image, 'bgr8'))
    
    
    def create_inference_task(self) -> InferenceTask:
        args = Arguments(
            arch = 'hrnet',
            base_size = 500,
            batch_size_val = 1,
            dataset = 'test_image_small',
            has_prediction = False,
            ignore_label = 255,
            img_name_unique = False,
            index_start = 0,
            index_step = 0,
            input_file = '',
            layers = 50,
            model_name = 'mseg-3m',
            model_path = '/home/nes/mseg-semantic/mseg-3m.pth',
            network_name = None,
            save_folder = 'default',
            scales = [1.0],
            small = True,
            split ='val',
            test_gpu = [0],
            test_h = 713,
            test_w = 713,
            version = 4.0,
            vis_freq = 20,
            workers = 16,
            zoom_factor = 8,
            u_classes = names_utils.get_universal_class_names()
        )
        
        itask = InferenceTask(
            args,
            base_size=args.base_size,
            crop_h=args.test_h,
            crop_w=args.test_w,
            input_file=args.input_file,
            model_taxonomy="universal",
            eval_taxonomy="universal",
            scales=args.scales,
        )
        return itask


    def create_message(self, segmented_image):  
        img = np.zeros((len(segmented_image), len(segmented_image[0]), 3), np.uint8)
        segmented_image_msg = []
        
        for i in range(len(segmented_image)):
            for j in range(len(segmented_image[i])):
                img[i, j] = segmented_image[i][j]               
                    
        return img


if __name__ == '__main__':
    try:
        MsegNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
