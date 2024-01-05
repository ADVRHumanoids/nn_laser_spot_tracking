### For tpo_vision: a keypoint on the image

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined
                     # seq used for image id?

uint16 x_pixel
uint16 y_pixel

uint8 label 

float32 confidence # This depend on the method used



