
from mtcnn.mtcnn import detect_face, create_mtcnn
import tensorflow as tf
from skimage import transform as trans
import math
from jade import *
face_crop_margin = 32
face_crop_size = 160
PRETREINED_MODEL_DIR = 'model/mtcnn_model'

def _setup_mtcnn():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        sess = tf.Session(config=config)
        with sess.as_default():
            return create_mtcnn(sess, PRETREINED_MODEL_DIR)


pnet, rnet, onet = _setup_mtcnn()


def img_to_np(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def alignment(img,bb,landmark,image_size):
  M = None
  # if len(image_size)>0:
  #   image_size = [int(x) for x in image_size.split(',')]
  #   if len(image_size)==1:
  #      image_size = [image_size[0], image_size[0]]

  if landmark is not None:
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
      src[:,0] += 8.0
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

  if M is None:
     ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
     if len(image_size)>0:
        ret = cv2.resize(ret, (image_size[1], image_size[0]))
     return ret
  else:
     warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
     return warped

def calculate_euler(img, landmark):
    model_points = np.array([
        [-165.0, 170.0, -115.0],  # Left eye left corner
        [165.0, 170.0, -115.0],  # Right eye right corne
        [0.0, 0.0, 0.0],  # Nose tip
        [-150.0, -150.0, -125.0],  # Left Mouth corner
        [150.0, -150.0, -125.0]], dtype=np.float32)  # Right mouth corner

    focal_length = img.shape[1]
    center = (img.shape[1] / 2, img.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float32
    )

    dst = landmark.astype(np.float32)
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, dst, camera_matrix, dist_coeffs)

    rotation3_3 = cv2.Rodrigues(rotation_vector)[0]

    q0 = np.sqrt(1 + rotation3_3[0][0] + rotation3_3[1][1] + rotation3_3[2][2]) / 2
    q1 = (rotation3_3[2][1] - rotation3_3[1][2]) / (4 * q0)
    q2 = (rotation3_3[0][2] - rotation3_3[2][0]) / (4 * q0)
    q3 = (rotation3_3[1][0] - rotation3_3[0][1]) / (4 * q0)

    yaw = math.asin(2 * (q0 * q2 + q1 * q3)) * (180 / math.pi)
    pitch = math.atan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * (180 / math.pi)
    # roll = math.atan2(2*(q0*q3-q1*q2), q0*q0+q1*q1-q2*q2-q3*q3)*(180/math.pi)
    euler = [yaw, pitch]

    return euler

def detect(image, threshold=0.5, minsize=30):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    # img = img_to_np(image)
    # face detection parameters
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    temp =None
    temp_area = 0.0
    bounding_boxes, points = detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

    idx = 0
    bboxes = []
    label_texts = []
    labels = []
    scores = []
    img = np.array([])
    for i in range(len(bounding_boxes)):
        landmark = points[:, i].reshape((2, 5)).T
        bb = bounding_boxes[i]
        area = (float(bb[2]) - float(bb[0])) * (float(bb[3] - float(bb[1])))
        if area > temp_area:
            temp_area = area
            idx = i
            img = alignment(image, bounding_boxes[i][0:4], landmark, (112, 112))
            if face_crop_size != 112:
                img = cv2.resize(img, (112, 112))
    if len(bounding_boxes) > 0:
        bboxes.append([bounding_boxes[idx][0], bounding_boxes[idx][1], bounding_boxes[idx][2], bounding_boxes[idx][3]])
        label_texts.append("face")
        labels.append(1)
        scores.append(bounding_boxes[idx][4])
    return DetectResultModel(bboxes, label_texts, labels, scores),img



if __name__ == '__main__':
    img = cv2.imread("examples/face.jpg")
    bboxes, label_tests, labels, scores = detect(img)
    CVShowBoxes(img,bboxes,label_tests,labels,scores,waitkey=0)