import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp

# ========== FACE VERIFICATION ========== #
def verify_faces(img1_path, img2_path):
    result = DeepFace.verify("D:\\Github repoS\\Face_Verification\\Face_Verification\data\\k1.jpg", "D:\\Github repoS\\Face_Verification\\Face_Verification\data\\k1.jpg", model_name='DeepFace')
    verified = result["verified"]
    distance = result["distance"]
    return verified, distance


# ========== HEAD POSE ESTIMATION ========== #
def estimate_head_pose(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    height, width = image.shape[:2]

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return "No face detected", image

    face_landmarks = results.multi_face_landmarks[0]

    # Use 6 key face landmarks for head pose estimation
    landmark_ids = [1, 33, 61, 199, 263, 291]  # Nose tip, eye corners, etc.
    image_points = []
    for idx in landmark_ids:
        x = int(face_landmarks.landmark[idx].x * width)
        y = int(face_landmarks.landmark[idx].y * height)
        image_points.append((x, y))
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # 3D model points for head pose estimation (approx)
    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (-30.0, -30.0, -30.0),  # Left eye corner
        (-30.0, 30.0, -30.0),   # Left mouth corner
        (0.0, 63.6, -12.5),     # Chin
        (30.0, -30.0, -30.0),   # Right eye corner
        (30.0, 30.0, -30.0),    # Right mouth corner
    ], dtype='double')

    image_points = np.array(image_points, dtype='double')

    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return "Pose estimation failed", image

    # Interpret orientation from rotation vector
    rmat, _ = cv2.Rodrigues(rotation_vector)
    pose = "Unknown"
    if rmat[2, 0] < -0.3:
        pose = "Looking Right"
    elif rmat[2, 0] > 0.3:
        pose = "Looking Left"
    elif rmat[1, 2] < -0.3:
        pose = "Looking Down"
    elif rmat[1, 2] > 0.3:
        pose = "Looking Up"
    else:
        pose = "Looking Straight"

    return pose, image
