import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

# Step 1: Face verification
def verify_faces(img1_path, img2_path, model_name="Facenet"):
    result = DeepFace.verify(img1_path, img2_path, model_name=model_name)
    print("Face Verification Result:", result['verified'])
    return result

# Step 2: Pose estimation using solvePnP
def get_head_pose(img_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("No face detected")
        return
    
    # Define 3D model points of the face
    face_3d_model_points = np.array([
        [0.0, 0.0, 0.0],         # Nose tip
        [0.0, -330.0, -65.0],    # Chin
        [-225.0, 170.0, -135.0], # Left eye left corner
        [225.0, 170.0, -135.0],  # Right eye right corner
        [-150.0, -150.0, -125.0],# Left mouth corner
        [150.0, -150.0, -125.0]  # Right mouth corner
    ])

    for face_landmarks in results.multi_face_landmarks:
        # Get 2D landmark points
        image_points = np.array([
            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),     # Nose tip
            (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # Chin
            (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), # Right eye right corner
            (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),   # Left eye left corner
            (face_landmarks.landmark[287].x * w, face_landmarks.landmark[287].y * h), # Right mouth corner
            (face_landmarks.landmark[57].x * w, face_landmarks.landmark[57].y * h)    # Left mouth corner
        ], dtype="double")

        # Camera internals
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4,1))  # No lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(
            face_3d_model_points, image_points, camera_matrix, dist_coeffs)

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        yaw, pitch, roll = euler_angles.flatten()
        print(f"Head Pose (Yaw, Pitch, Roll): {yaw:.2f}, {pitch:.2f}, {roll:.2f}")

        # Annotate image
        cv2.putText(img, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(img, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(img, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Head Pose Estimation", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ==== Example usage ====
img1 = "image1"
img2 = "image2"

# Step 1: Face verification
verify_faces(img1, img2)

# Step 2: Head pose estimation for both
get_head_pose(img1)
get_head_pose(img2)
