from deepface import DeepFace

result = DeepFace.verify("D:\\Github repoS\\Face_Verification\\Face_Verification\data\\k1.jpg","D:\\Github repoS\\Face_Verification\\Face_Verification\data\\k2.jpg" , model_name="Facenet")
print(result)
