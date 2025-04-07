from deepface import DeepFace

result = DeepFace.verify("image1","image2" , model_name="Facenet")
print(result)
