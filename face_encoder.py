import face_recognition
import os
import cv2
import numpy as np

def encode_faces(image_dir='images', save_path='details/face_encodings.npy'):
    known_encodings = []
    known_names = []

    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])

    np.save(save_path, {'encodings': known_encodings, 'names': known_names})
    print(f"[INFO] Encoded {len(known_names)} faces.")

if __name__ == "__main__":
    encode_faces()
