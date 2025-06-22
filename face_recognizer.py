import face_recognition
import numpy as np

class FaceRecognizer:
    def __init__(self, encoding_path='details/face_encodings.npy'):
        data = np.load(encoding_path, allow_pickle=True).item()
        self.known_encodings = data['encodings']
        self.known_names = data['names']

    def recognize(self, frame):
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, encoding)
            name = "Unknown"
            if True in matches:
                matched_idx = matches.index(True)
                name = self.known_names[matched_idx]
            names.append(name)
        return face_locations, names
