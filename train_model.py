import face_recognition
import numpy as np
import os
import pickle

PATH_TO_FACES = 'faces'
ENCODINGS_FILE = 'face_encodings.pkl'

def train_and_save_encodings():
    print("Starting face encoding process...")
    known_face_encodings = []
    known_face_names = []

    for name_folder in os.listdir(PATH_TO_FACES):
        name_path = os.path.join(PATH_TO_FACES, name_folder)
        if not os.path.isdir(name_path) or name_folder.startswith('.'): continue

        print(f"Processing folder: {name_folder}...")

        for image_file in os.listdir(name_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(name_path, image_file)
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(name_folder)
                except Exception as e:
                    print(f"   ERROR processing {image_file}: {e}")

    print("\nEncoding complete. Saving data...")
    data = {"encodings": known_face_encodings, "names": known_face_names}
    
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
        
    print(f"Successfully saved {len(known_face_encodings)} face encodings to {ENCODINGS_FILE}")
    print("Training complete!")

if __name__ == "__main__":
    train_and_save_encodings()
