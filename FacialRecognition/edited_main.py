import face_recognition
import cv2
import numpy as np

# referencing to the webcam #0 (which is the default webcamera)
capture_camera = cv2.VideoCapture(0)

# loading images
kerney_image = face_recognition.load_image_file("kerney_image.png")
kerney_image_encode = face_recognition.face_encodings(kerney_image)[0]

# Creating list of the known faces of loaded images.
known_faces_codes = [
    kerney_image_encode
]

face_labels = [
    "Kerney"
]

# initialize global variables
face_locations = []
face_encodings = []
names = []
processing_frame = True

while True:
    # Grabbing each single frame captured from the video
    returning, frame = capture_camera.read()

    # resizing frame of video to 1/4 for faster face recognition processing
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

    # converting images from BGR color (which OpenCV uses) to RGB color (Which this program shall use!)
    rgb_frames = small_frame[:, :, ::-1]

    # Processing every other frame of the video to save time.
    if processing_frame:
        # Searching for all facees and its encodes in the videos current frame.
        face_locations = face_recognition.face_locations(rgb_frames);
        face_encodings = face_recognition.face_encodings(rgb_frames, face_locations);

        names = []

        for encoding in known_faces_codes:
            # Checking if the face matches
            matched_face = face_recognition.compare_faces(known_faces_codes, face_encodings)
            name = "Unknown"

            # If the face does match
            face_distance = face_recognition.face_distance(known_faces_codes, face_encodings)
            best_match = np.argmin(face_distances)

            if matched_face[best_match]:
                    name = face_labels[best_match]

            names.append(name)
        processing_frame = not processing_frame

        # Displaying output
        for(top, right, bottom, left), name in zip(face_locations, names):
            #Scaling face location since frame we detected was scaled by 1/4 in size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Drawing a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2);

            # Drawing the name below the given face
            cv2.rectangle(frame(left, bottom-35), (right, bottom), (0,0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)

        # Displaying the output image
        cv2.imshow('Video', frame)

        # if 'q' leave program
        if cv2.waitKey(1) & 0xFF == ord('q'): break

# Rea;easing handling to webcam
video_capture.release()
cv2.destroyAllWindows()
