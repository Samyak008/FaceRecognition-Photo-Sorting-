import cv2
import face_recognition

# Load the known image (image of the person you want to recognize)
known_image_path = r'E:\Samyak\WhatsApp Image 2023-07-07 at 19.03.05.jpg'  # Replace with the actual path
known_image = face_recognition.load_image_file(r'E:\Samyak\WhatsApp Image 2023-07-07 at 19.03.05.jpg')
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load an image from file
img_path = 'C:\\Users\\Samyak Varia\\OneDrive\\Pictures\\Camera Roll\\WIN_20220728_19_31_28_Pro.jpg'  # Replace with the actual path
img = cv2.imread(img_path)

# Check if the image is loaded successfully
if img is not None:
    # Convert the image to RGB format (required by face_recognition)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find faces in the image
    face_locations = face_recognition.face_locations(rgb_img)
    print("Detected face locations:", face_locations)

    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    # Compare each detected face with the known face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        name = "Unknown"

        # If a match is found, use the name of the known person
        if matches[0]:
            name = "Known Person"

        print("Name:", name)
        print("Face Coordinates:", (left, top, right, bottom))

        # Draw a rectangle around the face and display the name
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the result
    cv2.imshow('Face Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Image not loaded.")
