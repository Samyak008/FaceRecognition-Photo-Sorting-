import cv2
import face_recognition
import os

# Directory containing the album images
album_directory = r'C:\Users\Samyak Varia\OneDrive\Pictures\Camera Roll' # Replace with the actual path to your album

# Lists to store results
matched_images = []
unknown_images = []

# Capture a photo of the person using the webcam
def capture_photo():
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Display the resulting frame
        cv2.imshow('Capture Photo', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

    return frame

# Load the captured photo
captured_photo = capture_photo()

# Convert the captured photo to RGB format
rgb_captured_photo = cv2.cvtColor(captured_photo, cv2.COLOR_BGR2RGB)

# Find faces in the captured photo
face_locations = face_recognition.face_locations(rgb_captured_photo)
face_encodings = face_recognition.face_encodings(rgb_captured_photo, face_locations)

# Process each image in the album
for filename in os.listdir(album_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter only image files
        img_path = os.path.join(album_directory, filename)

        # Load an image from file
        img = cv2.imread(img_path)

        # Check if the image is loaded successfully
        if img is not None:
            # Convert the image to RGB format (required by face_recognition)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Compare each detected face with the captured face
            for face_location, face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces([face_encoding], face_encoding, tolerance=0.6)

                name = "Unknown"

                # If a match is found, use the name of the known person
                if matches[0]:
                    name = "Matched Person"
                    matched_images.append(rgb_img)  # Append the matched image
                    cv2.imshow(f"Matched Image - {filename}", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

                else:
                    unknown_images.append(filename)

                print(f"Image: {filename}, Location: {face_location}, Match: {matches[0]}, Result: {name}")

            # Wait for a key press to close the window
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print(f"Error: Image not loaded - {img_path}")

# Display the final result in the console
print("\nMatched Images:", len(matched_images))
print("Unknown Images:", unknown_images)
