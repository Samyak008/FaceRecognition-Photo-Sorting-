import cv2
import face_recognition
import os

# Load the known image (image of the person you want to recognize)
known_image_path = r'E:\Samyak\WhatsApp Image 2023-07-07 at 19.03.05.jpg'  # Replace with the actual path
known_image = face_recognition.load_image_file(known_image_path)
known_encoding = face_recognition.face_encodings(known_image)[0]

# Directory containing the album images
album_directory = r'F:\southfamily1\1'  # Replace with the actual path to your album

# Lists to store results
matched_images = []
unknown_images = []

# Resize function to fit the images for processing
def resize_for_processing(image, max_height=800):
    h, w = image.shape[:2]
    if h > max_height:
        scale_factor = max_height / h
        new_height = int(h * scale_factor)
        new_width = int(w * scale_factor)
        return cv2.resize(image, (new_width, new_height))
    return image

# Resize function to fit the images for displaying
def resize_for_display(image, max_width=800):
    h, w = image.shape[:2]
    if w > max_width:
        scale_factor = max_width / w
        new_height = int(h * scale_factor)
        new_width = int(w * scale_factor)
        return cv2.resize(image, (new_width, new_height))
    return image

# Process each image in the album
for filename in os.listdir(album_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter only image files
        img_path = os.path.join(album_directory, filename)

        # Load an image from file
        img = cv2.imread(img_path)

        # Check if the image is loaded successfully
        if img is not None:
            # Resize the image for processing
            img = resize_for_processing(img)

            # Convert the image to RGB format (required by face_recognition)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Find faces in the image
            face_locations = face_recognition.face_locations(rgb_img)
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

            # Compare each detected face with the known face
            for face_location, face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.5)  # Adjust tolerance

                name = "Unknown"

                # If a match is found, use the name of the known person
                if matches[0]:
                    name = "Known Person"
                    matched_images.append(rgb_img)  # Append the matched image
                    cv2.imshow(f"Matched Image - {filename}", resize_for_display(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)))

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
