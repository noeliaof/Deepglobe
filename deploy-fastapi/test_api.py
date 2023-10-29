import requests
import cv2
# Define the URL of your FastAPI application
url = "http://localhost:8000/segment"  # Adjust the URL as needed

# Specify the file path of the image you want to upload
image_file_path = "/Users/noeliaotero/Documents/WeCloudData/Capstone_project/Deepglobe/data/test/209073_sat.jpg"  # Replace with the actual file path of your image

# Read the image and print its shape
#image = cv2.imread(image_file_path)
#if image is not None:
#    print("Image shape:", image.shape)

with open(image_file_path, "rb") as file:
    files = {'file': (image_file_path, file, 'image/jpeg')}  # Adjust 'image/jpeg' as needed for your image type
    response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()
        print(result.items())
        

        # Perform any further testing or validation here
    else:
        print("Request for", image_file_path, "failed with status code:", response.status_code)


