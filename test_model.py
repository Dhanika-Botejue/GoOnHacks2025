import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

api_key = os.getenv("API_KEY")

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)
image_folder = "/Users/dhanika/Go_on_hacks/GoOnHacks2025/tongue_images"
# infer on a local image
counter = 0
for filename in os.listdir(image_folder):
    if counter > 1:
        break
    img_path = os.path.join(image_folder, filename)
    result = CLIENT.infer(img_path, model_id="tongue-wltgn/2")
    counter += 1
    try:
        print(result["predictions"][0]["x"], result["predictions"][0]["y"], result["predictions"][0]["confidence"])
    except:
        print("No prediction")