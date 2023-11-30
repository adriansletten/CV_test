import requests 
from dotenv import load_dotenv
import os


load_dotenv()
# Access the API key



def predict_from_img_url(api_key, url, image_url, confidence=40, overlap=30):
    response = requests.post(
        url=url,
        params = {
        "api_key": api_key,
        "image": image_url,
        "confidence": confidence,
        "overlap": overlap,
        "format": "json"
        }
    )
    return response


if __name__ == "__main__":
    api_key = os.environ.get('ROBOFLOW_API_KEY')

    base_url = "https://detect.roboflow.com"
    project = "lions_and_hippos"
    model_id = "2"
    url = f"{base_url}/{project}/{model_id}"

    # Image to test
    image_url = "https://source.roboflow.com/dYaLBv8sqndggHrHEGx4Eb26sfw1/oXjTWtQAq1XHNayPuNy7/original.jpg"

    # Confidence threshold
    confidence = 40
    overlap = 30


    response = predict_from_img_url(api_key, url, image_url, confidence, overlap)
    print(response.url)
    print(response)
    print(response.json())