import streamlit as st
import requests 
import base64
from PIL import Image, ImageDraw
import io


def predict_from_img_url(image_url, confidence, iou_thresh, api_key, url):
    response = requests.post(
        url=url,
        params = {
        "api_key": api_key,
        "image": image_url,
        "confidence": confidence,
        "iou_threshold": iou_thresh,
        "format": "json"
        }
    )
    return response

def predict_from_img_data(image_bytes, confidence, iou_thresh, api_key, url):
    response = requests.post(
        url=url,
        params={
            "api_key": api_key,
            "confidence": confidence,
            "iou_threshold": iou_thresh,
            "format": "json"
        },
        data=base64.b64encode(image_bytes).decode("ascii"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return response

def draw_boxes(image, list_of_boxes):
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

    draw = ImageDraw.Draw(image)
    for box in list_of_boxes:
        points = (box["x"] - box["width"]/2, box["y"] - box["height"]/2, box["x"] + box["width"]/2, box["y"] + box["height"]/2)
        draw.rectangle(points, outline=colors[box["class_id"]], width=3)

        text = f"{box['class']} {box['confidence']:.2f}"
        draw.text((box["x"] - box["width"]/2, box["y"] - box["height"]/2), text, fill="white", anchor="ld")#, stroke_width=3, stroke_fill="black")
    return image


# API URL
base_url = "https://detect.roboflow.com"
project = "lions_and_hippos"
model_id = "1"
url = f"{base_url}/{project}/{model_id}"

# API key
api_key = st.secrets["roboflow_api_key"]
image = None


st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="CV Test", page_icon=":lion_face:")

# Sidebar
with st.sidebar:
    st.title("How to use:")
    st.markdown("1. **Upload an image or specify an image URL.**")


    # st.subheader("Image")
    # st.write("Upload an image or specify an image URL.")

    # choose one or other
    selected = st.radio("Select", ("Upload", "URL"), horizontal=True, label_visibility="collapsed")

    if selected == "Upload":
        image_file = st.file_uploader("Image file", label_visibility="collapsed")
        if image_file:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))

    elif selected == "URL":
        image_url = st.text_input("Image URL", placeholder="Image URL", label_visibility="collapsed")
        if image_url:
            image = Image.open(requests.get(image_url, stream=True).raw)

    st.markdown("2. **Adjust the confidence and IoU thresholds. Not usually needed.**")
    with st.expander("parameters", expanded=False):
        confidence = st.slider("Confidence Threshold", 0., 1., .4, .01)
        iou_thresh = st.slider("IoU Threshold", 0., 1., .5, .01)

    st.markdown("3. **Run the model trained on the [Hippos and Lions](https://universe.roboflow.com/adriansletten/lions_and_hippos) dataset.**")
    pressed = st.button("Run", disabled=image is None)

    if pressed:
        # Make inference
        with st.spinner("Running inference..."):
            if selected == "Upload":
                response = predict_from_img_data(image_bytes, confidence, iou_thresh, api_key, url)
            elif selected == "URL":
                response = predict_from_img_url(image_url, confidence, iou_thresh, api_key, url)


# Title
st.title("Modern Computer Vision Test :lion_face::hippopotamus:")

if image:
    st.subheader("Image")
    placeholder = st.empty()
    placeholder.image(image)    

if pressed:
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        st.error(f"Request failed: {response.status_code} {response.reason}")
        st.stop()

    # Display results
    placeholder.image(draw_boxes(image, response.json()["predictions"]))
    
    classes = [":hippopotamus:", ":lion_face:"]
    predictions = ""
    for pred in response.json()["predictions"]:
        predictions += f"{classes[pred['class_id']]}"
    st.write(f"**Predictions:** {predictions}")

    st.write("---")

    st.subheader("JSON model output")
    with st.expander("", expanded=False):
        st.json(response.json())


# Hide rainbow bar on the top of page, "Made with Streamlit", and hamburger menu
hide_streamlit_style = '''
    <style>
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
'''
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 