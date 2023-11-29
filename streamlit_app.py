import streamlit as st
import requests 
import base64
from PIL import Image, ImageDraw
import io


def get_img_prediction(api_key, url, confidence, overlap, image_bytes=None, image_url=None):
    """Make request to Roboflow's inference API given an image URL or image bytes."""
    params = {
        "api_key": api_key,
        "confidence": confidence,
        "overlap": overlap,
        "format": "json",
    }
    data = None
    headers = None
    if image_bytes:
        data = base64.b64encode(image_bytes).decode("ascii")
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
    elif image_url:
        params["image"] = image_url

    response = requests.post(
        url=url,
        params=params,
        data=data,
        headers=headers,
    )
    return response


def draw_boxes(image, list_of_boxes):
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

    draw = ImageDraw.Draw(image)
    for box in list_of_boxes:
        points = (box["x"] - box["width"]/2, box["y"] - box["height"]/2, box["x"] + box["width"]/2, box["y"] + box["height"]/2)
        draw.rectangle(points, outline=colors[box["class_id"] % len(colors)], width=3)

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
image_bytes = None
image_url = None


st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="CV Test", page_icon=":lion_face:")

# Sidebar
with st.sidebar:
    st.title("How to use:")
    st.markdown("1. **Upload an image or specify an image URL.**")

    # choose upload or url
    selected = st.radio("Select", ("Upload", "URL"), horizontal=True, label_visibility="collapsed")

    if selected == "Upload":
        image_file = st.file_uploader("Image file", label_visibility="collapsed", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
        if image_file:
            image = Image.open(image_file).convert("RGB")
            # convert img to jpeg (smaller size fits in the url)
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            image_bytes = image_bytes.getvalue()

    elif selected == "URL":
        image_url = st.text_input("Image URL", placeholder="Image URL", label_visibility="collapsed")
        if image_url:
            with requests.get(image_url, stream=True) as resp:
                try:
                    resp.raise_for_status()
                    image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                except requests.exceptions.HTTPError as e:
                    st.error(f"Request failed: {resp.status_code} {resp.reason}")
                except Exception as e:
                    st.error(f"Error: {e}")


    st.markdown("2. **Adjust the confidence and IoU thresholds. Not usually needed.**")
    with st.expander("parameters", expanded=False):
        confidence = st.slider("Confidence Threshold (default: 40)", 0, 100, 40, 1)
        overlap = st.slider("Overlap Threshold (default: 30)", 0, 100, 30, 1)

    st.markdown("3. **Run the model trained on the [Hippos and Lions](https://universe.roboflow.com/adriansletten/lions_and_hippos) dataset.**")
    pressed = st.button("Run", disabled=image is None)

    if pressed:
        # Make inference
        with st.spinner("Running inference..."):
            response = get_img_prediction(api_key, url, confidence, overlap, image_bytes, image_url)


# Title
st.title("Modern Computer Vision Test")
st.write("Given an image, this app will detect whether :lion_face: or :hippopotamus: are present.")

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