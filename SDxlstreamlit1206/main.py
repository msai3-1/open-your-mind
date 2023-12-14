import ast
from typing import Optional

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
# from diffusers.utils import load_image, make_image_grid
# import cv2
# import numpy as np
from streamlit_extras.let_it_rain import rain
import openai
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from io import BytesIO
from sd2.generate import PIPELINE_NAMES, MODEL_VERSIONS, generate

openai.api_key = "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"
DEFAULT_PROMPT = "나는 오늘 강아지를 산책시키고, 하루를 즐겁게 마무리 했어."
DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512
OUTPUT_IMAGE_KEY = "output_img"
LOADED_IMAGE_KEY = "loaded_image"


def get_image(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None


def set_image(key: str, img: Image.Image):
    st.session_state[key] = img


def width_and_height_sliders(prefix):
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider(
            "Width",
            min_value=64,
            max_value=1600,
            step=16,
            value=512,
            key=f"{prefix}-width",
        )
    with col2:
        height = st.slider(
            "Height",
            min_value=64,
            max_value=1600,
            step=16,
            value=512,
            key=f"{prefix}-height",
        )
    return width, height


def image_uploader(prefix):
    image = st.file_uploader("Image", ["jpg", "png"], key=f"{prefix}-uploader")
    if image:
        image = Image.open(image)
        print(f"loaded input image of size ({image.width}, {image.height})")
        return image

    return get_image(LOADED_IMAGE_KEY)


def inpainting():
    image = image_uploader("inpainting")

    if not image:
        return None, None

    brush_size = st.number_input("Brush Size", value=50, min_value=1, max_value=100)

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=brush_size,
        stroke_color="#FFFFFF",
        background_color="#000000",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="freedraw",
        key="inpainting-canvas",
    )

    if not canvas_result or canvas_result.image_data is None:
        return None, None

    mask = canvas_result.image_data
    mask = mask[:, :, -1] > 0
    if mask.sum() > 0:
        mask = Image.fromarray(mask)
        st.image(mask)
        return image, mask

    return None, None

def inpainting_tab():
    prefix = "inpaint"
    col1, col2 = st.columns(2)

    with col1:
        image_input, mask_input = inpainting()

    with col2:
        if image_input and mask_input:
            version = st.selectbox(
                "Model version", ["ControlNet","2.0", "XL 1.0"], key="inpaint-version"
            )
            strength = st.slider(
                "Strength of inpainting (1.0 essentially ignores the masked area of the original input image)",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                key=f"{prefix}-strength",
            )
            prompt_and_generate_button(
                prefix,
                "inpaint",
                image_input=image_input,
                mask_input=mask_input,
                version=version,
                strength=strength,
            )


def img2img_tab():
    prefix = "img2img"
    col1, col2 = st.columns(2)

    with col1:
        image = image_uploader(prefix)
        if image:
            st.image(image)

    with col2:
        if image:
            version = st.selectbox(
                "Model version", ["ControlNet+Lora", "ControlNet","2.1", "XL 1.0 refiner"], key=f"{prefix}-version"
            )
            strength = st.slider(
                "Strength (1.0 ignores the existing image so it's not a useful value)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key=f"{prefix}-strength",
            )
            prompt_and_generate_button(
                prefix, "img2img", image_input=image, version=version, strength=strength
            )


def txt2img_tab():
    prefix = "txt2img"
    width, height = width_and_height_sliders(prefix)
    version = st.selectbox("Model version", ["Lora", "2.1", "XL 1.0"], key=f"{prefix}-version")

    #prompt_and_generate_button(
    #    prefix, "txt2img", width=width, height=height, version=version
    #)

def prompt_and_generate_button(prefix, pipeline_name: PIPELINE_NAMES, **kwargs):
    prompt = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPT,
        key=f"{prefix}-prompt",
    )
    negative_prompt = st.text_area(
        "Negative prompt",
        value="",
        key=f"{prefix}-negative-prompt",
    )

    col1, col2 = st.columns(2)
    with col1:
        steps = st.slider(
            "Number of inference steps",
            min_value=1,
            max_value=200,
            value=20,
            key=f"{prefix}-inference-steps",
        )
    with col2:
        guidance_scale = st.slider(
            "Guidance scale",
            min_value=0.0,
            max_value=20.0,
            value=7.5,
            step=0.5,
            key=f"{prefix}-guidance-scale",
        )
    enable_attention_slicing = st.checkbox(
        "Enable attention slicing (enables higher resolutions but is slower)",
        key=f"{prefix}-attention-slicing",
    )
    enable_cpu_offload = st.checkbox(
        "Enable CPU offload (if you run out of memory, e.g. for XL model)",
        key=f"{prefix}-cpu-offload",
        value=False,
    )

    if st.button("Generate image", key=f"{prefix}-btn"):
        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::8UCeIm3y",
            messages=[
                {"role": "system",
                 "content": "Marv converts the input text into a first-person diary format. The converted diary is saved in Korean within 100 characters."},
                {"role": "user", "content": f"{prompt}"}
            ]
        )

        gpt = completion.choices[0].message["content"]

        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::8UE6t767",
            messages=[
                {"role": "system",
                 "content": "Marco is a model for extracting important words from diaries. Given input text in the form of a diary, Marco can do a number of things. Marco interprets the given diary, understands the context, and extracts noun words that are considered important in the context and translates them into English. Marco understands the input text and analyzes the sentiment of the text. Marco expresses the analyzed sentiment in three words and translates them into English. Marco collects all the extracted results, translates them into English, and lists them in a single line."},
                {"role": "user", "content": f"{gpt}"}
            ]
        )

        make_diary = completion['choices'][0]['message']['content']

        prompt = make_diary

        with st.spinner("Generating image..."):
            image = generate(
                prompt,
                pipeline_name,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                enable_attention_slicing=enable_attention_slicing,
                enable_cpu_offload=enable_cpu_offload,
                **kwargs,
            )
            set_image(OUTPUT_IMAGE_KEY, image.copy())
        st.image(image)
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im
def paste_images_and_text(background_image,sd_image,text):
    # 준비된 이미지 경로
    prepared_image = sd_image

    # 이미지를 Pillow Image 객체로 변환 및 크기 조절
    # prepared_image = Image.open(prepared_image).resize((512, 512))

    # 이미지를 배경에 붙여넣기 (좌표는 예시)
    background_image.paste(prepared_image, (59, 100))

    # 준비된 텍스트
    prepared_text = text

    # 텍스트를 이미지에 삽입 (좌표는 예시)
    draw = ImageDraw.Draw(background_image)
    font_size = 20
    font_path = "maruburi/TTF/MaruBuri-Regular.ttf"  # 사용할 폰트 파일 경로 설정
    font = ImageFont.truetype(font_path, size=font_size)  # 폰트 크기 설정

    # 초기 위치 설정
    x, y = 59, 640
    line_length = 0  # 현재 줄의 길이

    for char in prepared_text:
        # 현재 글자의 길이를 추가하여 줄의 길이 갱신
        line_length += font.getsize(char)[0]

        # (571, 640) 좌표에 닿으면 다음 줄로 이동
        if line_length >= 512:  # 571 - 59
            x = 59
            y += font.getsize(char)[1]  # 다음 줄로 이동
            line_length = 0  # 줄의 길이 초기화

        draw.text((x, y), char, font=font, fill="black")
        x += font.getsize(char)[0]  # 다음 글자 위치로 이동

    # 현재 시간을 가져오기
    current_time = datetime.now().strftime("%Y년 %m월 %d일 ")

    # 현재 시간을 이미지에 삽입 (좌표는 예시)
    draw = ImageDraw.Draw(background_image)
    font_size = 25
    font_path = "maruburi/TTF/MaruBuri-SemiBold.ttf"  # 사용할 폰트 파일 경로 설정
    font = ImageFont.truetype(font_path, size=font_size)  # 폰트 크기 설정
    draw.text((59, 35), f"오늘의 날짜 : {current_time}", font=font, fill="black")

def save_image(image, filename):
    image.save(filename)
def main():
    # Setting page layout
    st.set_page_config(
        page_title="MS AI SCHOOL 1팀",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="collapsed")
    # st.title("🎨 Create image via Stable Diffusion 👨🏻‍💻")
    st.header("🎨:rainbow[Create image via Stable Diffusion]👨🏻‍💻")

    # Displaying a rain animation with specified parameters
    rain(
        emoji="🎈",
        font_size=54,
        falling_speed=5,
        animation_length="1",
    )

    tab1, tab2= st.tabs(
        ["Make your diary","Diary"]
    )

    #tab1조정
    with tab1:
        prefix="txt2img"
        pipeline_name="txt2img"
        # version = st.selectbox("Model version", ["Ghibli", "2.1", "XL 1.0"], key=f"{prefix}-version")
        version="Ghibli"
        prompt = st.text_area(
            "Prompt",
            value=DEFAULT_PROMPT,
            key=f"{prefix}-prompt",
        )
        negative_prompt = st.text_area(
            "Negative prompt",
            value="bad quality",
            key=f"{prefix}-negative-prompt",
        )

        col1, col2 = st.columns(2)
        with col1:
            steps = st.slider(
                "Number of inference steps",
                min_value=1,
                max_value=200,
                value=20,
                key=f"{prefix}-inference-steps",
            )
        with col2:
            guidance_scale = st.slider(
                "Guidance scale",
                min_value=0.0,
                max_value=20.0,
                value=7.5,
                step=0.5,
                key=f"{prefix}-guidance-scale",
            )

        if st.button("Generate image", key=f"{prefix}-btn"):
            completion = openai.ChatCompletion.create(
                model="ft:gpt-3.5-turbo-0613:personal::8UCeIm3y",
                messages=[
                    {"role": "system",
                     "content": "Marv converts the input text into a first-person diary format. The converted diary is saved in Korean within 100 characters."},
                    {"role": "user", "content": f"{prompt}"}
                ]
            )

            gpt = completion.choices[0].message["content"]

            completion = openai.ChatCompletion.create(
                model="ft:gpt-3.5-turbo-0613:personal::8UE6t767",
                messages=[
                    {"role": "system",
                     "content": "Marco is a model for extracting important words from diaries. Given input text in the form of a diary, Marco can do a number of things. Marco interprets the given diary, understands the context, and extracts noun words that are considered important in the context and translates them into English. Marco understands the input text and analyzes the sentiment of the text. Marco expresses the analyzed sentiment in three words and translates them into English. Marco collects all the extracted results, translates them into English, and lists them in a single line."},
                    {"role": "user", "content": f"{gpt}"}
                ]
            )

            make_diary = completion['choices'][0]['message']['content']

            prompt = make_diary + ', better quality' + ', ghibli style'

            with st.spinner("Generating image..."):
                image = generate(
                    prompt,
                    pipeline_name,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    version=version,
                )
                # set_image(OUTPUT_IMAGE_KEY, image.copy())
            #tab2 시작
            with tab2:
                background_image = Image.new("RGB", (630, 891), "ivory")
                paste_images_and_text(background_image,image,gpt)
                st.image(background_image, caption="Today's Diary", use_column_width=True)
                current_time = datetime.now().strftime("%Y-%m-%d ")
                background_image.save(f"diary/{current_time}diary.png")

    with st.sidebar:
        st.header("Latest Output Image")
        output_image = get_image(OUTPUT_IMAGE_KEY)
        if output_image:
            st.image(output_image)
            if st.button("Use this image for img2img"):
                set_image(LOADED_IMAGE_KEY, output_image.copy())
                st.experimental_rerun()
            st.markdown(
                  f":rainbow[{st.session_state['txt2img-prompt']}]" if "txt2img-prompt" in st.session_state else""
            )
        else:
            st.markdown("No output generated yet")


if __name__ == "__main__":
    main()