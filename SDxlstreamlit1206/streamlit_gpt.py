import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
# import textwrap
from io import BytesIO

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def main():
    # Streamlit 앱의 제목 설정
    st.title("오늘 하루를 기록해 봤어요")
    st.write("")

    # 큰 배경 이미지 생성 (630x891)
    background_image = Image.new("RGB", (630, 891), "ivory")

    # 이미지와 텍스트를 붙여넣기
    paste_images_and_text(background_image)

    # 현재 시간을 가져오기
    current_time = datetime.now().strftime("%Y년 %m월 %d일 ")

    # 현재 시간을 이미지에 삽입 (좌표는 예시)
    draw = ImageDraw.Draw(background_image)
    font_size = 25
    font_path = "maruburi/TTF/MaruBuri-SemiBold.ttf"  # 사용할 폰트 파일 경로 설정
    font = ImageFont.truetype(font_path, size=font_size)  # 폰트 크기 설정
    draw.text((59, 35), f"오늘의 날짜 : {current_time}", font=font, fill="black")

    # 이미지를 Streamlit에 표시
    st.image(background_image, caption="Today's Diary", use_column_width=True)

    # 이미지 다운로드 버튼
    if st.button("Download Diary Image"):
        byte_image = convert_image(background_image)

        st.download_button(
            label="Download Diary Image",
            key="download_button",
            data=byte_image,
            file_name="Todays_Diary.png",
            mime="image/png"
        )

def paste_images_and_text(background_image):
    # 준비된 이미지 경로
    prepared_image_path = "yebbi.jpg"

    # 이미지를 Pillow Image 객체로 변환 및 크기 조절
    prepared_image = Image.open(prepared_image_path).resize((512, 512))

    # 이미지를 배경에 붙여넣기 (좌표는 예시)
    background_image.paste(prepared_image, (59, 100))

    # 준비된 텍스트
    prepared_text = (
        "고양이 고양이 아기고양이 귀여운 고양이 바보같은 고양이는 어디서 뭘할까 고양이 고양이 아기고양이 귀여운 고양이 바보같은 고양이 "
        "고양이 고양이 아기고양이 귀여운 고양이 바보같은 고양이 고양이 고양이 아기고양이 귀여운 고양이 바보같은 고양이 "
        "고양이 고양이 아기고양이 귀여운 고양이 바보같은 고양이 고양이 고양이 아기고양이 귀여운 고양이 바보같은 고양이 "
    )
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
"""
def test:
    for char in prepared_text:
        # 현재 글자의 길이를 추가하여 줄의 길이 갱신
        char_width = font.getbbox(char)[2] - font.getbbox(char)[0]
        line_length += char_width

        # (571, 640) 좌표에 닿으면 다음 줄로 이동
        if line_length >= 512:  # 571 - 59
            x = 59
            y += font.getbbox(char)[3] - font.getbbox(char)[1]  # 다음 줄로 이동
            line_length = 0  # 줄의 길이 초기화
"""

if __name__ == "__main__":
    main()