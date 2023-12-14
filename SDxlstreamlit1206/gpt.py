import openai

openai.api_key = "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"

user_content = input(" 오늘 하루를 입력하세요  : ")

completion = openai.ChatCompletion.create(
    model="ft:gpt-3.5-turbo-0613:personal::8UCeIm3y",
    messages=[
        {"role": "system", "content": "Marv converts the input text into a first-person diary format. The converted diary is saved in Korean within 100 characters."},
        {"role": "user", "content": f"{user_content}"}
    ]
)

gpt = completion.choices[0].message["content"]

completion = openai.ChatCompletion.create(
    model="ft:gpt-3.5-turbo-0613:personal::8UE6t767",
    messages=[
        {"role": "system", "content": "Marco is a model for extracting important words from diaries. Given input text in the form of a diary, Marco can do a number of things. Marco interprets the given diary, understands the context, and extracts noun words that are considered important in the context and translates them into English. Marco understands the input text and analyzes the sentiment of the text. Marco expresses the analyzed sentiment in three words and translates them into English. Marco collects all the extracted results, translates them into English, and lists them in a single line."},
        {"role": "user", "content": f"{gpt}"}
    ]
)

make_diary = completion['choices'][0]['message']['content']

