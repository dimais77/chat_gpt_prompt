import os
import re
import pandas as pd
from dotenv import load_dotenv
import openai
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def clean_dialog_text(text):
    text = re.sub(r'\[\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}]', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^\w\s,.!?]', '', text)
    text = re.sub(r'[!?.]{2,}', '.', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s([?.!,])', r'\1', text)
    return text


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['dialog_text'] = df['dialog_text'].apply(clean_dialog_text)
    return df


def create_prompt(dialog_text):
    prompt = (
        f"Создай задачу в CRM для поддержания договоренностей с клиентом на "
        f"основе анализа следующего диалога между менеджером и клиентом:"
        f"\n\n{dialog_text}\n\n"
        f"Задача должна содержать конкретные мероприятия на основе запроса "
        f"клиента и действий менеджера: выполнить запрос клиента, осуществить "
        f"напоминание клиенту, запросить обратную связь у клиента, "
        f"предоставить ответ клиенту или поблагодарить клиента."
    )
    return prompt


def send_to_openai(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты помощник, который создает задачи для CRM на основе анализа диалогов между клиентами и менеджерами."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']

    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)
    except openai.RateLimitError:
        print("A 429 status code was received; we should back off a bit.")
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)


def main():
    file_path = 'test_dialogs_dataset.csv'
    df = load_and_preprocess_data(file_path)

    results = []
    for dialog_text in df['dialog_text']:
        prompt = create_prompt(dialog_text)
        response = send_to_openai(prompt)
        results.append(response)

    df['response'] = results
    df.to_csv('dialog_responses.csv', index=False)


if __name__ == "__main__":
    main()
