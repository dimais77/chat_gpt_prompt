### Описание проекта

Данный проект предназначен для анализа диалогов между клиентами и менеджерами и создания задач в CRM на основе этих диалогов. Задачи формулируются при помощи модели GPT-3.5-turbo от OpenAI. С помощью предобработки текста и генерации промтов, система создает задачи для поддержания договоренностей с клиентами.

### Файлы

- `test_dialogs_dataset.csv` — входной файл с диалогами для анализа.
- `dialog_responses.csv` — выходной файл с результатами (сгенерированные задачи для CRM).

### Установка

1. Склонируйте репозиторий.
2. Установите зависимости:

   ```bash
   pip install openai pandas python-dotenv
   ```

3. Создайте файл `.env` и добавьте свой API ключ OpenAI:

   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

### Использование

Для запуска скрипта выполните следующую команду:

```bash
python chat_gpt_prompt.py
```

Скрипт выполнит следующие шаги:
1. Загрузит и предобработает текст диалогов.
2. Сгенерирует задачи в CRM на основе анализа диалогов.
3. Сохранит результаты в файл `dialog_responses.csv`.

### Основные функции

- `clean_dialog_text(text)` — функция для предобработки текста диалога (удаление лишних символов, ссылок, форматирование).
- `load_and_preprocess_data(file_path)` — загрузка и очистка данных из CSV-файла.
- `create_prompt(dialog_text)` — создание промта для генерации задачи в CRM на основе диалога.
- `send_to_openai(prompt)` — отправка запроса в OpenAI API для получения ответа.
- `main()` — основная функция для запуска процесса анализа и сохранения результатов.

### Ошибки

Обработка ошибок API OpenAI предусмотрена для таких случаев:
- Неправильный запрос (код 429).
- Ошибка соединения с сервером.
- Другие статус-коды.

### Контакты

Telegram: https://t.me/dimais77
