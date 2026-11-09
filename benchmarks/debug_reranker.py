from groq import Groq
from src.config import settings

client = Groq(api_key=settings.GROQ_API_KEY)
for m in ['openai/gpt-oss-20b', 'allam-2-7b']:
    try:
        resp = client.chat.completions.create(
            model=m,
            messages=[{'role': 'user', 'content': 'Output valid JSON with key "message": "hello"'}],
            response_format={'type': 'json_object'}
        )
        print(f'{m}: SUCCESS -> {resp.choices[0].message.content}')
    except Exception as e:
        print(f'{m}: FAILED -> {e}')
