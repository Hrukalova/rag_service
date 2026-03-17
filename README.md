# 🏛️ HSE Retrieval Service (pure-RAG)

Микросервис для **семантического поиска** по базе знаний НИУ ВШЭ.

## 🚀 Возможности
-   **Векторный поиск**: Использует `pgvector` и модель `mxbai-embed-large-v1` для ANN-поиска.
-   **Profile Scoping (Безопасность)**: Фильтрация документов на уровне БД через `scope_json` (учитывает вуз, кампус, факультет, программу и роль студента).
-   **Интеграция со схемой `library`**: Поддерживает актуальную структуру таблиц `documents`, `chunks`, `chunk_embeddings`.
-   **FastAPI**: Быстрый асинхронный поиск с валидацией через Pydantic.

## 🛠️ Стек
-   **Python 3.10+**
-   **FastAPI** (Web Framework)
-   **SQLAlchemy / AsyncPG** (DB Ops)
-   **Sentence-Transformers** (Embedder)
-   **PgVector** (Vector database engine)

## 📦 Быстрый старт
1.  **Установка зависимостей**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Настройка переменных окружения (`.env`)**:
    ```env
    DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/hse_rag
    EMBEDDER_MODEL_NAME=mixedbread-ai/mxbai-embed-large-v1
    ```
3.  **Запуск**:
    ```bash
    uvicorn main:app --reload --port 8001
    ```

## 🔌 API Workflow
Эндпоинт: `POST /retrieve`
```json
{
  "question": "Как получить стипендию?",
  "top_k": 5,
  "user_profile": {
    "university_id": "hse_moscow",
    "campus_id": "pokrovka",
    "year": 3,
    "role": "student"
  }
}
```

## 🔐 Фильтрация (Scope)
Поиск вернет чанки только из тех документов, где `scope_json` либо пуст, либо содержит значения, соответствующие профилю пользователя. Проверка идет по следующим полям:
- `university_ids`
- `campus_ids`
- `faculty_ids`
- `program_ids`
- `years`
- `roles`
