FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[dashboard]

EXPOSE 8501

CMD ["streamlit", "run", "examples/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]