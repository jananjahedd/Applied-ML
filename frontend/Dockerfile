# /frontend/ Dockerfile
FROM python:3.9-slim-bullseye

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "😴_Homepage.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.serverAddress=localhost"]
