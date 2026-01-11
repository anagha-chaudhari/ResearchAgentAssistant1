FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY agents ./agents
COPY tools ./tools
COPY pipeline.py ./app.py

RUN mkdir -p backend/data outputs

EXPOSE 8080

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
