FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY backend ./backend
COPY agents ./agents
COPY tools ./tools
COPY pipeline.py ./app.py

# Create directories
RUN mkdir -p backend/data outputs

# Cloud Run uses port 8080
EXPOSE 8080

# Use PORT provided by Cloud Run
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
