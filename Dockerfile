FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend code
COPY backend ./backend
COPY agents ./agents
COPY tools ./tools
COPY pipeline.py ./app.py

# Create directories
RUN mkdir -p backend/data outputs

# Expose port
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
