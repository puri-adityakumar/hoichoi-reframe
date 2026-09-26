# Build context: worker/ (bl deploy runs from this directory)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# worker.reframe imports resolve via /app on sys.path (namespace package)
COPY reframe/ /app/worker/reframe/
COPY spec/ /app/spec/
COPY models/ /app/worker/models/
COPY job_entry.py /app/job_entry.py

CMD ["python", "-u", "/app/job_entry.py"]
