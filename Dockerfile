FROM python:3.12.7-slim-bookworm

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .