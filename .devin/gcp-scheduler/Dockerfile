FROM python:3.11.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --requirement /app/requirements.txt
COPY app.py /app/app.py
USER 65532:65532
EXPOSE 8080
CMD ["python", "/app/app.py"]
