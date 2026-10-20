FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY opengem.py .
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
USER nobody
CMD ["python", "opengem.py"]
