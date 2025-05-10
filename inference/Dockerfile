FROM python:3.12
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY infer.py ./
COPY ../iris_model.pt ./
COPY ../data/ ./data/
CMD ["python", "infer.py"]