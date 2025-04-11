FROM python:3.8-slim

WORKDIR /src

COPY . .

RUN pip install --upgrade pip && \
    pip install mlflow[extras]

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001"]