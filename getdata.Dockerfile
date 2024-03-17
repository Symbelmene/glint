# syntax=docker/dockerfile:1

FROM python:3.10
WORKDIR /app

RUN apt-get update

RUN apt-get install libpq-dev -y
RUN apt-get install gcc -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

WORKDIR /app/src
CMD ["python", "main.py"]
