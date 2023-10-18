FROM python:3.8

WORKDIR /ml-model-deploy-1
COPY . /ml-model-deploy-1/

RUN apt-get update && apt-get install libgl1 -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

#Install requirements
RUN pip3 install -r requirements.txt

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "flask_app.app:app", "--reload"]

