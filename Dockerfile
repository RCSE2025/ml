FROM python:3.11

WORKDIR /nsfw_checker

COPY ./requirements.txt ./requirements.txt
COPY ./install.sh ./install.sh

RUN ["chmod", "+x", "install.sh"]
RUN ["./install.sh"]

COPY . .

EXPOSE 8000
EXPOSE 8001
CMD ["python", "main.py"]