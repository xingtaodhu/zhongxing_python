FROM python:3.6

COPY . /code

WORKDIR /code

RUN python -m pip install --upgrade pip
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com
RUN pip install -r requirements.txt --default-timeout=100

CMD ["python","/code/train.py"]

