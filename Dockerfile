FROM python:3.10

WORKDIR /

COPY . .

RUN pip install --no-cache-dir -r requirements/requirements.txt
RUN pip install --no-cache-dir -r requirements/pt2.txt
RUN pip install --no-cache-dir .

CMD [ "python", "-u", "/serverless.py" ]
