From python:3.7.1

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt --no-cache-dir

EXPOSE 8501

COPY . /app

ENTRYPOINT ["streamlit","run"]

CMD ["src/main_app.py"]