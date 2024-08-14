FROM tiangolo/uvicorn-gunicorn:python3.9
RUN pip install -U pip
WORKDIR /code
COPY api_requirements.txt .
RUN pip install --no-cache-dir --upgrade -r api_requirements.txt
COPY ./app /code/app
COPY model.pkl .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
