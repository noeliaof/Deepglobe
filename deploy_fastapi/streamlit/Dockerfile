FROM python:3.11-slim

RUN mkdir /streamlit

COPY requirements.txt /streamlit

WORKDIR /streamlit

RUN pip install -r requirements.txt --no-cache-dir

COPY . /streamlit

EXPOSE 8501

CMD ["streamlit", "run", "ui.py"]

#CMD source /app/.venv/bin/activate && exec streamlit run ui.py
