# CityPulse AI - container image
# Build:  docker build -t citypulse-ai .
# Run Streamlit:  docker run -p 8501:8501 citypulse-ai streamlit run app/app.py --server.address=0.0.0.0
# Run API:        docker run -p 8000:8000 citypulse-ai uvicorn api.main:app --host 0.0.0.0

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build data + models at image build time so the container is ready to serve immediately.
RUN python run_pipeline.py

EXPOSE 8501 8000

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
