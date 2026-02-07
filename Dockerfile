FROM python:3.12-slim

WORKDIR /app

# Install git (needed for setup.sh to clone repos)
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy app code
COPY . .
RUN chmod +x start.sh setup.sh

EXPOSE 5000

CMD ["./start.sh"]
