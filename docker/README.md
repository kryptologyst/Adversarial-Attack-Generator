# Adversarial Attack Generator - Docker Setup

## Dockerfile

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data checkpoints

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Docker Compose

```yaml
version: '3.8'

services:
  adversarial-attack-generator:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped

  # Optional: Add a database service
  # postgres:
  #   image: postgres:13
  #   environment:
  #     POSTGRES_DB: adversarial_attacks
  #     POSTGRES_USER: user
  #     POSTGRES_PASSWORD: password
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data
  #   ports:
  #     - "5432:5432"

# volumes:
#   postgres_data:
```

## Usage

### Build and run with Docker
```bash
# Build the image
docker build -t adversarial-attack-generator .

# Run the container
docker run -p 8501:8501 adversarial-attack-generator
```

### Build and run with Docker Compose
```bash
# Build and start services
docker-compose up --build

# Run in background
docker-compose up -d

# Stop services
docker-compose down
```

### Access the application
Open your browser and navigate to `http://localhost:8501`

## Environment Variables

You can customize the application using environment variables:

```bash
# Model configuration
MODEL_TYPE=modern
LEARNING_RATE=0.001
BATCH_SIZE=64

# Attack configuration
FGSM_EPSILON=0.25
PGD_EPSILON=0.25

# Database configuration
DB_PATH=adversarial_attacks.db

# UI configuration
PAGE_TITLE="Adversarial Attack Generator"
```

## Production Deployment

For production deployment, consider:

1. **Use a reverse proxy** (nginx) for SSL termination
2. **Set up proper logging** and monitoring
3. **Use environment-specific configurations**
4. **Implement health checks** and auto-restart
5. **Use container orchestration** (Kubernetes, Docker Swarm)

### Example nginx configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
