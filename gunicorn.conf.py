import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker settings
# Use 1 worker to prevent OOM on 1GB RAM VM
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = int(os.getenv("TIMEOUT", "120"))
keepalive = 5

# Logging settings
loglevel = os.getenv("LOG_LEVEL", "info")
accesslog = "-"
errorlog = "-"

proc_name = "litenetx-backend"
