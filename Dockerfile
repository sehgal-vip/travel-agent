FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source code
COPY src/ src/

# Create data directory for SQLite
RUN mkdir -p /app/data

# Run the bot
CMD ["python", "-m", "src.main"]
