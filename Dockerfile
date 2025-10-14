# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of your code
COPY . .

# Expose port 8080
ENV PORT=8080

# Command to run your app
CMD ["python", "percentile_g.py"]
