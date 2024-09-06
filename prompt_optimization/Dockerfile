FROM python:3.11-slim-bullseye
WORKDIR /workspace

# Install necessary packages
RUN apt-get update && apt-get install -y \
	vim \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Copy the minimum required stuff
COPY data data
COPY prompts prompts
COPY \
  config.py \
  evaluators.py \
  main.py \
  optimizers.py \
  predictors.py \
  scorers.py \
  tasks.py \
  utils.py \
  .

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
