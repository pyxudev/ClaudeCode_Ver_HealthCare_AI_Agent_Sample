# Design doc section 7 pins Python 3.13.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/srv/app/src

WORKDIR /srv/app

# Dependencies first so code edits do not invalidate the wheel layer.
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/src ./src
COPY app/tests ./tests
COPY app/pytest.ini ./pytest.ini

# Fallback copy of the dataset. docker-compose bind-mounts the real one over
# /data/sample_doctors.json; this keeps the image runnable on its own.
COPY sample_doctors.json /data/sample_doctors.json

# Non-root: the app never needs to write to the filesystem.
RUN useradd --create-home --uid 10001 hdai && chown -R hdai:hdai /srv/app
USER hdai

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "hdai.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
