# FROM python:3.15-rc-trixie
FROM python:3.10-slim

WORKDIR /training-app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  && apt-get autoremove -yqq --purge \
  && apt-get install -y python3-pip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/ scripts/
COPY configs/ configs/
COPY requirements/ requirements/


RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r /training-app/requirements/requirements.txt \
  && pip install --no-cache-dir jupyter

EXPOSE 8081

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8081", "--no-browser", "--allow-root"]