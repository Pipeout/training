FROM python:3.10-slim

WORKDIR /training-app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/entrypoint.sh entrypoint.sh
COPY scripts/ scripts/
COPY configs/ configs/

RUN chmod +x entrypoint.sh

COPY infra/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8081

ENTRYPOINT [ "./entrypoint.sh" ]
