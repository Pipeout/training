FROM python:3.10-slim

WORKDIR /training-app


COPY scripts/ scripts/
COPY configs/ configs/
COPY requirements/ requirements/


RUN pip install --no-cache-dir -r /training-app/requirements/requirements.txt \
  && pip install --no-cache-dir jupyter

EXPOSE 8081

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8081", "--no-browser", "--allow-root"]
