# Stack geoespacial/científica — imagem única para todos os estágios.
# Base com GDAL pré-compilado evita builds longos.
   FROM python:3.12-slim
   RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
       && rm -rf /var/lib/apt/lists/*
   WORKDIR /pipeline
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt