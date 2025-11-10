FROM python:3.11-slim

WORKDIR /app

# Cria usuário
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app

# Copia requirements da raiz do projeto
COPY requirements.txt /app/requirements.txt

# Instala dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest pytest-cov pytest-mock httpx

# Copia código
COPY intent_classifier/ /app/intent_classifier/
COPY app/ /app/app/
COPY db/ /app/db/
COPY tests/ /app/tests/

# Ajusta permissões
RUN chown -R appuser:appuser /app

USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]