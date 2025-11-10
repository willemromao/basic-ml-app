# basic-ml-app

Este repositório foi criado com propósitos educacionais para o curso IMD3005 - MLOPS, demonstrando como transformar um modelo treinado em um serviço web a ser implantado em produção. Atenção, pode conter pequenos bugs que precisam ser consertados. Para reportar bugs ou solicitar apoio, entre em contato por e-mail `adelson.araujo@imd.ufrn.br`.


---

## 🌱 Overview do progresso:

Acompanhe abaixo a linha temporal das alterações realizadas até o momento: 

> _______________
> ### 1️⃣ : Servindo predições com FastAPI
> Nesta aula, focamos em transformar o módulo `intent_classifier/` em uma API RESTful utilizando o FastAPI.
>
> **Tópicos abordados**:
> *   Exploração dos conceitos básicos do FastAPI para construção de APIs web.
> *   Treinamento de modelos de ML e observação dos experimentos (via integração com `W&B`) para selecionar modelo eficaz.
> *   Demonstração de como carregar um modelo de ML previamente treinado (`.keras`) para uso em produção.
> *   Implementação de um endpoint HTTP (`/predict`) para receber requisições e retornar predições do modelo.
> *   Criação do arquivo `app/app.py` com a lógica essencial para inicializar o FastAPI e expor o modelo. 
> 
> _______________

> _______________
> ### 2️⃣ : Incorporando persistência, autenticação, e containerização
> 
> Nesta aula, expandimos a arquitetura do projeto para incluir persistência de dados (via Mongo-DB), autenticação simples por token de acesso, e conteinerização com Docker.
> 
> **Tópicos abordados:**
> 
> *   Discussão sobre a separação de responsabilidades (backend, ML, banco de dados, testes, DAGs) para um projeto MLOps escalável.
> *   Persistência de dados com MongoDB e PyMongo, salvando inputs e predições.
> *   Autenticação simples baseada em token de acesso.
> *   Criação de um `Dockerfile` (e `docker-compose.yml`) para empacotar o serviço web em um container isolado.
> _______________

> _______________
> ### 3️⃣ : Implementando integração contínua (CI) com GitHub actions
>
> **Tópicos abordados:**
> *   Importância dos testes automatizados e da Integração Contínua (CI) no desenvolvimento de MLOps.
> *   Criação testes unitários e de integração (16 testes, ~52% cobertura).
> *   Configurar um workflow básico de GitHub Actions para executar os testes unitários e construir a imagem Docker do serviço FastAPI.
> 
> **Como rodar os testes:**
> ```bash
> # Usando o script automatizado
> chmod +x test.sh
> ./test.sh
> 
> # Ou diretamente no container
> docker compose exec backend pytest tests/ -v -m "not integration"
> 
> # Com cobertura detalhada
> docker compose exec backend pytest tests/ -v --cov=app --cov=db --cov-report=html
> # Abra htmlcov/index.html no navegador
> ```
> _______________



---

## 🏛️ Estrutura atual do projeto

```shell
.                               # "Working directory"
├── app/                        # Lógica do serviço web
│   ├── app.py                  # Implementação do backend com FastAPI
│   ├── app.Dockerfile          # Definição do container em que o backend roda
│   └── auth.py                 # Implementação do backend
├── db/                         # Lógica do banco de dados
│   └── engine.py               # Encapsulamento do pymongo
├── intent-classifier/          # Scripts relacionados ao modelo de ML
│   ├── data/                   # Dados para os modelos de ML
│   ├── models/                 # Modelos treinados
│   └── intent-classifier.py    # Código principal do modelo de ML
├── dags/                       # Workflows integrados no Airflow
│   └── ...                     # TODO
├── tests/                      # Testes unitários e de integração
│   └── ...                     # TODO
├── docker-compose.yml          # Arquivo de orquestração dos serviços envolvidos
├── requirements.txt            # Dependências do Python
├── .env                        # Variáveis de ambiente
└── .gitignore
```

## ⚙️ Instruções para deploy em ambiente de teste

### Localmente
```shell
# Crie e ative um ambiente conda com as dependências do projeto
conda create -n intent-clf python=3.11
conda activate intent-clf
pip install -r requirements.txt # instalar as dependências
## Ajuste seu .env com as variáveis de ambiente necessárias
export ENV=dev
## Em .env, se ENV=prod, você precisará criar um token
## python app/auth.py create --owner="nome" --expires_in_days=365
# Suba o serviço web e acesse-o em localhost:8000
uvicorn app.app:app --host 0.0.0.0 --port 8000 --log-level debug
```

### Utilizando o Docker

### Construindo a imagem do container
```shell
docker build -t intent-clf:0.1 -f app/app.Dockerfile .
```

### Executando o container 
```shell
docker run -d -p 8080:8000 --name intent-clf-container intent-clf:0.1
# Checar os containers ativos
docker ps
# Acompanhar os logs do container
docker logs -f intent-clf-container
```
Ou construa um arquivo `docker-compose.yml` (útil para execução de vários containers com um só comando) e execute:
```shell
docker-compose up -d
# Checar os containers ativos
docker ps
# Acompanhar os logs do container
docker logs -f intent-clf-container
```
Para interromper a execução do container:
```shell
# Parar o container
docker stop intent-clf-container
# Deletar o container (com -f ou --force você deleta sem precisar parar)
docker rm -f intent-clf-container
```

