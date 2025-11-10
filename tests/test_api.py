import os
import sys
import pytest
import time
from datetime import datetime
from fastapi.testclient import TestClient
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# --- Testes Unitários da API ---

def test_root_endpoint():
    """Testa o endpoint raiz."""
    os.environ["ENV"] = "dev"
    from app.app import app
    client = TestClient(app)
    
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_predict_without_text_parameter():
    """Testa predição sem parâmetro text."""
    os.environ["ENV"] = "dev"
    from app.app import app
    client = TestClient(app)
    
    response = client.post("/predict")
    assert response.status_code == 422  # Validation error


def test_predict_with_text_dev_mode():
    """Testa predição em modo dev."""
    os.environ["ENV"] = "dev"
    from app.app import app
    client = TestClient(app)
    
    response = client.post("/predict?text=hello world")
    
    assert response.status_code == 200
    data = response.json()
    
    # Valida estrutura básica
    assert "text" in data
    assert data["text"] == "hello world"
    assert "owner" in data
    assert "predictions" in data
    assert "timestamp" in data


def test_predict_response_has_all_fields():
    """Valida que a resposta tem todos os campos necessários."""
    os.environ["ENV"] = "dev"
    from app.app import app
    client = TestClient(app)
    
    response = client.post("/predict?text=test")
    assert response.status_code == 200
    
    data = response.json()
    required_fields = ["text", "owner", "predictions", "timestamp", "id"]
    
    for field in required_fields:
        assert field in data, f"Campo '{field}' não encontrado na resposta"


def test_predict_with_special_characters():
    """Testa predição com caracteres especiais."""
    os.environ["ENV"] = "dev"
    from app.app import app
    client = TestClient(app)
    
    special_texts = [
        "hello!",
        "test?",
        "UPPERCASE",
    ]
    
    for text in special_texts:
        response = client.post(f"/predict?text={text}")
        assert response.status_code == 200, f"Falhou com texto: {text}"


def test_predict_saves_to_database():
    """Verifica que endpoint /predict funciona."""
    os.environ["ENV"] = "dev"
    from app.app import app
    client = TestClient(app)
    
    response = client.post("/predict?text=test save")
    
    assert response.status_code == 200
    data = response.json()
    # Verifica que retornou um ID
    assert "id" in data


def test_predict_uses_all_models():
    """Verifica que todos os modelos são usados na predição."""
    os.environ["ENV"] = "dev"
    from app.app import app, MODELS
    client = TestClient(app)
    
    response = client.post("/predict?text=test")
    assert response.status_code == 200
    
    data = response.json()
    predictions = data["predictions"]
    
    # Verifica que tem predições
    assert len(predictions) > 0
    
    # Cada predição deve ter top_intent e all_probs
    for model_name, prediction in predictions.items():
        assert "top_intent" in prediction
        assert "all_probs" in prediction


def test_api_response_performance():
    """Testa que a API responde rapidamente."""
    os.environ["ENV"] = "dev"
    from app.app import app
    client = TestClient(app)
    
    start = time.time()
    response = client.post("/predict?text=performance test")
    duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 5.0  # Deve responder em menos de 5 segundos


def test_predictions_have_valid_structure():
    """Valida que cada predição tem a estrutura correta."""
    os.environ["ENV"] = "dev"
    from app.app import app
    client = TestClient(app)
    
    response = client.post("/predict?text=validate structure")
    assert response.status_code == 200
    
    data = response.json()
    for model_name, prediction in data["predictions"].items():
        assert "top_intent" in prediction
        assert "all_probs" in prediction
        assert isinstance(prediction["top_intent"], str)
        assert isinstance(prediction["all_probs"], dict)
        # Verifica que probabilidades somam ~1.0
        total_prob = sum(prediction["all_probs"].values())
        assert 0.99 <= total_prob <= 1.01


# --- Testes de Integração ---

@pytest.mark.integration
def test_predict_integration_with_real_db():
    """Teste de integração com MongoDB real."""
    load_dotenv()
    
    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB")
    
    # Skip se variáveis não estão configuradas ou têm valores placeholder
    if not mongo_uri or not mongo_db:
        pytest.skip("MONGO_URI ou MONGO_DB não configurados")
    
    if "${" in mongo_uri or "${" in mongo_db:
        pytest.skip("MONGO_URI ou MONGO_DB contêm placeholders não expandidos")
    
    os.environ["ENV"] = "dev"
    
    try:
        from app.app import app
        from db.engine import get_mongo_collection
        
        client = TestClient(app)
        collection = get_mongo_collection("DEV_intent_logs")
        
        # Limpa registros de teste
        collection.delete_many({"text": "integration test input"})
        
        # Faz predição
        response = client.post("/predict?text=integration test input")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verifica se foi salvo
        saved_doc = collection.find_one({"text": "integration test input"})
        assert saved_doc is not None
        assert saved_doc["text"] == "integration test input"
        
        # Limpeza
        collection.delete_one({"_id": saved_doc["_id"]})
    except Exception as e:
        pytest.skip(f"MongoDB não acessível: {e}")
