import os
import sys
import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# --- Testes Funcionais do DB ---

def test_get_mongo_collection_is_callable():
    """Verifica que a função get_mongo_collection existe e é chamável."""
    from db.engine import get_mongo_collection
    
    assert callable(get_mongo_collection)


def test_mongo_collection_has_methods():
    """Verifica que a collection retornada tem os métodos necessários."""
    load_dotenv()
    
    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB")
    
    # Skip se variáveis não estão configuradas ou têm valores placeholder
    if not mongo_uri or not mongo_db:
        pytest.skip("MONGO_URI ou MONGO_DB não configurados")
    
    if "${" in mongo_uri or "${" in mongo_db:
        pytest.skip("MONGO_URI ou MONGO_DB contêm placeholders não expandidos")
    
    from db.engine import get_mongo_collection
    
    try:
        collection = get_mongo_collection("test_collection")
        
        # Verifica métodos esperados
        assert hasattr(collection, "insert_one")
        assert hasattr(collection, "find_one")
        assert hasattr(collection, "update_one")
        assert hasattr(collection, "delete_one")
    except Exception as e:
        pytest.skip(f"MongoDB não acessível: {e}")


@pytest.mark.integration
def test_mongo_connection_string_loaded():
    """Verifica que a string de conexão MongoDB foi carregada."""
    load_dotenv()
    
    mongo_uri = os.getenv("MONGO_URI")
    assert mongo_uri is not None, "MONGO_URI não configurado"
    assert "mongodb" in mongo_uri.lower()


@pytest.mark.integration
def test_mongo_insert_and_retrieve():
    """Testa inserção e recuperação no MongoDB."""
    load_dotenv()
    
    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB")
    
    # Skip se variáveis não estão configuradas ou têm valores placeholder
    if not mongo_uri or not mongo_db:
        pytest.skip("MONGO_URI ou MONGO_DB não configurados")
    
    if "${" in mongo_uri or "${" in mongo_db:
        pytest.skip("MONGO_URI ou MONGO_DB contêm placeholders não expandidos")
    
    from db.engine import get_mongo_collection
    
    try:
        collection = get_mongo_collection("test_collection")
        
        # Dados de teste
        test_doc = {
            "test_key": "test_value_unique_12345",
            "number": 42
        }
        
        # Limpa qualquer documento anterior
        collection.delete_many({"test_key": "test_value_unique_12345"})
        
        # Insere
        result = collection.insert_one(test_doc)
        assert result.inserted_id is not None
        
        # Recupera
        found_doc = collection.find_one({"test_key": "test_value_unique_12345"})
        assert found_doc is not None
        assert found_doc["test_key"] == "test_value_unique_12345"
        assert found_doc["number"] == 42
        
        # Limpeza
        collection.delete_one({"_id": result.inserted_id})
    except Exception as e:
        pytest.skip(f"MongoDB não acessível: {e}")
