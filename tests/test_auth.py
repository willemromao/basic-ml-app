import os
import sys
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from fastapi import HTTPException, Request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_verify_token_missing_header():
    """Testa verificação sem header Authorization."""
    from app.auth import verify_token
    
    mock_request = Mock(spec=Request)
    mock_request.headers.get.return_value = None
    
    with pytest.raises(HTTPException) as exc_info:
        verify_token(mock_request)
    
    assert exc_info.value.status_code == 401


def test_verify_token_invalid():
    """Testa verificação com token inválido."""
    from app.auth import verify_token
    
    mock_request = Mock(spec=Request)
    mock_request.headers.get.return_value = "Bearer invalid-token"
    
    with patch("app.auth.get_mongo_collection") as mock_get_coll:
        mock_coll = Mock()
        mock_coll.find_one.return_value = None
        mock_get_coll.return_value = mock_coll
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(mock_request)
        
        assert exc_info.value.status_code == 403


def test_verify_token_expired():
    """Testa verificação com token expirado."""
    from app.auth import verify_token
    
    mock_request = Mock(spec=Request)
    mock_request.headers.get.return_value = "Bearer expired-token"
    
    expired_token_data = {
        "token": "expired-token",
        "owner": "test_user",
        "active": True,
        "expires_at": datetime.utcnow() - timedelta(days=1)
    }
    
    with patch("app.auth.get_mongo_collection") as mock_get_coll:
        mock_coll = Mock()
        mock_coll.find_one.return_value = expired_token_data
        mock_get_coll.return_value = mock_coll
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(mock_request)
        
        assert exc_info.value.status_code == 403
        assert "expired" in str(exc_info.value.detail).lower()


def test_verify_token_valid():
    """Testa verificação com token válido."""
    from app.auth import verify_token
    
    mock_request = Mock(spec=Request)
    mock_request.headers.get.return_value = "Bearer valid-token"
    
    valid_token_data = {
        "token": "valid-token",
        "owner": "test_user",
        "active": True,
        "expires_at": datetime.utcnow() + timedelta(days=30)
    }
    
    with patch("app.auth.get_mongo_collection") as mock_get_coll:
        mock_coll = Mock()
        mock_coll.find_one.return_value = valid_token_data
        mock_get_coll.return_value = mock_coll
        
        owner = verify_token(mock_request)
        assert owner == "test_user"


def test_token_manager_create():
    """Testa criação de token."""
    from app.auth import TokenManager
    
    with patch("app.auth.get_mongo_collection") as mock_get_coll:
        mock_coll = Mock()
        mock_get_coll.return_value = mock_coll
        
        manager = TokenManager()
        manager.create(owner="test_user", expires_in_days=30)
        
        assert mock_coll.insert_one.called
        call_args = mock_coll.insert_one.call_args[0][0]
        assert call_args["owner"] == "test_user"
        assert call_args["active"] is True


def test_token_manager_delete_expired():
    """Testa remoção de tokens expirados."""
    from app.auth import TokenManager
    
    with patch("app.auth.get_mongo_collection") as mock_get_coll:
        mock_coll = Mock()
        mock_coll.delete_many.return_value = Mock(deleted_count=3)
        mock_get_coll.return_value = mock_coll
        
        manager = TokenManager()
        manager.delete_expired()
        
        assert mock_coll.delete_many.called


@pytest.mark.integration
def test_token_full_lifecycle():
    """Teste de integração: ciclo completo de token."""
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("MONGO_URI"):
        pytest.skip("MONGO_URI não configurado")
    
    from app.auth import TokenManager, verify_token
    from db.engine import get_mongo_collection
    
    manager = TokenManager()
    tokens_coll = get_mongo_collection("api_tokens")
    
    # Cria token
    test_owner = f"test_{datetime.utcnow().timestamp()}"
    manager.create(owner=test_owner, expires_in_days=1)
    
    # Busca token
    token_doc = tokens_coll.find_one({"owner": test_owner})
    assert token_doc is not None
    
    # Valida token
    mock_request = Mock(spec=Request)
    mock_request.headers.get.return_value = f"Bearer {token_doc['token']}"
    
    owner = verify_token(mock_request)
    assert owner == test_owner
    
    # Limpeza
    tokens_coll.delete_one({"_id": token_doc["_id"]})
