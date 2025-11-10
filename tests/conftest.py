"""
Configurações compartilhadas para todos os testes.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def pytest_configure(config):
    """Configura markers personalizados."""
    config.addinivalue_line(
        "markers", "integration: marca testes de integração (requerem DB real)"
    )
    config.addinivalue_line(
        "markers", "slow: marca testes lentos"
    )

@pytest.fixture(scope="function", autouse=True)
def reset_environment():
    """Reseta o ambiente antes de cada teste."""
    # Salva ENV original
    original_env = os.environ.get("ENV")
    
    # Define padrão como dev
    os.environ["ENV"] = "dev"
    
    yield
    
    # Restaura ENV original ou mantém dev
    if original_env:
        os.environ["ENV"] = original_env
    
    # Limpa módulos importados para evitar cache entre testes
    modules_to_reload = [
        "app.app",
        "app.auth",
        "db.engine"
    ]
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            del sys.modules[module_name]


@pytest.fixture(scope="session")
def test_data_paths():
    """Fornece caminhos para dados de teste."""
    test_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(test_dir, ".."))
    
    return {
        "project_root": project_root,
        "test_dir": test_dir,
        "data_dir": os.path.join(project_root, "intent_classifier", "data"),
        "models_dir": os.path.join(project_root, "intent_classifier", "models"),
    }