#!/bin/bash
# filepath: test.sh

set -e  # Para na primeira falha

echo "🧪 Rodando testes no container..."

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Verifica se o container está rodando
if ! docker compose ps backend | grep -q "Up"; then
    echo -e "${YELLOW}⚠️  Container não está rodando. Iniciando...${NC}"
    docker compose up -d backend
    sleep 8
    echo -e "${GREEN}✅ Container iniciado${NC}"
fi

# Verifica saúde do container
echo -e "${YELLOW}🔍 Verificando saúde do container...${NC}"
if docker compose exec backend python -c "print('OK')" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Container está saudável${NC}"
else
    echo -e "${RED}❌ Container não está respondendo${NC}"
    exit 1
fi

# Executa os testes
echo -e "${YELLOW}🧪 Executando testes unitários...${NC}"
docker compose exec backend pytest tests/ -v -m "not integration" \
    --cov=app --cov=db \
    --cov-report=term \
    --cov-report=html \
    --tb=short \
    --color=yes

exit_code=$?

echo ""
echo "═════════════════════════════════════════════════════"
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✅ SUCESSO: Todos os testes passaram!${NC}"
    echo -e "${YELLOW}📊 Relatório de cobertura HTML: htmlcov/index.html${NC}"
    echo -e "${YELLOW}🔍 Para ver no navegador: open htmlcov/index.html${NC}"
else
    echo -e "${RED}❌ FALHA: Alguns testes falharam (código: $exit_code)${NC}"
    echo -e "${YELLOW}💡 Dica: Rode com -vv para mais detalhes${NC}"
fi
echo "═════════════════════════════════════════════════════"

exit $exit_code
