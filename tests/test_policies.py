import pytest

from emporio.policies import PolicyIndex, load_chunks


@pytest.fixture(scope="module")
def index():
    return PolicyIndex()


def test_chunks_keep_heading_path():
    sections = [c["section"] for c in load_chunks()]
    assert "4. Trocas e devoluções > 4.1 Direito de arrependimento (compras online)" in sections


@pytest.mark.parametrize("query,expected", [
    ("posso devolver por arrependimento", "4.1"),
    ("quantas parcelas no cartão de crédito", "3."),
    ("horário de funcionamento sábado", "2."),
    ("prazo de entrega para outras cidades Jadlog", "5.2"),
    ("PIX desconto promoção cumulativa", "6.2"),
    ("garantia legal 90 dias", "8."),
])
def test_top_result_is_relevant(index, query, expected):
    top = index.search(query, top_k=2)
    assert top, query
    assert any(expected in r["section"] for r in top), (query, [r["section"] for r in top])


def test_no_match_returns_empty(index):
    assert index.search("zzzzqqq") == []
