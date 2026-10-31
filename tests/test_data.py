from emporio.data import data_issues, digits, normalize


def test_tables_load(store):
    assert len(store.products) == 65
    assert len(store.customers) == 50
    assert len(store.orders) == 20


def test_prices_are_floats_and_stock_ints(store):
    p = store.products[95]
    assert p["name"].startswith("Takamine GD20")
    assert p["price_brl"] == 2199.0
    assert isinstance(p["stock_quantity"], int)
    assert p["specs"]["strings"] == "steel"


def test_only_active_promotions_apply(store):
    assert store.active_promotion(95) is None  # promo 4 is inactive
    assert store.active_promotion(121)["discount_percent"] == 20


def test_data_issues_reported_not_hidden(store):
    issues = data_issues(store)
    assert any("order 3" in i for i in issues)
    assert any("product 96" in i and "zero stock" in i for i in issues)


def test_helpers():
    assert normalize("Violão Aço") == "violao aco"
    assert digits("(67) 99812-3456") == "67998123456"
