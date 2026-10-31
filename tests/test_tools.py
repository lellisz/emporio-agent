import json
from datetime import datetime

import pytest

from emporio.tools import CAMPO_GRANDE_TZ, Tools


def at(y, m, d, hh=10, mm=0):
    return lambda: datetime(y, m, d, hh, mm, tzinfo=CAMPO_GRANDE_TZ)


@pytest.fixture
def tools(store, tmp_path):
    return Tools(store=store, now=at(2026, 3, 14), sessions_dir=tmp_path)  # a Saturday, 10:00


def test_price_without_promotion_has_pix(tools):
    p = tools.get_product("Takamine GD20")["products"][0]
    assert p["final_price_brl"] == 2199.0
    assert p["pix_price_brl"] == 2089.05
    assert p["promotion"] is None


def test_promotion_is_not_cumulative_with_pix(tools):
    p = tools.get_product("Ohana CK-20")["products"][0]
    assert p["promotion"]["discount_percent"] == 20
    assert p["final_price_brl"] == 439.2
    assert p["pix_price_brl"] is None


def test_inactive_promotion_ignored(tools):
    assert tools.get_product("93")["products"][0]["promotion"] is None  # promos 6 and 23 are inactive


def test_unknown_product(tools):
    assert "error" in tools.get_product("Stradivarius")


def test_out_of_stock_and_discontinued_flags(tools):
    assert tools.get_product("Giannini GF-3D")["products"][0]["availability"] == "out_of_stock"
    assert tools.get_product("Shelby SN-7C")["products"][0]["availability"] == "discontinued"
    assert tools.get_product("Kala KA-ABP-CTG")["products"][0]["availability"] == "coming_soon"


def test_search_guitars_under_1000(tools):
    r = tools.search_products(query="violão", max_price=1000)
    names = [p["name"] for p in r["products"]]
    assert r["total_matches"] >= 8
    assert all(p["final_price_brl"] <= 1000 for p in r["products"])
    prices = [p["final_price_brl"] for p in r["products"]]
    assert prices == sorted(prices)
    assert "Shelby SN-7C 7 Cordas Nylon Natural" not in names  # discontinued


def test_search_filters_and_bad_category(tools):
    r = tools.search_products(category="ukuleles", max_price=400)
    assert r["products"] and all(p["category"] == "Ukuleles" for p in r["products"])
    assert "error" in tools.search_products(category="harps")


def test_search_max_price_uses_promotional_price(tools):
    # Ohana CK-20: list 549, promo price 439.20
    names = [p["name"] for p in tools.search_products(query="ohana ck-20", max_price=450)["products"]]
    assert names == ["Ohana CK-20 Concert Natural"]


@pytest.mark.parametrize("amount,max_n", [(2199, 12), (400, 5), (150, 3), (45, 1), (1000, 10)])
def test_installment_limits(tools, amount, max_n):
    assert tools.calculate_installments(amount)["max_installments"] == max_n


def test_installment_tiers(tools):
    plans = {p["installments"] for p in tools.calculate_installments(500)["plans"]}
    assert 6 in plans and 7 not in plans  # 500/6 = 83.33 >= 80; 500/7 = 71.4 < 100


def test_shipping_metro(tools):
    assert tools.shipping_quote("Campo Grande", 499.9)["shipping_fee_brl"] == 35.0
    assert tools.shipping_quote("campo grande", 500.01)["shipping_fee_brl"] == 0.0


def test_shipping_other_city_has_no_invented_price(tools):
    r = tools.shipping_quote("Dourados", 900, product_ids=[139])
    assert r["shipping_fee_brl"] is None
    assert len(r["options"]) == 3
    assert r["special_freight_items"]


def test_store_open_saturday_morning(tools):
    assert tools.store_info()["open_now"] is True


@pytest.mark.parametrize("when,expected", [
    (at(2026, 3, 14, 14), False),   # Saturday 14:00
    (at(2026, 3, 15, 10), False),   # Sunday
    (at(2026, 3, 16, 8, 59), False),
    (at(2026, 3, 16, 17, 59), True),
    (at(2026, 3, 16, 18, 0), False),
])
def test_store_hours(store, tmp_path, when, expected):
    assert Tools(store=store, now=when, sessions_dir=tmp_path).store_info()["open_now"] is expected


def test_order_requires_matching_contact(tools):
    assert "error" in tools.lookup_order(8, "someone@else.com")
    assert "error" in tools.lookup_order(8, "")
    assert "error" in tools.lookup_order(999, "anacarol.ferreira@coldmail.com")


def test_order_lookup_by_email_or_phone(tools):
    by_mail = tools.lookup_order(8, "AnaCarol.Ferreira@coldmail.com")
    by_phone = tools.lookup_order(8, "67 99654-7821")
    assert by_mail == by_phone
    assert by_mail["status"] == "shipped" and by_mail["tracking_code"] == "BRJL5544332BR"
    assert "email" not in json.dumps(by_mail) and by_mail["customer_first_name"] == "Ana"


def test_return_regret_inside_and_outside_window(store, tmp_path):
    t = Tools(store=store, now=at(2026, 2, 20), sessions_dir=tmp_path)
    ok = t.check_return_eligibility(7, "leticia.rocha@jmail.com", "regret")  # delivered est. 2026-02-17
    assert ok["eligible"] is True and ok["days_since_receipt"] == 3
    late = Tools(store=store, now=at(2026, 3, 1), sessions_dir=tmp_path)
    assert late.check_return_eligibility(7, "leticia.rocha@jmail.com", "regret")["eligible"] is False


def test_return_defect_uses_30_days_from_purchase(store, tmp_path):
    t = Tools(store=store, now=at(2026, 2, 20), sessions_dir=tmp_path)
    r = t.check_return_eligibility(7, "leticia.rocha@jmail.com", "defect")  # bought 2026-02-03
    assert r["eligible"] is True and r["legal_warranty_active"] is True


def test_return_not_delivered_or_cancelled(tools):
    pending = tools.check_return_eligibility(13, "fernanda.ribeiro@coldmail.com", "regret")
    assert pending["eligible"] is None
    cancelled = tools.check_return_eligibility(17, "matheus.correia@hayoo.com.br", "regret")
    assert cancelled["eligible"] is False


def test_return_requires_verification(tools):
    assert "error" in tools.check_return_eligibility(7, "wrong@x.com", "regret")


def test_escalation_writes_ticket(tools, tmp_path):
    assert tools.escalate_to_human("complaint", "late delivery")["ticket"] == 1
    assert tools.escalate_to_human("complaint", "again")["ticket"] == 2
    assert len((tmp_path / "escalations.jsonl").read_text().splitlines()) == 2


def test_run_never_raises(tools):
    assert "unknown tool" in tools.run("nope", {})
    assert "invalid arguments" in tools.run("get_product", {})
    assert "invalid arguments" in tools.run("get_product", {"bogus": 1, "product": "x"})
    assert "arguments must be" in tools.run("get_product", [1])
    assert json.loads(tools.run("get_product", {"product": "Takamine GD20"}))["products"]
