"""Deterministic tools the agent can call.

Every number the customer sees (prices, promotions, installments, shipping,
return windows, store hours) is computed here, never by the LLM. Tool output
keys/values may be in Portuguese; the system prompt tells the model to translate
them faithfully.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .data import Store, digits, load_store, normalize
from .policies import PolicyIndex

PIX_DISCOUNT = 0.05
METRO_FREE_SHIPPING_ABOVE = 500.00
METRO_FLAT_FEE = 35.00
REGRET_DAYS = 7           # manual 4.1
PREFERENCE_DAYS = 7       # manual 4.3
DEFECT_DAYS = 30          # manual 4.2
LEGAL_WARRANTY_DAYS = 90  # manual 8.1
CAMPO_GRANDE_TZ = timezone(timedelta(hours=-4))  # America/Campo_Grande (no DST)

STORE_INFO = {
    "name": "Empório da Música",
    "address": "Rua 14 de Maio, 3200 — Centro, Campo Grande - MS, CEP 79202-333",
    "store_phone": "(67) 3341-4444",
    "whatsapp": "(67) 3321-4500",
    "email": "contato@emporiodamusica.com.br",
    "hours": {"segunda a sexta": "09:00 às 18:00", "sábado": "09:00 às 13:00",
              "domingo e feriados": "fechado"},
    "note": "Em datas especiais (Black Friday, Aniversário da Loja, Natal) o horário pode ser estendido.",
}

CATEGORY_IDS = {
    "electric_guitars": 1, "basses": 2, "drums_percussion": 3, "keyboards_pianos": 4,
    "acoustic_guitars": 5, "woodwinds": 6, "brass": 7, "orchestral_strings": 8, "ukuleles": 9,
}
# Light EN->PT synonyms so an English query still hits the Portuguese catalog.
SYNONYMS = {"acoustic": "violao", "guitar": "guitarra", "bass": "baixo", "drum": "bateria",
            "drums": "bateria", "keyboard": "teclado", "piano": "teclado", "steel": "aco",
            "classical": "classico", "nylon": "nylon", "electric": "eletrico", "sunburst": "sunburst"}
# Categories that need an individually quoted freight outside the metro area (manual 5.2).
SPECIAL_FREIGHT_CATEGORIES = {3, 4, 8}


def _money(x: float) -> float:
    return round(x + 1e-9, 2)


def price_info(store: Store, product: dict) -> dict:
    """Price breakdown. PIX 5% is never stacked on promotional prices (manual 6.2)."""
    promo = store.active_promotion(product["product_id"])
    list_price = product["price_brl"]
    if promo:
        final = _money(list_price * (1 - promo["discount_percent"] / 100))
        return {"list_price_brl": list_price, "promotion": {
                    "name": promo["description"], "discount_percent": promo["discount_percent"]},
                "final_price_brl": final, "pix_price_brl": None,
                "pix_note": "PIX discount does not apply to promotional prices"}
    return {"list_price_brl": list_price, "promotion": None, "final_price_brl": list_price,
            "pix_price_brl": _money(list_price * (1 - PIX_DISCOUNT)),
            "pix_note": "5% PIX discount on the list price"}


def availability(product: dict) -> str:
    if product["status"] == "discontinued":
        return "discontinued"
    if product["status"] == "coming_soon":
        return "coming_soon"
    if product["status"] != "active":
        return "unavailable"
    return "in_stock" if product["stock_quantity"] > 0 else "out_of_stock"


def _summary(store: Store, p: dict) -> dict:
    return {"product_id": p["product_id"], "name": p["name"], "category": p["category"],
            "availability": availability(p), **price_info(store, p)}


class Tools:
    """Holds shared state (store, policy index, clock) and exposes the tool functions."""

    def __init__(self, store: Store | None = None, index: PolicyIndex | None = None,
                 now: Callable[[], datetime] | None = None, sessions_dir: Path | None = None):
        self.store = store or load_store()
        self.index = index or PolicyIndex()
        self._now = now or (lambda: datetime.now(CAMPO_GRANDE_TZ))
        self.sessions_dir = sessions_dir or Path("sessions")

    # ---- catalog -----------------------------------------------------------
    def search_products(self, query: str = "", category: str | None = None,
                        max_price: float | None = None, min_price: float | None = None,
                        string_type: str | None = None, electronics: bool | None = None,
                        limit: int = 8) -> dict:
        cat_id = CATEGORY_IDS.get(category) if category else None
        if category and cat_id is None:
            return {"error": f"unknown category '{category}'", "valid": sorted(CATEGORY_IDS)}
        tokens = [SYNONYMS.get(t, t) for t in normalize(query).split()]
        results = []
        for p in self.store.products.values():
            if p["status"] != "active":  # discontinued / coming soon are not sellable
                continue
            if cat_id and p["category_id"] != cat_id:
                continue
            haystack = normalize(" ".join([p["name"], p["description"], p["category"], *p["specs"].values()]))
            if not all(t in haystack for t in tokens):
                continue
            info = price_info(self.store, p)
            price = info["final_price_brl"]
            if max_price is not None and price > max_price:
                continue
            if min_price is not None and price < min_price:
                continue
            if string_type and p["specs"].get("strings") != string_type:
                continue
            if electronics is not None and (p["specs"].get("electronics") == "yes") != electronics:
                continue
            results.append((price, _summary(self.store, p)))
        results.sort(key=lambda r: r[0])
        limit = max(1, min(int(limit), 15))
        return {"total_matches": len(results), "shown": min(limit, len(results)),
                "products": [r[1] for r in results[:limit]],
                "note": "Only active (sellable) products are listed; out-of-stock ones are flagged."}

    def get_product(self, product: str) -> dict:
        ref = str(product).strip()
        found: list[dict] = []
        if ref.isdigit() and int(ref) in self.store.products:
            found = [self.store.products[int(ref)]]
        else:
            toks = normalize(ref).split()
            found = [p for p in self.store.products.values()
                     if toks and all(t in normalize(p["name"]) for t in toks)]
        if not found:
            return {"error": f"no product matches '{ref}'"}
        if len(found) > 5:
            return {"error": "too many matches, ask the customer to be more specific",
                    "candidates": [p["name"] for p in found[:10]]}
        out = []
        for p in found:
            item = _summary(self.store, p)
            item.update({"description": p["description"], "specs": p["specs"]})
            if p["status"] == "active":
                item["stock_quantity"] = p["stock_quantity"]
            out.append(item)
        return {"products": out}

    # ---- money -------------------------------------------------------------
    def calculate_installments(self, amount: float) -> dict:
        """Valid credit-card plans (manual 3.1). Per-tier minimum installment values."""
        if amount <= 0:
            return {"error": "amount must be positive"}
        plans = []
        for n in range(1, 13):
            value = _money(amount / n)
            minimum = 0 if n == 1 else 50 if n <= 3 else 80 if n <= 6 else 100
            if value >= minimum:
                plans.append({"installments": n, "installment_value_brl": value, "total_brl": _money(value * n)
                              if n > 1 else _money(amount), "interest": "none"})
        return {"amount_brl": _money(amount), "plans": plans, "max_installments": plans[-1]["installments"],
                "pix_price_brl": _money(amount * (1 - PIX_DISCOUNT)),
                "pix_note": "PIX/promotion rules: use get_product for a specific item; PIX never stacks on promotions",
                "combined_payment_allowed": amount > 2000}

    def shipping_quote(self, city: str, order_total_brl: float, product_ids: list[int] | None = None) -> dict:
        in_metro = normalize(city) == "campo grande"
        if in_metro:
            free = order_total_brl > METRO_FREE_SHIPPING_ABOVE
            return {"region": "Campo Grande metropolitan area", "shipping_fee_brl": 0.0 if free else METRO_FLAT_FEE,
                    "free_shipping": free, "delivery_time": "1 a 3 dias úteis",
                    "method": "motoboy próprio (o cliente é contactado por telefone antes)"}
        special = [self.store.products[i]["name"] for i in (product_ids or [])
                   if i in self.store.products and self.store.products[i]["category_id"] in SPECIAL_FREIGHT_CATEGORIES]
        return {"region": "outside the Campo Grande metropolitan area", "shipping_fee_brl": None,
                "fee_note": "Fee depends on destination ZIP code, weight and size; it cannot be quoted here",
                "options": [{"service": "PAC (Correios)", "delivery_time": "5 a 12 dias úteis"},
                            {"service": "SEDEX (Correios)", "delivery_time": "2 a 5 dias úteis"},
                            {"service": "Jadlog (.package)", "delivery_time": "3 a 8 dias úteis"}],
                "tracking_and_insurance": "incluídos",
                "special_freight_items": special,
                "special_freight_note": "Large instruments need an individual freight quote" if special else None}

    # ---- store / policies --------------------------------------------------
    def store_info(self) -> dict:
        now = self._now()
        wd, t = now.weekday(), now.hour * 60 + now.minute
        open_now = (wd <= 4 and 9 * 60 <= t < 18 * 60) or (wd == 5 and 9 * 60 <= t < 13 * 60)
        return {**STORE_INFO, "now_local": now.strftime("%Y-%m-%d %H:%M (%A)"), "open_now": open_now,
                "open_now_note": "Computed from the regular schedule; public holidays are not tracked"}

    def search_policies(self, query: str) -> dict:
        hits = self.index.search(query, top_k=3)
        if not hits:
            return {"results": [], "note": "No policy found. Do not invent one; offer a human agent."}
        return {"results": [{"section": h["section"], "text": h["text"]} for h in hits]}

    # ---- orders (privacy: order id + matching e-mail or phone) -------------
    def _verified_order(self, order_id: Any, contact: str) -> tuple[dict | None, dict | None]:
        try:
            order = self.store.orders.get(int(order_id))
        except (TypeError, ValueError):
            order = None
        if not order:
            return None, None
        cust = self.store.customers[order["customer_id"]]
        c = (contact or "").strip().lower()
        ok = bool(c) and (c == cust["email"] or (digits(c) and digits(c) == digits(cust["phone"])))
        return (order, cust) if ok else (None, None)

    _NOT_VERIFIED = {"error": "Order not found for the provided data. For privacy, order details are only "
                              "shared when the order number and the e-mail or phone on the order match."}

    def lookup_order(self, order_id: int, email_or_phone: str) -> dict:
        order, cust = self._verified_order(order_id, email_or_phone)
        if not order:
            return dict(self._NOT_VERIFIED)
        return {"order_id": order["order_id"], "customer_first_name": cust["name"].split()[0],
                "order_date": order["order_date"], "status": order["status"], "total_brl": order["total_brl"],
                "payment_method": order["payment_method"], "tracking_code": order["tracking_code"],
                "estimated_delivery": order["estimated_delivery"], "notes": order["notes"],
                "items": [{"name": self.store.products[i["product_id"]]["name"], "quantity": i["quantity"]}
                          for i in order["items"]]}

    def check_return_eligibility(self, order_id: int, email_or_phone: str, reason: str) -> dict:
        """reason: regret (7 days), defect (30 days exchange), preference (7 days)."""
        order, _ = self._verified_order(order_id, email_or_phone)
        if not order:
            return dict(self._NOT_VERIFIED)
        if reason not in ("regret", "defect", "preference"):
            return {"error": "reason must be one of: regret, defect, preference"}
        base = {"order_id": order["order_id"], "status": order["status"], "reason": reason}
        if order["status"] == "cancelled":
            return {**base, "eligible": False, "explanation": "Order was cancelled; there is nothing to return."}
        if order["status"] != "delivered":
            return {**base, "eligible": None, "explanation": "The order has not been delivered yet. The "
                    "right of withdrawal counts from receipt; the customer can refuse/return after delivery."}
        today = self._now().date()
        # No actual delivery date in the data: estimated delivery is used as the receipt date.
        received = date.fromisoformat(order["estimated_delivery"])
        purchased = date.fromisoformat(order["order_date"])
        days_since_receipt = (today - received).days
        base.update({"receipt_date_assumed": received.isoformat(), "days_since_receipt": days_since_receipt})
        if reason in ("regret", "preference"):
            window = REGRET_DAYS
            ok = days_since_receipt <= window
            extra = ("Item must be in original packaging, unused, with all accessories and manuals. Refund to the "
                     "original payment method within 10 business days; return shipping paid by the store."
                     if reason == "regret" else
                     "Subject to availability of the desired item; price difference charged or refunded.")
            return {**base, "eligible": ok, "window_days": window, "conditions": extra,
                    "explanation": f"{days_since_receipt} days since receipt; limit is {window} days.",
                    "if_not_eligible": None if ok else "Offer a human agent to review; manufacturer or legal warranty "
                                       "only covers defects."}
        days_since_purchase = (today - purchased).days
        in_exchange = days_since_purchase <= DEFECT_DAYS
        in_legal = days_since_receipt <= LEGAL_WARRANTY_DAYS
        return {**base, "eligible": in_exchange, "window_days": DEFECT_DAYS, "days_since_purchase": days_since_purchase,
                "legal_warranty_active": in_legal, "legal_warranty_days": LEGAL_WARRANTY_DAYS,
                "explanation": ("Within the 30-day defect exchange window." if in_exchange else
                                "Past the 30-day exchange window: the customer should use the manufacturer warranty "
                                "(the store can mediate on request)."),
                "exclusions": "Misuse, drops, humidity, unauthorized modifications and natural wear are not covered."}

    # ---- escalation ----------------------------------------------------------
    def escalate_to_human(self, reason: str, summary: str, contact: str = "") -> dict:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self.sessions_dir / "escalations.jsonl"
        ticket = sum(1 for _ in path.open(encoding="utf-8")) + 1 if path.exists() else 1
        record = {"ticket": ticket, "created_at": self._now().isoformat(), "reason": reason,
                  "summary": summary, "contact": contact}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ticket": ticket, "status": "registered", "response_time": "24 horas úteis"}

    # ---- dispatch ------------------------------------------------------------
    def run(self, name: str, args: dict) -> str:
        """Execute a tool and return a JSON string. Never raises."""
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            return json.dumps({"error": f"unknown tool '{name}'", "valid_tools": sorted(TOOL_NAMES)})
        if not isinstance(args, dict):
            return json.dumps({"error": "arguments must be a JSON object"})
        try:
            return json.dumps(fn(**args), ensure_ascii=False)
        except TypeError as exc:  # missing/unexpected argument
            return json.dumps({"error": f"invalid arguments for {name}: {exc}"})
        except Exception as exc:  # defensive: a tool bug must not crash the conversation
            return json.dumps({"error": f"tool {name} failed: {type(exc).__name__}: {exc}"})


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOL_SCHEMAS = [
    _fn("search_products",
        "Search sellable products by free text (use Portuguese words, e.g. 'violão nylon') and filters. "
        "Prices already include active promotions. Use for 'what options do you have' questions.",
        {"query": {"type": "string", "description": "Portuguese keywords, e.g. 'violão aço'"},
         "category": {"type": "string", "enum": sorted(CATEGORY_IDS)},
         "max_price": {"type": "number", "description": "Max final price in BRL"},
         "min_price": {"type": "number"},
         "string_type": {"type": "string", "enum": ["nylon", "steel"]},
         "electronics": {"type": "boolean", "description": "true = electro-acoustic only"},
         "limit": {"type": "integer"}}, []),
    _fn("get_product",
        "Get exact price, promotion, PIX price, availability, stock and specs of ONE product by name or id. "
        "Always call before stating a price or stock.",
        {"product": {"type": "string", "description": "Product name (or part of it) or numeric id"}}, ["product"]),
    _fn("calculate_installments", "Valid credit-card installment plans and PIX price for an amount in BRL.",
        {"amount": {"type": "number"}}, ["amount"]),
    _fn("shipping_quote", "Shipping fee/deadline for a destination city and order total in BRL.",
        {"city": {"type": "string"}, "order_total_brl": {"type": "number"},
         "product_ids": {"type": "array", "items": {"type": "integer"}}}, ["city", "order_total_brl"]),
    _fn("store_info", "Address, phones, e-mail, opening hours and whether the store is open right now.", {}, []),
    _fn("search_policies",
        "Search the store policy manual (returns, warranty, payment, shipping, promotions, privacy). "
        "Query in Portuguese.", {"query": {"type": "string"}}, ["query"]),
    _fn("lookup_order",
        "Order status/tracking. Requires the order number AND the e-mail or phone registered on the order.",
        {"order_id": {"type": "integer"}, "email_or_phone": {"type": "string"}}, ["order_id", "email_or_phone"]),
    _fn("check_return_eligibility",
        "Check whether a delivered order can be returned (regret, 7 days) or exchanged (defect, 30 days; "
        "preference, 7 days). Requires order number AND e-mail or phone.",
        {"order_id": {"type": "integer"}, "email_or_phone": {"type": "string"},
         "reason": {"type": "string", "enum": ["regret", "defect", "preference"]}},
        ["order_id", "email_or_phone", "reason"]),
    _fn("escalate_to_human",
        "Hand the case to a human agent (complaints, anything not covered, rule exceptions). "
        "Human follow-up within 24 business hours.",
        {"reason": {"type": "string"}, "summary": {"type": "string"}, "contact": {"type": "string"}},
        ["reason", "summary"]),
]
TOOL_NAMES = {s["function"]["name"] for s in TOOL_SCHEMAS}
