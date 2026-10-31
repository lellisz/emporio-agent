"""Loading and cleaning of the store's CSV tables.

All cleaning is deterministic and done once at import time of the store object.
Known inconsistencies in the raw data are *reported* by ``data_issues()`` instead
of being silently "fixed" (see README, "Data inconsistencies").
"""
from __future__ import annotations

import csv
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Statuses accepted by the catalog; anything else is treated as unavailable.
PRODUCT_STATUSES = {"active", "discontinued", "coming_soon"}


def normalize(text: str) -> str:
    """Lowercase and strip accents, for accent-insensitive matching."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


def digits(text: str) -> str:
    return "".join(c for c in (text or "") if c.isdigit())


def _read(name: str) -> list[dict[str, str]]:
    with open(DATA_DIR / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@dataclass
class Store:
    products: dict[int, dict] = field(default_factory=dict)
    categories: dict[int, dict] = field(default_factory=dict)
    promotions: list[dict] = field(default_factory=list)
    customers: dict[int, dict] = field(default_factory=dict)
    orders: dict[int, dict] = field(default_factory=dict)

    def active_promotion(self, product_id: int) -> dict | None:
        """Return the active promotion for a product (highest discount wins)."""
        active = [p for p in self.promotions if p["product_id"] == product_id and p["is_active"]]
        return max(active, key=lambda p: p["discount_percent"], default=None)


def load_store(data_dir: Path | None = None) -> Store:
    global DATA_DIR
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    store = Store()

    for row in _read("categories.csv"):
        store.categories[int(row["category_id"])] = {
            "category_id": int(row["category_id"]),
            "name": row["name"].strip(),
            "description": row["description"].strip(),
        }

    for row in _read("products.csv"):
        status = row["status"].strip().lower()
        if status not in PRODUCT_STATUSES:
            status = "unknown"
        try:
            specs = json.loads(row["specs"]) if row["specs"].strip() else {}
        except json.JSONDecodeError:
            specs = {}
        pid = int(row["product_id"])
        store.products[pid] = {
            "product_id": pid,
            "name": row["name"].strip(),
            "price_brl": round(float(row["price_brl"]), 2),
            "category_id": int(row["category_id"]),
            "category": store.categories[int(row["category_id"])]["name"],
            "description": row["description"].strip(),
            "stock_quantity": max(0, int(row["stock_quantity"])),
            "status": status,
            "specs": {k: str(v).strip() for k, v in specs.items()},
            "created_at": row["created_at"],
        }

    for row in _read("promotions.csv"):
        store.promotions.append({
            "promotion_id": int(row["promotion_id"]),
            "product_id": int(row["product_id"]),
            "discount_percent": float(row["discount_percent"]),
            "description": row["description"].strip(),
            "is_active": row["is_active"].strip() == "1",
        })

    for row in _read("customers.csv"):
        cid = int(row["customer_id"])
        store.customers[cid] = {
            "customer_id": cid,
            "name": row["name"].strip(),
            "phone": row["phone"].strip(),
            "email": row["email"].strip().lower(),
            "city": row["city"].strip(),
        }

    items: dict[int, list[dict]] = {}
    for row in _read("order_items.csv"):
        items.setdefault(int(row["order_id"]), []).append(
            {"product_id": int(row["product_id"]), "quantity": int(row["quantity"])}
        )

    for row in _read("orders.csv"):
        oid = int(row["order_id"])
        store.orders[oid] = {
            "order_id": oid,
            "customer_id": int(row["customer_id"]),
            "order_date": row["order_date"],
            "status": row["status"].strip().lower(),
            "total_brl": round(float(row["total_brl"]), 2),
            "payment_method": row["payment_method"].strip().lower(),
            "tracking_code": row["tracking_code"].strip() or None,
            "estimated_delivery": row["estimated_delivery"].strip() or None,
            "notes": row["notes"].strip() or None,
            "items": items.get(oid, []),
        }
    return store


def data_issues(store: Store) -> list[str]:
    """Detect inconsistencies in the raw data (used by tests and the README)."""
    issues: list[str] = []
    for o in store.orders.values():
        computed = round(sum(store.products[i["product_id"]]["price_brl"] * i["quantity"]
                             for i in o["items"]), 2)
        if abs(computed - o["total_brl"]) > 0.01:
            issues.append(f"order {o['order_id']}: total {o['total_brl']} != items sum {computed}")
    for p in store.products.values():
        if p["status"] == "active" and p["stock_quantity"] == 0:
            issues.append(f"product {p['product_id']} is active but has zero stock")
        if p["status"] != "active" and p["stock_quantity"] > 0:
            issues.append(f"product {p['product_id']} is {p['status']} but has stock {p['stock_quantity']}")
    return issues
