"""
etl_api_to_csv_products.py - Product Catalog ETL
════════════════════════════════════

Use case
--------
Nightly job that pulls the full product catalog from DummyJSON REST API,
enriches each product with business labels (price tier, rating label), and
stores the result as a clean CSV in S3 for downstream analytics / BI tools.

Pipeline
--------
DummyJSON API -> transform -> S3 CSV (products/processed/catalog.csv)

Config JSON stored in S3 (loaded via --CONFIG_PATH arg):
--------------------------------------------------------
{
  "API_URL": "https://dummyjson.com/products?limit=0",
  "OUTPUT_BUCKET_NAME": "your-output-bucket",
  "OUTPUT_PREFIX": "products/processed",
  "OUTPUT_FILE_NAME": "catalog.csv"
}

DummyJSON vs FakeStore — differences handled here:
  - Response is wrapped: { "products": [...], "total": N }  → extract ["products"]
  - rating is a direct float (not nested {"rate": x, "count": y})
  - image field is called "thumbnail"
  - ?limit=0 returns all products (194 total)
"""

import io
import json
import logging
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

import boto3
import pandas as pd
import requests

try:
    from awsglue.utils import getResolvedOptions
except ImportError:
    def getResolvedOptions(argv: list, options: list) -> dict:
        import argparse
        parser = argparse.ArgumentParser()
        for opt in options:
            parser.add_argument(f"--{opt}")
        args, _ = parser.parse_known_args(argv[1:])
        return vars(args)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger("first_etl")


# ─────────────────────────────────────────────
# ARGS & CONFIG
# ─────────────────────────────────────────────

def get_args() -> dict:
    return getResolvedOptions(sys.argv, ["CONFIG_PATH"])


def load_config(config_path: str) -> dict:
    logger.info("Loading config from %s", config_path)

    parsed = urlparse(config_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)

    return json.loads(response["Body"].read().decode("utf-8"))


# ─────────────────────────────────────────────
# FETCH  ← CORRIGÉ POUR DUMMYJSON
# ─────────────────────────────────────────────

def fetch_products(api_url: str) -> list[dict]:
    """
    DummyJSON renvoie :
      { "products": [...], "total": 194, "skip": 0, "limit": 0 }

    On extrait uniquement la liste "products".
    Utiliser ?limit=0 dans l'URL pour récupérer TOUS les produits.
    """
    logger.info("Fetching products from %s", api_url)

    response = requests.get(api_url, timeout=30)
    response.raise_for_status()

    payload = response.json()

    # ← différence clé : DummyJSON wrappe dans {"products": [...]}
    if isinstance(payload, dict) and "products" in payload:
        products = payload["products"]
        logger.info("DummyJSON response → %d products (total: %s)", len(products), payload.get("total"))
    elif isinstance(payload, list):
        # compatibilité FakeStore au cas où
        products = payload
        logger.info("Direct list response → %d products", len(products))
    else:
        raise ValueError(f"Unexpected API response format: {type(payload)}")

    return products


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _price_tier(price: float) -> str:
    if price < 20:
        return "budget"
    if price <= 100:
        return "mid-range"
    return "premium"


def _rating_label(rate: float) -> str:
    if rate < 3.0:
        return "low"
    if rate < 4.0:
        return "medium"
    return "high"


# ─────────────────────────────────────────────
# TRANSFORM  ← CORRIGÉ POUR DUMMYJSON
# ─────────────────────────────────────────────

def transform_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaptations DummyJSON :
      - rating  : colonne directe float  (pas nested rating.rate / rating.count)
      - stock   : remplace rating.count  (disponibilité produit)
      - thumbnail / images : supprimées  (comme image dans FakeStore)
      - Colonnes spécifiques DummyJSON supprimées : reviews, meta, dimensions...
    """
    logger.info("Transforming %d rows", len(df))

    # ── renommage ──────────────────────────────────────────────
    rename_map = {"id": "product_id"}

    # DummyJSON : rating est direct (float), pas nested
    # FakeStore : rating.rate / rating.count (après json_normalize)
    if "rating" in df.columns:
        rename_map["rating"] = "rating_rate"
    elif "rating.rate" in df.columns:
        rename_map["rating.rate"] = "rating_rate"
        if "rating.count" in df.columns:
            rename_map["rating.count"] = "rating_count"

    if "stock" in df.columns:
        rename_map["stock"] = "rating_count"   # stock joue le rôle de count

    df = df.rename(columns=rename_map)

    # ── drop des colonnes inutiles ─────────────────────────────
    cols_to_drop = [
        "thumbnail", "images", "image",        # URLs images
        "reviews", "meta",                      # objets imbriqués DummyJSON
        "dimensions", "tags",                   # listes / objets
        "warrantyInformation", "shippingInformation",
        "returnPolicy", "availabilityStatus",
        "minimumOrderQuantity", "sku", "weight",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # ── nettoyage ──────────────────────────────────────────────
    critical = [c for c in ("price", "category", "rating_rate") if c in df.columns]
    df = df.dropna(subset=critical)

    df["category"] = df["category"].str.strip().str.lower()

    # ── enrichissement ─────────────────────────────────────────
    df["price_tier"]    = df["price"].apply(_price_tier)
    df["rating_label"]  = df["rating_rate"].apply(_rating_label)
    df["ingestion_timestamp"] = datetime.now(timezone.utc).isoformat()

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_to_s3(df: pd.DataFrame, bucket: str, key: str) -> str:
    logger.info("Uploading to s3://%s/%s", bucket, key)

    buf = io.StringIO()
    df.to_csv(buf, index=False)

    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )

    return f"s3://{bucket}/{key}"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    logger.info("START ETL")

    args   = get_args()
    config = load_config(args["CONFIG_PATH"])

    api_url       = config["API_URL"]            # https://dummyjson.com/products?limit=0
    output_bucket = config["OUTPUT_BUCKET_NAME"]
    output_prefix = config.get("OUTPUT_PREFIX", "products/processed").rstrip("/")
    output_file   = config.get("OUTPUT_FILE_NAME", "catalog.csv")
    output_key    = f"{output_prefix}/{output_file}"

    raw      = fetch_products(api_url)
    df       = pd.json_normalize(raw)
    df_clean = transform_products(df)

    logger.info("Columns in final CSV: %s", list(df_clean.columns))
    logger.info("Rows: %d", len(df_clean))

    uri = save_to_s3(df_clean, output_bucket, output_key)
    logger.info("DONE -> %s", uri)


if __name__ == "__main__":
    main()