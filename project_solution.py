import pandas as pd
import numpy as np
import os
import re
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import OpenAIServerModel, ToolCallingAgent, tool

# Create an SQLite database
db_engine = create_engine("sqlite:///beavers_choice.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Beaver's Choice database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

# Load environment variables from a local .env file (UDACITY_OPENAI_API_KEY).
dotenv.load_dotenv()

# This project talks to an OpenAI-compatible proxy hosted by Udacity/Vocareum.
OPENAI_BASE_URL = "https://openai.vocareum.com/v1"
MODEL_ID = "gpt-4o-mini"

api_key = os.getenv("UDACITY_OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "UDACITY_OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
    )

# The shared LLM ("model") that powers every agent in the system.
model = OpenAIServerModel(
    model_id=MODEL_ID,
    api_base=OPENAI_BASE_URL,
    api_key=api_key,
)



"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

# Quick lookup maps built from the master catalog (paper_supplies).
CATALOG_PRICES = {item["item_name"]: item["unit_price"] for item in paper_supplies}
CATALOG_NAMES = list(CATALOG_PRICES.keys())


# The authoritative date for the request currently being processed. It is set from the
# request's "(Date of request: ...)" tag before the agents run, so the tools never have
# to trust a date argument the language model might hallucinate. When set, it overrides
# whatever as_of_date the model passes to a tool.
CURRENT_REQUEST_DATE: Union[str, None] = None


def _effective_date(as_of_date: str) -> str:
    """
    Return the date a tool should actually use. The request date captured in
    CURRENT_REQUEST_DATE is authoritative; the model-supplied as_of_date is only a
    fallback for when no request date has been set (e.g. ad-hoc tool calls).
    """
    return CURRENT_REQUEST_DATE or as_of_date


def resolve_item_name(item_name: str) -> Union[str, None]:
    """
    Map a free-text item description to an exact catalog item name.

    Transactions fail unless the exact catalog name is used, but customers (and the
    LLM) often paraphrase. This resolver tries, in order: exact match, case-insensitive
    match, then a simple substring overlap, returning the closest catalog name or None.

    Args:
        item_name (str): The raw item name or description to resolve.

    Returns:
        Union[str, None]: The matching exact catalog name, or None if no reasonable match.
    """
    if not item_name:
        return None

    query = item_name.strip().lower()

    # 1. Exact (case-insensitive) match against catalog names.
    for name in CATALOG_NAMES:
        if name.lower() == query:
            return name

    # 2. Substring match in either direction (e.g. "A4" -> "A4 paper").
    for name in CATALOG_NAMES:
        lowered = name.lower()
        if query in lowered or lowered in query:
            return name

    # 3. Token-overlap fallback: pick the catalog name sharing the most words.
    query_tokens = set(query.split())
    best_name, best_overlap = None, 0
    for name in CATALOG_NAMES:
        overlap = len(query_tokens & set(name.lower().split()))
        if overlap > best_overlap:
            best_name, best_overlap = name, overlap

    return best_name


# Tools for inventory agent

@tool
def check_inventory(item_name: str, as_of_date: str) -> str:
    """
    Check the current stock level for one catalog item as of a given date, including
    whether it has fallen to or below its minimum stock level (i.e. needs reordering).

    Args:
        item_name: The paper/product item to look up (free text is resolved to the catalog).
        as_of_date: The date (YYYY-MM-DD) to evaluate stock as of, inclusive.

    Returns:
        A human-readable summary with the resolved item name, current stock,
        minimum stock level, unit price, and a reorder recommendation.
    """
    as_of_date = _effective_date(as_of_date)
    resolved = resolve_item_name(item_name)
    if resolved is None:
        return f"Item '{item_name}' was not found in the catalog."

    stock_df = get_stock_level(resolved, as_of_date)
    current_stock = int(stock_df["current_stock"].iloc[0])

    # Look up the item's minimum stock level from the inventory reference table.
    inv = pd.read_sql(
        "SELECT min_stock_level FROM inventory WHERE item_name = :name",
        db_engine,
        params={"name": resolved},
    )
    min_level = int(inv["min_stock_level"].iloc[0]) if not inv.empty else 0
    unit_price = CATALOG_PRICES.get(resolved, 0.0)

    needs_reorder = current_stock <= min_level
    recommendation = (
        "AT OR BELOW minimum - reorder recommended."
        if needs_reorder
        else "Above minimum - no reorder needed."
    )

    return (
        f"Item: {resolved}\n"
        f"Current stock: {current_stock} units (as of {as_of_date})\n"
        f"Minimum stock level: {min_level} units\n"
        f"Unit price: ${unit_price:.2f}\n"
        f"Status: {recommendation}"
    )


@tool
def get_inventory_snapshot(as_of_date: str) -> str:
    """
    Get a snapshot of all items currently in stock (positive quantity) as of a date.

    Args:
        as_of_date: The date (YYYY-MM-DD) to evaluate stock as of, inclusive.

    Returns:
        A newline-separated list of every in-stock item and its quantity, or a
        message if no stock is available.
    """
    as_of_date = _effective_date(as_of_date)
    inventory = get_all_inventory(as_of_date)
    if not inventory:
        return f"No items are in stock as of {as_of_date}."

    lines = [f"- {name}: {stock} units" for name, stock in sorted(inventory.items())]
    return f"In-stock items as of {as_of_date}:\n" + "\n".join(lines)


@tool
def restock_item(item_name: str, quantity: int, as_of_date: str) -> str:
    """
    Place a supplier restock order for an item, recording it as a 'stock_orders'
    transaction and returning the estimated delivery date. Only restock when stock
    is low, and verify there is enough cash to cover the purchase cost.

    Args:
        item_name: The item to reorder (free text is resolved to the catalog).
        quantity: The number of units to order (must be positive).
        as_of_date: The order date (YYYY-MM-DD); delivery is estimated from this date.

    Returns:
        A confirmation with the resolved item, quantity, cost, and estimated delivery
        date, or an explanation if the order cannot be placed.
    """
    as_of_date = _effective_date(as_of_date)
    resolved = resolve_item_name(item_name)
    if resolved is None:
        return f"Item '{item_name}' was not found in the catalog; cannot restock."

    if quantity <= 0:
        return "Restock quantity must be a positive number of units."

    unit_price = CATALOG_PRICES.get(resolved, 0.0)
    cost = round(quantity * unit_price, 2)

    # Ensure the company can afford the restock before committing.
    cash = get_cash_balance(as_of_date)
    if cost > cash:
        return (
            f"Cannot restock {quantity} units of {resolved}: cost ${cost:.2f} "
            f"exceeds available cash ${cash:.2f}."
        )

    delivery_date = get_supplier_delivery_date(as_of_date, quantity)
    create_transaction(resolved, "stock_orders", quantity, cost, as_of_date)

    return (
        f"Restock order placed: {quantity} units of {resolved} for ${cost:.2f}.\n"
        f"Order date: {as_of_date}. Estimated delivery: {delivery_date}."
    )


# Tools for quoting agent

# Bulk discount tiers applied to a quote's pre-discount subtotal.
# Each tuple is (minimum subtotal in dollars, discount rate).
BULK_DISCOUNT_TIERS = [
    (1000.0, 0.10),  # 10% off orders over $1000
    (500.0, 0.05),   # 5% off orders over $500
]


def _bulk_discount_rate(subtotal: float) -> float:
    """Return the bulk discount rate that applies to a given pre-discount subtotal."""
    for threshold, rate in BULK_DISCOUNT_TIERS:
        if subtotal > threshold:
            return rate
    return 0.0


@tool
def get_quote_history(search_terms: List[str], limit: int = 3) -> str:
    """
    Retrieve similar historical quotes to inform pricing for a new request.

    Args:
        search_terms: Keywords describing the request (e.g. event type, item, job type).
        limit: Maximum number of past quotes to return (default 3).

    Returns:
        A readable summary of matching past quotes, including their total amount and
        explanation, or a message if no similar quotes are found.
    """
    history = search_quote_history(search_terms, limit=limit)
    if not history:
        return "No similar historical quotes were found."

    lines = []
    for i, quote in enumerate(history, start=1):
        lines.append(
            f"{i}. Total: ${quote.get('total_amount', 0):.2f} | "
            f"Job: {quote.get('job_type', 'n/a')} | "
            f"Event: {quote.get('event_type', 'n/a')} | "
            f"Size: {quote.get('order_size', 'n/a')}\n"
            f"   Explanation: {quote.get('quote_explanation', '').strip()}"
        )
    return "Similar historical quotes:\n" + "\n".join(lines)


@tool
def generate_quote(item_name: str, quantity: int, as_of_date: str) -> str:
    """
    Generate a price quote for a single item and quantity, applying catalog pricing
    and any qualifying bulk discount. This does NOT reserve stock or record a sale.

    Args:
        item_name: The item being quoted (free text is resolved to the catalog).
        quantity: The number of units requested (must be positive).
        as_of_date: The quote date (YYYY-MM-DD).

    Returns:
        A quote summary with unit price, subtotal, any bulk discount applied, and the
        final total, or an explanation if the item cannot be quoted.
    """
    as_of_date = _effective_date(as_of_date)
    resolved = resolve_item_name(item_name)
    if resolved is None:
        return f"Item '{item_name}' was not found in the catalog; cannot generate a quote."

    if quantity <= 0:
        return "Quote quantity must be a positive number of units."

    unit_price = CATALOG_PRICES.get(resolved, 0.0)
    subtotal = round(quantity * unit_price, 2)
    discount_rate = _bulk_discount_rate(subtotal)
    discount_amount = round(subtotal * discount_rate, 2)
    total = round(subtotal - discount_amount, 2)

    discount_line = (
        f"Bulk discount: {int(discount_rate * 100)}% (-${discount_amount:.2f})\n"
        if discount_rate > 0
        else "Bulk discount: none (order does not meet discount threshold)\n"
    )

    return (
        f"Quote for {quantity} units of {resolved} (as of {as_of_date}):\n"
        f"Unit price: ${unit_price:.2f}\n"
        f"Subtotal: ${subtotal:.2f}\n"
        f"{discount_line}"
        f"Total: ${total:.2f}"
    )


# Tools for ordering agent


@tool
def finalize_sale(item_name: str, quantity: int, as_of_date: str) -> str:
    """
    Finalize a customer order by recording it as a 'sales' transaction, but only if
    there is enough stock on hand. Applies the same catalog pricing and bulk discount
    used when quoting so the charged amount matches the quote. This permanently updates
    the database and reduces available stock.

    Args:
        item_name: The item being sold (free text is resolved to the catalog).
        quantity: The number of units the customer is ordering (must be positive).
        as_of_date: The order date (YYYY-MM-DD).

    Returns:
        A confirmation with the item, quantity, amount charged, and any discount applied,
        or an explanation if the order cannot be fulfilled (e.g. insufficient stock).
    """
    as_of_date = _effective_date(as_of_date)
    resolved = resolve_item_name(item_name)
    if resolved is None:
        return f"Item '{item_name}' was not found in the catalog; the order cannot be fulfilled."

    if quantity <= 0:
        return "Order quantity must be a positive number of units."

    # Confirm there is enough stock to fulfill the order before charging the customer.
    stock_df = get_stock_level(resolved, as_of_date)
    current_stock = int(stock_df["current_stock"].iloc[0])
    if quantity > current_stock:
        return (
            f"Unable to fulfill the order for {quantity} units of {resolved}: only "
            f"{current_stock} units are in stock as of {as_of_date}. Consider ordering "
            f"the available quantity or allowing time for a restock."
        )

    # Price the order with the same logic as the quote so the customer is charged consistently.
    unit_price = CATALOG_PRICES.get(resolved, 0.0)
    subtotal = round(quantity * unit_price, 2)
    discount_rate = _bulk_discount_rate(subtotal)
    discount_amount = round(subtotal * discount_rate, 2)
    total = round(subtotal - discount_amount, 2)

    create_transaction(resolved, "sales", quantity, total, as_of_date)

    discount_line = (
        f"Bulk discount applied: {int(discount_rate * 100)}% (-${discount_amount:.2f})\n"
        if discount_rate > 0
        else ""
    )

    return (
        f"Order confirmed: {quantity} units of {resolved} sold on {as_of_date}.\n"
        f"Subtotal: ${subtotal:.2f}\n"
        f"{discount_line}"
        f"Amount charged: ${total:.2f}"
    )


@tool
def check_delivery_date(quantity: int, as_of_date: str) -> str:
    """
    Estimate when a supplier restock order of a given size would be delivered, based on
    the order date. Useful for telling a customer when out-of-stock items could arrive.

    Args:
        quantity: The number of units that would be ordered from the supplier.
        as_of_date: The order date (YYYY-MM-DD) the estimate is based on.

    Returns:
        The estimated delivery date (YYYY-MM-DD) for an order of that size.
    """
    as_of_date = _effective_date(as_of_date)
    if quantity <= 0:
        return "Quantity must be a positive number of units to estimate delivery."

    delivery_date = get_supplier_delivery_date(as_of_date, quantity)
    return (
        f"An order of {quantity} units placed on {as_of_date} would be delivered "
        f"by {delivery_date}."
    )


@tool
def get_financial_report(as_of_date: str) -> str:
    """
    Produce a financial summary of the company as of a date, including cash balance,
    inventory value, total assets, and the top-selling products. Use this to confirm
    the company can support large restock orders or to review overall financial health.

    Args:
        as_of_date: The date (YYYY-MM-DD) to report as of, inclusive.

    Returns:
        A readable financial summary covering cash, inventory value, total assets, and
        the best-selling products to date.
    """
    as_of_date = _effective_date(as_of_date)
    report = generate_financial_report(as_of_date)

    top_products = report.get("top_selling_products", [])
    if top_products:
        top_lines = "\n".join(
            f"  - {p.get('item_name', 'n/a')}: ${p.get('total_revenue', 0):.2f} revenue"
            for p in top_products
        )
    else:
        top_lines = "  (no sales recorded yet)"

    return (
        f"Financial report as of {report.get('as_of_date', as_of_date)}:\n"
        f"Cash balance: ${report.get('cash_balance', 0):.2f}\n"
        f"Inventory value: ${report.get('inventory_value', 0):.2f}\n"
        f"Total assets: ${report.get('total_assets', 0):.2f}\n"
        f"Top-selling products:\n{top_lines}"
    )


# Set up your agents and create an orchestration agent that will manage them.

# Each customer request carries a "(Date of request: YYYY-MM-DD)" tag; every tool needs
# that date, so each agent is reminded to extract and reuse it for all tool calls.
_DATE_REMINDER = (
    "Every customer request includes its date as '(Date of request: YYYY-MM-DD)'. "
    "Extract that date and pass it as the as_of_date argument to every tool call so "
    "stock, pricing, and financials are evaluated for the correct day."
)

# --- Worker agent: inventory management ---
# Responsibility: report stock levels, flag items needing reorder, and place supplier
# restock orders. It owns all inventory reads/writes and nothing about pricing or sales.
inventory_agent = ToolCallingAgent(
    tools=[check_inventory, get_inventory_snapshot, restock_item],
    model=model,
    name="inventory_agent",
    description=(
        "Handles inventory questions: checks current stock for an item, lists everything "
        "in stock, decides whether an item needs reordering, and places supplier restock "
        "orders. Ask this agent to verify availability before quoting or selling, and to "
        "restock items that are low or out of stock."
    ),
    instructions=(
        "You are the inventory specialist for a paper-supply company. Use your tools to "
        "report exact stock levels, identify items at or below their minimum, and place "
        "restock orders only when stock is low and the company can afford them. "
        + _DATE_REMINDER
    ),
    max_steps=8,
    provide_run_summary=True,
)

# --- Worker agent: quoting ---
# Responsibility: turn an item + quantity into a price, applying catalog pricing and bulk
# discounts, and ground decisions in past quotes. It never reserves stock or records sales.
quoting_agent = ToolCallingAgent(
    tools=[get_quote_history, generate_quote],
    model=model,
    name="quoting_agent",
    description=(
        "Produces price quotes for an item and quantity, applying catalog pricing and "
        "bulk discounts, and can look up similar historical quotes to inform pricing. "
        "Ask this agent to price a request once availability is understood."
    ),
    instructions=(
        "You are the quoting specialist for a paper-supply company. For each requested "
        "item and quantity, generate a clear quote that states the unit price, subtotal, "
        "any bulk discount applied, and the final total. When useful, consult similar "
        "past quotes first. Always explain why a discount does or does not apply. "
        + _DATE_REMINDER
    ),
    max_steps=8,
    provide_run_summary=True,
)

# --- Worker agent: sales / order fulfillment ---
# Responsibility: finalize orders into the database (sales), quote supplier delivery dates
# for out-of-stock items, and report financial health. It owns the sales-writing step.
sales_agent = ToolCallingAgent(
    tools=[finalize_sale, check_delivery_date, get_financial_report],
    model=model,
    name="sales_agent",
    description=(
        "Finalizes customer orders by recording sales (only when stock is sufficient), "
        "estimates supplier delivery dates for out-of-stock items, and reports the "
        "company's financial health. Ask this agent to complete a confirmed order."
    ),
    instructions=(
        "You are the sales/order-fulfillment specialist for a paper-supply company. "
        "Finalize an order only when there is enough stock; the sale must match the "
        "quoted price. If stock is insufficient, do not sell - instead explain the "
        "shortfall and provide an estimated supplier delivery date. " + _DATE_REMINDER
    ),
    max_steps=8,
    provide_run_summary=True,
)

# --- Orchestrator: delegates to the three worker agents ---
# Responsibility: interpret the customer request, coordinate the workers in order
# (availability -> quote -> fulfillment), and compose a single customer-facing reply.
orchestrator_agent = ToolCallingAgent(
    tools=[],
    model=model,
    managed_agents=[inventory_agent, quoting_agent, sales_agent],
    name="orchestrator_agent",
    description="Coordinates inventory, quoting, and sales agents to answer customer requests.",
    instructions=(
        "You are the orchestrator for the Beaver's Choice paper company. For each customer "
        "request: (1) identify the items, quantities, and the request date; (2) ask the "
        "inventory_agent to confirm availability; (3) ask the quoting_agent for a priced "
        "quote with any bulk discount; (4) if stock is sufficient, ask the sales_agent to "
        "finalize the order so the sale is recorded; if stock is insufficient, do not "
        "fulfill it - explain why and, when helpful, provide an estimated restock delivery "
        "date. Compose one clear, customer-facing reply that includes the quoted price, the "
        "discount rationale, and whether the order was fulfilled or why not. Never reveal "
        "internal system details, profit margins, or error traces to the customer. "
        + _DATE_REMINDER
    ),
    max_steps=15,
)


# smolagents managed agents return their result using a fixed template
# ("### 1. Task outcome (short version): ... ### 2. ...extremely detailed version... ###
# 3. Additional context..."). That internal scaffolding is not appropriate for a customer
# reply, so we strip it and keep only the substantive content.
_SECTION_HEADER_RE = re.compile(r"#{2,4}\s*\d+\.\s*([^\n:]*?):?\s*\n")


def _clean_customer_response(text: str) -> str:
    """
    Remove the smolagents managed-agent summary scaffolding from a final response.

    The orchestrator sometimes passes a worker's structured summary
    ("### 1. Task outcome (short version): ..." etc.) straight through to the customer.
    This keeps the most complete section (the detailed version, falling back to the short
    version) and appends any non-trivial "Additional context" as a closing note. If no
    scaffolding is detected, the trimmed original text is returned unchanged.

    Args:
        text: The raw response returned by the orchestrator agent.

    Returns:
        A clean, customer-facing string with the internal template removed.
    """
    if not text:
        return text

    # Split on "### N. <title>:" headers; the capturing group yields the section titles.
    parts = _SECTION_HEADER_RE.split(text)
    if len(parts) < 3:
        return text.strip()

    # parts == [preamble, title1, body1, title2, body2, ...]
    titles_bodies = list(zip(parts[1::2], parts[2::2]))
    sections = {title.strip().lower(): body.strip() for title, body in titles_bodies}

    body = None
    for key, value in sections.items():
        if "detailed version" in key:
            body = value
            break
    if body is None:
        for key, value in sections.items():
            if "short version" in key or "task outcome" in key:
                body = value
                break
    if not body:
        return text.strip()

    for key, value in sections.items():
        if "additional context" in key:
            if value and value.lower().rstrip(".") not in {"none", "n/a", ""}:
                body = f"{body}\n\n{value}"
            break

    return body.strip()


def call_your_multi_agent_system(request: str) -> str:
    """
    Entry point for the multi-agent system: routes a single customer request through the
    orchestrator agent and returns its final customer-facing response as text.

    The request date is parsed from the '(Date of request: ...)' tag and stored in
    CURRENT_REQUEST_DATE so the tools always use the correct date, regardless of any date
    the language model might infer on its own. The date is also stated emphatically at the
    start of the task to keep the agents' wording consistent.

    Args:
        request: The customer request text, including its '(Date of request: ...)' tag.

    Returns:
        The orchestrator's final response as a string.
    """
    global CURRENT_REQUEST_DATE

    # Capture the authoritative request date from the tag so tools never depend on a date
    # the model might hallucinate.
    match = re.search(r"Date of request:\s*(\d{4}-\d{2}-\d{2})", request)
    CURRENT_REQUEST_DATE = match.group(1) if match else None

    if CURRENT_REQUEST_DATE:
        task = (
            f"TODAY'S DATE IS {CURRENT_REQUEST_DATE}. Use this exact date for every tool "
            f"call and in your reply; never use any other date or your own knowledge of "
            f"the current date, and only state dates that a tool returned to you.\n\n"
            f"{request}"
        )
    else:
        task = request

    try:
        result = orchestrator_agent.run(task)
        return _clean_customer_response(str(result))
    finally:
        CURRENT_REQUEST_DATE = None


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = call_your_multi_agent_system(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
