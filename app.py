from fastapi import FastAPI

app = FastAPI()


# some fake data to work with (like a mini database)
transactions = [
    {"id": 1, "amount": 50.00, "merchant": "Amazon", "status": "approved"},
    {"id": 2, "amount": 3200.00, "merchant": "Unknown Shop", "status": "flagged"},
    {"id": 3, "amount": 15.99, "merchant": "Netflix", "status": "approved"},
]


@app.get("/")
def home():
    return {"message": "Welcome to the Fraud Detection API!"}


@app.get("/transactions")
def get_transactions():
    return transactions


@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: int):
    for txn in transactions:
        if txn["id"] == transaction_id:
            return txn
    return {"error": "Transaction not found"}


@app.post("/transactions")
def add_transaction(amount: float, merchant: str):
    new_id = len(transactions) + 1
    new_txn = {
        "id": new_id,
        "amount": amount,
        "merchant": merchant,
        "status": "approved" if amount < 1000 else "flagged",
    }
    transactions.append(new_txn)
    return new_txn


@app.delete("/transactions/{transaction_id}")
def delete_transaction(transaction_id: int):
    for i, txn in enumerate(transactions):
        if txn["id"] == transaction_id:
            deleted = transactions.pop(i)
            return {"message": "Deleted", "transaction": deleted}
    return {"error": "Transaction not found"}
