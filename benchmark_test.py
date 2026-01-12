import asyncio
import json
import time
from typing import Dict, List
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score

# Mocking the client for standalone execution
# In real usage, import `classify_document` and `extract_data` from backend_app
# or call the API endpoint running on localhost.

@dataclass
class TestSample:
    text_content: str
    true_type: str
    true_data: Dict

# --- MOCK DATASET ---
dataset = [
    TestSample(
        text_content="PASSPORT United States of America. Surname: DOE. Given Names: JOHN. Passport No: 123456789. DOB: 12 Aug 1980.",
        true_type="PASSPORT",
        true_data={"passport_number": "123456789", "surname": "DOE"}
    ),
    TestSample(
        text_content="INVOICE #INV-2023-001. Vendor: ACME Corp. Date: 2023-01-01. Total: $500.00. Bill To: BankDocAI.",
        true_type="INVOICE",
        true_data={"invoice_number": "INV-2023-001", "total_amount": 500.00}
    ),
    TestSample(
        text_content="Driver License State of California. Lic# D1234567. Exp: 01/01/2025. Class C.",
        true_type="DRIVER_LICENSE",
        true_data={"license_number": "D1234567"}
    )
]

# --- SIMULATED PREDICTOR ---
# Replace this with actual calls to your LangGraph or API
async def predict(text: str):
    # Simulating latency
    await asyncio.sleep(0.1)
    
    # Simple heuristics to simulate LLM for this script
    if "PASSPORT" in text.upper():
        return "PASSPORT", {"passport_number": "123456789", "surname": "DOE"}
    elif "INVOICE" in text.upper():
        return "INVOICE", {"invoice_number": "INV-2023-001", "total_amount": 500.00}
    elif "DRIVER LICENSE" in text.upper():
        return "DRIVER_LICENSE", {"license_number": "D1234567"}
    return "UNKNOWN", {}

# --- BENCHMARK ENGINE ---

async def run_benchmark():
    print("Starting Benchmark on Small Dataset (N=3)...")
    print("-" * 50)
    
    y_true = []
    y_pred = []
    
    start_time = time.time()
    
    for sample in dataset:
        pred_type, pred_data = await predict(sample.text_content)
        
        y_true.append(sample.true_type)
        y_pred.append(pred_type)
        
        print(f"Sample: {sample.true_type} | Predicted: {pred_type}")
        
        # In deep testing, we would also compare JSON keys/values similarity
        # e.g. using Jaccard similarity or strict equality on specific fields
    
    total_time = time.time() - start_time
    
    print("-" * 50)
    print("Results:")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Avg Latency: {total_time/len(dataset):.4f}s")
    
    # Calculate Metrics
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())