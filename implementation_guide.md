Implementation Guide
​Backend:
​Install dependencies: pip install fastapi uvicorn sqlalchemy pydantic langchain-openai langgraph
​Run server: uvicorn backend_app:app --reload
​The app uses sqlite by default. For production, update DATABASE_URL to point to the Azure Postgres instance created by Terraform.
​Frontend:
​Open dashboard.html in any browser. It connects to localhost:8000 by default.
​Login with admin / admin.
​Upload simulated text files (since OCR is mocked in backend_app.py, uploading a .txt file with content like "PASSPORT DETAILS..." will trigger the extraction logic).
​Infrastructure:
​Install Terraform and Azure CLI.
​Run az login.
​Run terraform init and terraform apply.
​Testing:
​Run python benchmark_test.py to see how the logic performs against a test set. This file mimics the API logic for rapid local testing.