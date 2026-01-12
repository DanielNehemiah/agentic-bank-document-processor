# BankDocAI - Architecture Document (arc42)
1. Introduction and Goals
BankDocAI is an intelligent document processing system designed for banking institutions. It automates the ingestion, classification, and data extraction of financial documents (e.g., Passports, Paystubs, Tax Returns) using Generative AI agents.
1.1 Requirements
 * Multi-step Processing: Classify -> Route -> Extract -> Validate.
 * Extensibility: Support 20+ document types with customizable prompts.
 * Interface: REST API & Admin UI.
 * Infrastructure: Cloud-native (Azure), scalable, and secure.
1.2 Quality Goals
 * Accuracy: High precision in data extraction (measured via benchmarks).
 * Scalability: Ability to handle concurrent uploads during peak banking hours.
 * Security: Role-Based Access Control (RBAC) and data encryption.
2. Architecture Constraints
 * Cloud Provider: Microsoft Azure (Infrastructure only).
 * AI Models: Open-source models (e.g., Llama 3, Mixtral) hosted via standard APIs (e.g., Groq, vLLM, or Azure Open Model endpoints).
 * Stack: Python, FastAPI, Terraform, React.
3. System Scope and Context
3.1 Business Context
Users (Bank Operations): Upload documents via UI or API.
External Systems: Core Banking System (requests data via API).
BankDocAI: Processes files and returns structured JSON.
3.2 Technical Context
 * Input: PDFs, Images (JPEG/PNG).
 * Output: JSON structured data.
 * Communication: REST (HTTPS).
4. Solution Strategy
 * Agentic Workflow: Uses LangGraph to model the document processing pipeline as a state machine.
   * Node 1: Classifier - Determines if the doc is a Passport, Invoice, etc.
   * Node 2: Router - Selects the specific schema and prompt.
   * Node 3: Extractor - LLM extraction.
   * Node 4: Validator - Pydantic validation.
 * Infrastructure: Azure Container Apps for serverless scaling of the API. Azure PostgreSQL for persistent storage.
5. Building Block View
5.1 Level 1: Whitebox
 * API Gateway (Azure Load Balancer): Entry point.
 * Application Server (FastAPI):
   * Auth Module: JWT handling.
   * DocProcessor: LangGraph workflow.
   * PromptManager: Loads prompts from DB/Config.
 * Database (PostgreSQL): Stores document metadata, extracted JSON, and prompts.
 * Object Storage (Azure Blob): Stores raw document files.
6. Runtime View (Processing Flow)
 * User uploads file.pdf to /upload.
 * API saves file to Blob Storage and creates a DB record (Status: PENDING).
 * LangGraph Agent picks up the task:
   * Step 1: Sends first page text to Classifier LLM. Result: "US_PASSPORT".
   * Step 2: Fetches "US_PASSPORT" prompt and Pydantic schema.
   * Step 3: Sends full text + Prompt to Extractor LLM.
   * Step 4: Validates JSON output.
 * API updates DB record with JSON result (Status: COMPLETED).
7. Deployment View (Azure)
 * Region: East US (example).
 * VNet: 10.0.0.0/16.
   * Subnet App: 10.0.1.0/24 (Container Apps).
   * Subnet DB: 10.0.2.0/24 (PostgreSQL Private Endpoint).
 * Compute: Azure Container Apps (Autoscaling: 1-10 replicas based on HTTP traffic).
 * Storage: Azure Blob Storage (Hot tier).
8. Cross-cutting Concepts
 * Observability: OpenTelemetry instrumentation sending traces to Azure Monitor.
 * Security: Secrets stored in Azure Key Vault.
