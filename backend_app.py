import os
import json
import logging
import datetime
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import uuid

# FastAPI & Pydantic
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, create_model

# Database (SQLAlchemy + SQLite for demo, switch string for Postgres)
from sqlalchemy import create_engine, Column, String, Integer, JSON, DateTime, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- CONFIGURATION ---
DATABASE_URL = "sqlite:///./bankdocai.db" # Use postgresql://user:pass@host/db for prod
# Ideally, set OPENAI_API_KEY and OPENAI_BASE_URL env vars for your OS model provider
# e.g. export OPENAI_BASE_URL="https://api.groq.com/openai/v1" 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BankDocAI")

# --- DATABASE SETUP ---
Base = declarative_base()

class DocumentRecord(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String) # PENDING, PROCESSING, COMPLETED, FAILED
    doc_type = Column(String, nullable=True)
    extracted_data = Column(JSON, nullable=True)
    raw_text = Column(Text, nullable=True)
    used_prompt_id = Column(String, nullable=True)

class PromptConfig(Base):
    __tablename__ = "prompts"
    doc_type = Column(String, primary_key=True)
    system_prompt = Column(Text)
    extraction_schema = Column(JSON) # Store JSON schema representation
    is_active = Column(Boolean, default=True)

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    password_hash = Column(String) # In prod, verify hash
    role = Column(String) # admin, viewer

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# --- PYDANTIC MODELS FOR API ---
class Token(BaseModel):
    access_token: str
    token_type: str

class DocResponse(BaseModel):
    id: str
    filename: str
    status: str
    doc_type: Optional[str]
    extracted_data: Optional[Dict[str, Any]]
    upload_date: datetime.datetime

class PromptCreate(BaseModel):
    doc_type: str
    system_prompt: str
    extraction_schema: Dict[str, Any]

# --- LANGGRAPH AGENT WORKFLOW ---

class GraphState(BaseModel):
    """State for the document processing workflow."""
    doc_id: str
    raw_text: str
    doc_type: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Initialize LLM (Uses OpenAI format, works with Groq/Together/vLLM)
# Ensure API Key is set in environment
llm = ChatOpenAI(model="llama-3.1-70b-versatile", temperature=0) 

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 1. Classification Node
def classify_document(state: GraphState):
    logger.info(f"Classifying document {state.doc_id}")
    
    # Simple prompt for classification
    classification_prompt = """
    Analyze the following text from a document and classify it into one of these categories:
    [PASSPORT, PAYSTUB, BANK_STATEMENT, TAX_RETURN, INVOICE, DRIVER_LICENSE, UTILITY_BILL, UNKNOWN]
    
    Return ONLY the category name.
    
    Document Text snippet:
    {text}
    """
    try:
        # Taking first 2000 chars for classification is usually enough
        response = llm.invoke([HumanMessage(content=classification_prompt.format(text=state.raw_text[:2000]))])
        doc_type = response.content.strip().upper()
        # Clean up if model adds extra chars
        for valid in ["PASSPORT", "PAYSTUB", "BANK_STATEMENT", "TAX_RETURN", "INVOICE"]:
            if valid in doc_type:
                doc_type = valid
                break
        
        return {"doc_type": doc_type}
    except Exception as e:
        return {"error": str(e)}

# 2. Extraction Node
def extract_data(state: GraphState):
    logger.info(f"Extracting data for {state.doc_type}")
    
    if not state.doc_type or state.doc_type == "UNKNOWN":
        return {"extracted_data": {"summary": "Could not classify document type"}}

    # Fetch prompt from DB
    db = SessionLocal()
    prompt_cfg = db.query(PromptConfig).filter(PromptConfig.doc_type == state.doc_type).first()
    db.close()

    system_instruction = "Extract data in JSON format."
    if prompt_cfg:
        system_instruction = prompt_cfg.system_prompt
    
    extraction_prompt = f"""
    {system_instruction}
    
    Return ONLY valid JSON.
    
    Document Text:
    {state.raw_text}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        # Basic cleanup for JSON parsing
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return {"extracted_data": data}
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {"error": f"Extraction failed: {str(e)}", "extracted_data": {}}

# Define the Graph
workflow = StateGraph(GraphState)
workflow.add_node("classify", classify_document)
workflow.add_node("extract", extract_data)

workflow.set_entry_point("classify")
workflow.add_edge("classify", "extract")
workflow.add_edge("extract", END)

app_graph = workflow.compile()

# --- BACKGROUND WORKER ---

def process_document_task(doc_id: str, text_content: str):
    db = SessionLocal()
    doc = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not doc:
        db.close()
        return

    doc.status = "PROCESSING"
    doc.raw_text = text_content
    db.commit()

    try:
        # Run LangGraph
        initial_state = GraphState(doc_id=doc_id, raw_text=text_content)
        result = app_graph.invoke(initial_state)
        
        doc.doc_type = result.get("doc_type")
        doc.extracted_data = result.get("extracted_data")
        doc.status = "COMPLETED"
    except Exception as e:
        logger.error(f"Error processing {doc_id}: {e}")
        doc.status = "FAILED"
        doc.extracted_data = {"error": str(e)}
    
    db.commit()
    db.close()

# --- API ROUTES ---

app = FastAPI(title="BankDocAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Dummy auth for demo
    if form_data.username == "admin" and form_data.password == "admin":
        return {"access_token": "dummy_jwt_token", "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Incorrect username or password")

@app.post("/upload", response_model=DocResponse)
async def upload_document(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    # In a real app, upload to Azure Blob Storage here and get URL/Text via OCR
    # For this demo, we assume the file is a text file or we do simple dummy OCR
    content = await file.read()
    try:
        text_content = content.decode("utf-8")
    except:
        text_content = "Binary file content placeholder. Real implementation needs OCR (Tesseract/Azure AI)."

    new_doc = DocumentRecord(filename=file.filename, status="PENDING")
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    background_tasks.add_task(process_document_task, new_doc.id, text_content)

    return DocResponse(
        id=new_doc.id, 
        filename=new_doc.filename, 
        status=new_doc.status, 
        upload_date=new_doc.upload_date,
        doc_type=None,
        extracted_data=None
    )

@app.get("/documents", response_model=List[DocResponse])
def get_documents(db: Session = Depends(get_db)):
    docs = db.query(DocumentRecord).order_by(DocumentRecord.upload_date.desc()).all()
    return [
        DocResponse(
            id=d.id, filename=d.filename, status=d.status, 
            doc_type=d.doc_type, extracted_data=d.extracted_data, 
            upload_date=d.upload_date
        ) 
        for d in docs
    ]

@app.get("/documents/{doc_id}", response_model=DocResponse)
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocResponse(
        id=doc.id, filename=doc.filename, status=doc.status, 
        doc_type=doc.doc_type, extracted_data=doc.extracted_data, 
        upload_date=doc.upload_date
    )

@app.post("/prompts")
def create_or_update_prompt(prompt: PromptCreate, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    existing = db.query(PromptConfig).filter(PromptConfig.doc_type == prompt.doc_type).first()
    if existing:
        existing.system_prompt = prompt.system_prompt
        existing.extraction_schema = prompt.extraction_schema
    else:
        new_prompt = PromptConfig(**prompt.dict())
        db.add(new_prompt)
    db.commit()
    return {"status": "success"}

# --- SEED DATA ---
@app.on_event("startup")
def seed_data():
    db = SessionLocal()
    # Seed default prompt for Passport
    if not db.query(PromptConfig).filter(PromptConfig.doc_type == "PASSPORT").first():
        db.add(PromptConfig(
            doc_type="PASSPORT",
            system_prompt="Extract the following fields from the passport: full_name, passport_number, nationality, date_of_birth, expiration_date. Date format: YYYY-MM-DD.",
            extraction_schema={"type": "object", "properties": {"passport_number": {"type": "string"}}}
        ))
    db.commit()
    db.close()

