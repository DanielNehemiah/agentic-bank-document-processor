Create the architecture and code for an AI document processing system for a bank. 

The system should have the following:
1. Use a prompt to classify the document type.
2. Use a separate prompt for each type of document to extract the data and return a json
3. Store the response including prompt and document in a database
4. Have the option to access the extracted data over REST API and also to make requests
5. Have a UI to upload document and also view existing documents
6. Implement a simple access control as well
7. Support all types of documents that a bank would handle. Upto 20 types. 

Prefer the following: Python, FastAPI, Terraform, LangGraph, Pydantic validations, Agentic AI, Open source models, Open source datasets, Azure for cloud (don't use Azure specific options like document classifier in Azure, only use it for infra) 

Make the system very exhaustive. Make hardware choices, cloud deployment details, hosting infrastructure (terraform) for systems. Also think of things I may have missed (eg. Load balancing or network configuration).

Things I need:
- Architecture document in arc42 format
- Code for the entire system 
- Code and documentation about how to test the model on a small dataset to create a benchmark on performance 
- API docs
- Infrastructure scripts including documentation
- A simple modular and extensible prompt library
- A UI to add or modify prompts
