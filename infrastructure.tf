# Azure Infrastructure for BankDocAI
# Deploys: Resource Group, Postgres Flexible Server, Container Registry, Container App Environment

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "rg" {
  name     = "rg-bankdocai-prod-001"
  location = "East US"
}

# --- Networking ---
resource "azurerm_virtual_network" "vnet" {
  name                = "vnet-bankdocai"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  address_space       = ["10.0.0.0/16"]
}

resource "azurerm_subnet" "subnet_app" {
  name                 = "snet-app"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

# --- Database (PostgreSQL) ---
resource "azurerm_postgresql_flexible_server" "db" {
  name                   = "psql-bankdocai-prod"
  resource_group_name    = azurerm_resource_group.rg.name
  location               = azurerm_resource_group.rg.location
  version                = "13"
  administrator_login    = "bankadmin"
  administrator_password = "ChangeMe123!" # Use KeyVault in production
  storage_mb             = 32768
  sku_name               = "B_Standard_B1ms"
}

resource "azurerm_postgresql_flexible_server_database" "default" {
  name      = "bankdocai_db"
  server_id = azurerm_postgresql_flexible_server.db.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# --- Container Registry ---
resource "azurerm_container_registry" "acr" {
  name                = "acrbankdocai"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = true
}

# --- Container Apps Environment ---
resource "azurerm_container_app_environment" "env" {
  name                       = "cae-bankdocai"
  location                   = azurerm_resource_group.rg.location
  resource_group_name        = azurerm_resource_group.rg.name
  log_analytics_workspace_id = null # Set up Log Analytics in prod
}

# --- Container App (Backend) ---
resource "azurerm_container_app" "backend" {
  name                         = "ca-bankdocai-api"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = azurerm_resource_group.rg.name
  revision_mode                = "Single"

  template {
    container {
      name   = "bankdocai-api"
      image  = "acrbankdocai.azurecr.io/bankdocai:latest" # Needs image build
      cpu    = 0.5
      memory = "1.0Gi"
      
      env {
        name  = "DATABASE_URL"
        value = "postgresql://bankadmin:ChangeMe123!@${azurerm_postgresql_flexible_server.db.fqdn}:5432/bankdocai_db"
      }
      env {
        name  = "OPENAI_API_KEY"
        value = "placeholder-key"
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      percentage = 100
      latest_revision = true
    }
  }
}