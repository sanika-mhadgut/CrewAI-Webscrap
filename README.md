# CrewAI

## Local Host:

docker build -t crewai-streamlit-app .

docker run -p 8501:8501 \
  -e OPENAI_API_KEY="..." \
  -e LANGFUSE_PUBLIC_KEY="..." \
  -e LANGFUSE_SECRET_KEY="..." \
  -e LANGFUSE_HOST="https://us.cloud.langfuse.com" \
  crewai-streamlit-app


## Azure web app:

az login

az group create --name crewai-rg --location "Central US"

### Create ACR (use a unique name)
az acr create --resource-group crewai-rg --name crewaiacrsm --sku Basic

### Enable admin access (required for Azure Web App to pull images)
az acr update -n crewaiacrsm --admin-enabled true

cd Desktop/CrewAI-Webscrap

### Build for Azure’s platform and push directly to ACR
docker buildx build --platform linux/amd64 -t crewaiacrsm.azurecr.io/crewai-streamlit-app:v8 --push .

### Create an App Service Plan (Linux)
az appservice plan create \
  --name crewai-plan \
  --resource-group crewai-rg \
  --sku B1 \
  --is-linux

### Create the Web App with Docker container
az webapp create \
  --resource-group crewai-rg \
  --plan crewai-plan \
  --name crewai-streamlit-app \
  --deployment-container-image-name crewaiacrsm.azurecr.io/crewai-streamlit-app:v1

### Configure container registry credentials
az webapp config container set \
  --name crewai-streamlit-app \
  --resource-group crewai-rg \
  --container-image-name crewaiacrsm.azurecr.io/crewai-streamlit-app:v8 \
  --container-registry-url https://crewaiacrsm.azurecr.io \
  --container-registry-user $(az acr credential show -n crewaiacrsm --query "username" -o tsv) \
  --container-registry-password $(az acr credential show -n crewaiacrsm --query "passwords[0].value" -o tsv)

### Set environment variables (app secrets, keys, etc.)
az webapp config appsettings set \
  --resource-group crewai-rg \
  --name crewai-streamlit-app \
  --settings \
    OPENAI_API_KEY="your_openai_api_key" \
    LANGFUSE_PUBLIC_KEY="your_langfuse_public_key" \
    LANGFUSE_SECRET_KEY="your_langfuse_secret_key" \
    LANGFUSE_HOST="https://us.cloud.langfuse.com" \
    WEBSITES_PORT=80 \
    STREAMLIT_HEADLESS=true

az webapp restart --name crewai-streamlit-app --resource-group crewai-rg

az webapp log tail --name crewai-streamlit-app --resource-group crewai-rg

https://crewai-streamlit-app.azurewebsites.net

## Folder Structure

CrewAI-Webscrap/
├── Dockerfile
├── requirements.txt
└── app/
    ├── streamlit_app.py
    └── agent.py
