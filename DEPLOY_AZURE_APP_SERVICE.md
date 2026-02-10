## Deploy to Azure App Service (container) with Azure Files mount

This app exposes:
- `POST /classify/rag`
- `POST /classify/agentic`

### 1) Build and push container
Use ACR or Docker Hub. Example with ACR:

```bash
az acr create -g <rg> -n <acrName> --sku Basic
az acr login -n <acrName>
docker build -t <acrName>.azurecr.io/hs-classifier:latest .
docker push <acrName>.azurecr.io/hs-classifier:latest
```

### 2) Create Azure Files and upload Chroma DB
You need the Chroma persistent directory (the folder containing `chroma.sqlite3` and collection files).

```bash
az storage account create -g <rg> -n <storageName> -l <region> --sku Standard_LRS
az storage share create --account-name <storageName> --name <shareName>
```

Upload the local Chroma directory to the share (use Azure Storage Explorer or `az storage file upload-batch`):

```bash
az storage file upload-batch --account-name <storageName> --destination <shareName> --source ./vector_db/chroma_hs_codes
```

### 3) Create App Service (Linux, container)

```bash
az appservice plan create -g <rg> -n <planName> --is-linux --sku B1
az webapp create -g <rg> -n <appName> --plan <planName> \
  --deployment-container-image-name <acrName>.azurecr.io/hs-classifier:latest
```

If using ACR, grant pull access:
```bash
az webapp config container set -g <rg> -n <appName> \
  --docker-custom-image-name <acrName>.azurecr.io/hs-classifier:latest \
  --docker-registry-server-url https://<acrName>.azurecr.io
```

### 4) Mount Azure Files into the container
Mount the share at `/mnt/hs-chroma`:

```bash
az webapp config storage-account add -g <rg> -n <appName> \
  --custom-id hs-chroma \
  --storage-type AzureFiles \
  --share-name <shareName> \
  --account-name <storageName> \
  --mount-path /mnt/hs-chroma \
  --access-key <storageKey>
```

### 5) App settings (required)
Set these in App Service -> Configuration -> Application settings:

- `OPENAI_API_KEY` = your OpenAI key
- `CHROMA_PATH` = `/mnt/hs-chroma`
- `WEBSITES_PORT` = `8000`

### 5.1) Startup command (if not using a custom container)
If you deploy as a Python app (not a container), set the App Service **Startup Command** so Oryx doesn’t fall back to the default site:

```
gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 main:app
```

### 6) Test
```bash
curl -X POST https://<appName>.azurewebsites.net/classify/agentic \
  -H "Content-Type: application/json" \
  -d "{\"description\":\"cotton t-shirt\",\"k\":20}"
```

### Local container test (optional)
```bash
docker build -t hs-classifier:local .
docker run -p 8000:8000 -e OPENAI_API_KEY=... -e CHROMA_PATH=/mnt/hs-chroma \
  -v "$(pwd)/vector_db/chroma_hs_codes:/mnt/hs-chroma" hs-classifier:local
```
