# Setting Up Your Open-Source Endpoint

> NOTE: If you do not wish to deploy a dedicated on-demand endpoint, you can use the serverless endpoint offered at:

- `accounts/fireworks/models/gpt-oss-20b`

## Your Generator

First, navigate to [fireworks.ai/models](https://fireworks.ai/models) and search for the model we'll be using today:

- `gpt-oss-20b`

### Option A: Serverless

You can use the serverless endpoint directly without any setup:

- Model identifier: `accounts/fireworks/models/gpt-oss-20b`

### Option B: On-Demand Deployment (Dedicated)

For a dedicated endpoint with guaranteed capacity, use `firectl` to deploy:

```bash
firectl create deployment gpt-oss-20b \
  --model accounts/fireworks/models/gpt-oss-20b \
  --min-replica 1 \
  --max-replica 1
```

> NOTE: Please ensure you configure auto-shutdown or remember to delete your deployment when finished to avoid unexpected charges.

### API Key

You'll need an API key from Fireworks. Get one at [fireworks.ai/api-keys](https://fireworks.ai/api-keys).

## Your Embeddings

Fireworks offers serverless endpoints for embedding models. We'll be using the Qwen3 Embedding 4B model today:

- `accounts/fireworks/models/qwen3-embedding-4b`


# Creating a Deployment in Fireworks AI (Web UI)

## 1. Open the Deployments Page

1. In the Fireworks AI console sidebar, click **Deployments**.
2. On the Deployments page, click the **Create Deployment** button.

This opens the deployment configuration screen where you can configure your model and compute settings.

---

# 2. Configure Basic Deployment Settings

Fill in the basic deployment details:

### Deployment Name
- Enter a unique **deployment name**.
- Example: `deployment-02`

### Base Model
- Select the **model you want to deploy** from the dropdown.
- Example shown: **MiniMax-M2.5**

### Region
- Choose the deployment **region**.
- Currently the interface shows **GLOBAL**.

---

# 3. Choose Performance Configuration

You can configure compute resources in **two ways**:

## Option A — Deployment Shapes (Recommended)

Choose a **preconfigured deployment shape**.

Example shown:

- **Minimal**
- GPU: **NVIDIA B200**
- Memory: **180GB**
- **4 GPUs**
- Precision: **FP8 MM**
- Estimated cost: **~$36/hr**

This option provides a quick setup with predefined performance settings.

---

## Option B — Manual Configuration

If you want more control, switch to **Manual Config**.

Available options include:

### Accelerator Type
- Example: **NVIDIA H100 80GB**

### Accelerator Count
- Number of GPUs to allocate
- Example: **Auto**

### Quantization Precision
- Example: **FP8**

### Optional settings
- Optimize for **Long Prompts**
- Enable **Multi-LoRA**

Manual configuration allows fine-tuning GPU resources and performance characteristics.

---

# 4. Configure Deployment Scaling

Under **Deployment Scaling**, configure autoscaling behavior:

- **Autoscaling:** Enabled
- **Replicas:** 0–1
- **Scale to zero:** after 60 minutes of inactivity

This helps control costs by automatically scaling resources up or down based on usage.

---

# 5. Deploy the Model

At the bottom of the page:

1. Review the **Projected Hourly Cost** (example: `$0 – $36`).
2. Click the **Deploy** button.

Fireworks AI will begin provisioning the infrastructure and deploying the model.

---

# 6. Monitor Deployment

Once deployed:

- The deployment will appear in the **Deployments dashboard**.
- You can monitor:
  - Status
  - Resource usage
  - Endpoint availability

After deployment completes, your model will be **available for inference via API**.

---

# Summary

1. Open **Deployments** → Click **Create Deployment**
2. Enter **Deployment Name**
3. Select **Base Model** and **Region**
4. Choose **Deployment Shape** or **Manual Configuration**
5. Configure **Autoscaling**
6. Click **Deploy**