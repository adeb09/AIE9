# Session 16: 🔓 LLM Servers

🎯 Learn to deploy remotely-hosted open LLMs and embedding models to use in your agent applications.

📚 **Learning Outcomes**

- Understand how to deploy open-source LLMs and embeddings to scalable, remote, production-ready endpoints
- Learn how to use leaderboards to pick the best OSS models
- Learn AI Makerspace's picks for the best OSS models today

🧰 **New Tools**

LLM: [gpt-oss-20b](https://openai.com/index/introducing-gpt-oss/)
Embeddings: [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
LLM Serving & Inference: [Fireworks.ai](https://fireworks.ai/)

## 📛 Required Tooling & Account Setup

In addition to the tools we've already learned, in this session you'll need:

1. A [Fireworks.ai](https://fireworks.ai/) account for managed LLM and embedding endpoints

## 📜 Recommended Reading

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Open LLM Leaderboard Blog](https://huggingface.co/spaces/open-llm-leaderboard/blog)
- [Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/)
- [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://qwen.ai/blog?id=qwen3-embedding)
---

# 🦙 Open-Source Endpoints for Production

Open-source large language models (LLMs) have gained traction with their increasing performance across many dimensions, and when choosing the right one for your job, there are a few dimensions that you'll want to consider.

TL;DR there are typically obvious choices for you to start with when building, shipping, and sharing production LLM application prototypes like the ones in this class. We'll cover them during this session.

## Licensing

Of course, it's important to understand exactly what `open source` means for your use case, and where the "openness" starts to get restricted. In other words, "open-source" is used loosely in the AI Engineering world. Some companies and model families that have historically shown leadership (e.g., **Llama 3.1**) are open-weight under a [community license with use-case restrictions](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE). Other models from leading model providers (e.g., Mistral's [Mixtral 8×22B](https://mistral.ai/news/mixtral-8x22b), DeepSeek's [V3](https://api-docs.deepseek.com/news/news250325) and R1) are more permissive OSI-style licenses (Apache-2.0/MIT).

## Inference Servers

Unlike serving up an entire application, as with LangGraph Platform and LangGraph Server, the goal of hosting LLM chat and embedding endpoints is just to do inference. Thus, we are setting up inference servers when we set up these endpoints.

Leading OSS inference servers include [vLLM](https://github.com/vllm-project/vllm) (e.g., Virtual LLM), [Hugging Face Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/en/index) (the inference engine that underlies HF Inference Endpoints), [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [SGLang](https://docs.sglang.ai/), [ollama](https://ollama.com/), and more.

Importantly, while these OSS inference servers are highly configurable and very useful and performant off the shelf, it is nice to abstract away much of the detail that we don't need when building, shipping, and sharing production LLM applications.

## Fully Managed Endpoints

All we want is an [OpenAI-style API](https://platform.openai.com/docs/api-reference/introduction) to provide requests in the form of `system`, `user`, and `assistant` instructions, and to return responses from. Perhaps we'll be interested in choosing the hardware that we run on, and in some cases maybe we want to detail the scaling strategy that we plan to use for usage spikes (beyond simple autoscaling).

To accomplish this, managed services that sit atop inference servers like [Fireworks.ai](https://fireworks.ai/), [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated), [NVIDIA NIM](https://developer.nvidia.com/nim), and others are often used. These make many of the decisions for you when you're trying to get up and running quickly.

With the rise of accessible tools like [Fireworks.ai](https://fireworks.ai/), enabling scalable deployments without the need for complex cloud infrastructures has never been easier. By integrating endpoint scaling and GPU management, it allows developers to focus on building without worrying about backend complexities. This edge makes tools like these strong contenders for startups or small teams, providing an alternative to managing custom infrastructure on AWS, GCP, or Azure.

**Key Considerations**:

- 🚀 **Ease of setup** for one-person or small teams.
- ⚙️ **Managed GPU scaling** and cost-efficiency.
- 📈 **Automatic scaling features** allow you to handle high traffic spikes without additional manual setup.
- 🦙 **Using open-source models** like Llama 3, DeepSeek, or Qwen, developers can launch applications faster while still maintaining flexibility.

---

# 🏅 Choosing Open-Source Models

There are many considerations when it comes to choosing open-source models, including the country it was made in! This has been particularly important recently, and American-made 🇺🇸 models like [Arcee Trinity](https://www.arcee.ai/) and [gpt-oss](https://openai.com/index/introducing-gpt-oss/) have started to tout this as a real feature.

As the age of open-source Llama models that have led us forward to now [appears to be coming to a close](https://techcrunch.com/2025/07/30/zuckerberg-says-meta-likely-wont-open-source-all-of-its-superintelligence-ai-models/), we will opt to check out [gpt-oss-20b](https://openai.com/index/introducing-gpt-oss/) on Fireworks.ai for this session! It's also worth noting that during last week's mastermind session, we covered the new [Gemma 3 270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/), small enough to run directly within a web browser!

In general, selecting open-source models is a procedure that has historically started with LLM leaderboards on Hugging Face. As you'll notice, this was recently archived:

[Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/)

**Reading this blog is the best way to get up to speed as of June 2024 on the SOTA**.

[Open LLM Leaderboard Blog](https://huggingface.co/spaces/open-llm-leaderboard/blog)

> *Depending on your practical use case, you should focus on various aspects of the leaderboard. The overall ranking will tell you which model is better on average, but you might be more interested in specific capabilities.*

You can select the right chat model for your application if:

1. You understand the general availability and landscape of open-source models that are available today
2. You have an understanding of the kind of performance, latency, and throughput that you need (e.g., you have observed a baseline during prototyping), and
3. You understand how much horsepower you need for each part of your stack (e.g., how much value does improving the embedding or chat model beyond where you are actually provide to the end user)

These are, of course, difficult things to get a handle on. But fortunately, there's typically an easier way, and we'll provide that for you. In short, **we'll give you some default options** for 2026.

In lieu of that, we can note that the best chat model is in general more difficult than the best embedding model. Why? Because embedding models are not aimed at "AGI" or "ASI." Rather, they have specific uses. Thus, the MTEB leaderboard is still quite useful.

[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

## Embedding Models

> No particular embedding model dominates across all tasks. ~ [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)

How to choose the best model? Consider the following requirements:

- No. **parameters**
- **Memory** usage
- **Metrics** and **benchmark** **datasets** used to determine position on **leaderboard**
- (Custom) Domain?

Answer: Use the **smallest** one that **gets the job done**!

- Choose < 500M
- Check performance
- If not good enough, think through your problem again

Alternatively, just pick the *biggest, baddest model on the leaderboard*.

---

# Conclusions

In general, selecting the right open-source models involves understanding the trade-offs between size, task-specific performance, and deployment complexity.

For both LLMs and embedding models, these best-practice choices are shifting and changing, *but not all that much*. It's the goal of this session to draw a line in the sand that you can edit in the future as you stay out on [The LLM Edge](https://newsletter.aimakerspace.io/).
