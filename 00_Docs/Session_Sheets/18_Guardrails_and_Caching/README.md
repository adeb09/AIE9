# Session #: [emoji] Session Name

🎯 **Goal**: Learn a few upgrades: for performance and security/trustworthiness.

📚 **Learning Outcomes**
- Understand guardrails, including the key categories of guardrails
- Understand the importance of semantic caching
- How to use Prompt and Embedding caches
- Learn how to use a skills.md file

🧰 **New Tools**
- [Guardrails](https://guardrailsai.com/)
- [CacheBackedEmbeddings](https://docs.langchain.com/oss/python/integrations/text_embedding#caching) 
- [Prompt Caching](https://docs.langchain.com/oss/python/langchain/models#prompt-caching)
## 📛 Required Tooling & Account Setup
In addition to the tools we've already learned, in this session you'll need:
    
1. 🔑 Get your Guardrails AI API key from [here](https://hub.guardrailsai.com/keys).
   
## 📜 Recommended Reading

- [The AI Guardrails Index](https://www.guardrailsai.com/blog/introducing-the-ai-guardrails-index) (Feb 2025)
- [Caching](https://planetscale.com/blog/caching) (July 2025)
- [Semantic Caching for Low-Cost LLM Serving](https://arxiv.org/html/2508.07675v1) (Aug 2025)
- [Guardrails](https://docs.langchain.com/oss/python/langchain/guardrails), by LangChain

# 🗺️ Overview

In Session 18, we’ll cover a few nice-to-haves for our production LLM applications that will become more and more important as we scale:  Skills.md, Caching, and Guardrails

As they say, cache is king. If we can save some cash with 🏦 caching, we should. We can leverage both prompt and embedding caches in general, depending on our use cases.

🛤️  Guardrails are the runtime checks that keep your AI application’s inputs and outputs on track—enforcing safety, policy, brand, correctness, and structure. This one‑pager clarifies definitions, shows where guards live in your stack, summarizes a minimally sufficient guard set for 2025, and gives a tiny flowchart to choose the right guard quickly.



