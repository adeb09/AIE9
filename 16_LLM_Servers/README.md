<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 16: LLM Servers</h1>

| 📰 Session Sheet                                  | ⏺️ Recording                           | 🖼️ Slides                                   | 👨‍💻 Repo       | 📝 Homework                                              | 📁 Feedback                        |
| ------------------------------------------------- | -------------------------------------- | ------------------------------------------- | ------------- | -------------------------------------------------------- | ---------------------------------- |
| [LLM Servers](../00_Docs/Session_Sheets/16_LLM_Servers) |[Recording!](https://us02web.zoom.us/rec/share/HDunij9p7eCXeP_OgsRDRjTdWUqiEhDBGWrFJEn1bwWR1wz1jKX6EHXSOM45d0sC.rHiyo_znZ-R8Jh6S) <br> passcode: `D80X^YjL`| [Session 16 Slides](https://www.canva.com/design/DAG-EBu7B5A/POcowC5rDLENSPcSVpbf8g/edit?utm_content=DAG-EBu7B5A&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 16 Assignment: LLM Servers](https://forms.gle/Riqvwf6KrZcCRKV86) <br><br> [Demo Day Submission (3/12)](https://forms.gle/7xyuBUn69GX4v6K98)  | [Feedback 3/5](https://forms.gle/W28QFWJXpSS4ZAR6A) |

**⚠️!!! PLEASE BE SURE TO SHUTDOWN YOUR DEDICATED ENDPOINT ON FIREWORKS AI WHEN YOU'RE FINISHED YOUR ASSIGNMENT !!!⚠️**

# Build 🏗️

In today's assignment, we'll be creating Fireworks AI endpoints, and then building a RAG application.

- 🤝 Breakout Room #1
  - Set-up Open Source Endpoint (Instructions [here](./ENDPOINT_SETUP.md)) ((This process may take 15-20min.))
  - Test Endpoint and Embeddings with the `endpoint_slammer.ipynb` notebook.

- 🤝 Breakout Room #2
  - Use the Open Source Endpoints to build a RAG LangGraph application

# Ship 🚢

The completed notebook and your RAG app/notebook!

### Deliverables

- A short Loom of either:
  - the notebook and the RAG application you built for the Main Homework Assignment; or
  - the notebook you created for the Advanced Build

# Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped a RAG application powered by open-source endpoints! 🎉🤖

🔍 Three Key Takeaways:
1️⃣
2️⃣
3️⃣

Let's continue pushing the boundaries of what's possible in the world of AI and question-answering. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#LangChain #QuestionAnswering #RetrievalAugmented #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```

# Submitting You Homework [OPTIONAL]

## Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:

1. Follow the instructions in `ENDPOINT_SETUP.md`
2. Replace both `model` values in `endpoint_slammer.ipynb` with the `gpt-oss` endpoint you created in Step 1
3. Run the code cells in `endpoint_slammer.ipynb`
4. Respond to the questions in the section below
5. Build a sample RAG
6. Record a Loom video reviewing what you have learned from this session

**⚠️!!! PLEASE BE SURE TO SHUTDOWN YOUR DEDICATED ENDPOINT ON FIREWORKS AI WHEN YOU HAVE FINISHED YOUR ASSIGNMENT !!!⚠️**

## Questions

### ❓ Question #1:

What is the difference between serverless and dedicated endpoints?

#### ✅ Answer:

- serverless endpoints are usually shared infrastructure that has a different payment model (usually based on the amount of tokens used instead of time that the endpoint is up)
  - usually serverless endpoints may have more limits on concurrency/and usage limits compared to dedicated endpoints
- dedicated endpoints are usually infrastructure that is just hosting your specific endpoint and its completely dedicated to your workload
  - the pricing model for a dedicated endpoint will usually be based on how long the endpoint is up and running
  - usually you can pick the specific hardware you want

### ❓ Question #2:

Why is it important to consider token throughput and latency when choosing an LLM for user-facing applications?

#### ✅ Answer:

- **Latency** directly affects the user experience of your app
  - if a model takes several seconds to return the first token, users perceive the application as slow or unresponsive, leading to frustration and drop-off
  - for real-time or conversational interfaces, low time-to-first-token (TTFT) is a critical metric
- **Token throughput** (tokens per second) determines how quickly a complete response is delivered
  - a slow generation rate means users wait a long time for longer responses
- Together, these metrics set a practical ceiling on how many concurrent users you can serve
- **Low throughput** means each request ties up infrastructure longer, reducing the number of simultaneous users you can handle without degrading quality.
- Cost is also tied to throughput — a higher-throughput model can serve more requests per unit time on the same hardware, which directly lowers cost per query at scale.
- you need to balance quality with responsiveness: a highly capable model that is too slow or has low throughput may actually deliver a worse product experience than a slightly less capable but faster model.

## Activity 1: RAGAS Evaluation with Cost Analysis

Use RAGAS to evaluate your open-source Fireworks AI powered RAG app against an OpenAI `gpt-4.1-mini` powered equivalent. Compare retrieval quality, answer faithfulness, and end-to-end accuracy across both providers.

Additionally, instrument both pipelines with **LangSmith** to capture token usage and cost per query. Use LangSmith's tracing and cost dashboards to compare the total cost of running each provider at scale. Include your evaluation results, cost breakdown, and analysis in your Loom video.

## Advanced Activity: Local Models

Swap out the Fireworks AI endpoints for **locally-running open-source models** using [Ollama](https://ollama.com/) or another local inference server of your choice. Run both your embedding model and your chat model locally, and rebuild the RAG pipeline on top of them.

- Compare quality and latency between the local setup and your Fireworks AI hosted endpoint.
- Reflect: what are the trade-offs of local models vs. managed endpoints in a production setting?

Include your findings and a demo in your Loom video.
