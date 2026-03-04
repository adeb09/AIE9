

## # Session 14: MCP Connectors


| 📰 Session Sheet                                             | ⏺️ Recording                           | 🖼️ Slides                                  | 👨‍💻 Repo    | 📝 Homework                                      | 📁 Feedback                                          |
| ------------------------------------------------------------ | -------------------------------------- | ------------------------------------------- | ------------- | ------------------------------------------------ | ---------------------------------------------------- |
| [MCP Connectors](../00_Docs/Session_Sheets/14_MCP_Connectors) |[Recording!](https://us02web.zoom.us/rec/share/s9Cu44bGFiDaGZImyXfBgu7QM0VEFJaCuUF73p0p5878HlkYeHuZX6PTOCtoBAmF.nc6-njf7sNkdQOkR) <br> passcode: `2W$Rrs5=`| [Session 14 Slides](https://www.canva.com/design/DAG-EM1Hsh8/Z3bEjI4cgFHEpIIJrHg3fg/edit?utm_content=DAG-EM1Hsh8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 14 Assignment: MCP Connectors](https://forms.gle/dGecHpHLkF9Ctkz57) | [Feedback 2/26](https://forms.gle/zsm3T7gvKJewCJRK7) |



### Prerequisites

Before starting, ensure you have the following:

- A **GitHub account** (free tier is fine) with a **Personal Access Token** (fine-grained, `repo` scope)
- An **X (Twitter) Developer account** with a Bearer Token (Free tier provides basic access)
- **Python 3.11+** installed
- Basic familiarity with LangGraph (Sessions 4-6) and Python

Create a `.env` file in this directory with your API keys:

1. Run `uv sync` to install dependencies.

# Build 🏗️

Run the notebook and complete the following:

- 🤝 Breakout Room Part #1: Setup, Tools & Agent Construction
  - Task 1: Dependencies & Environment
  - Task 2: X API as LangChain Tools (`@tool` decorator)
  - Task 3: Connect to GitHub MCP Server & Load Tools (`langchain-mcp-adapters`)
  - Task 4: Build the LangGraph Agent with Memory (`StateGraph` + `MemorySaver`)
  - Task 5: Test the Agent — Search & Summarize X Posts
  - **Activity #1: Extend the Agent with a Custom X API Tool**
- 🤝 Breakout Room Part #2: MCP Workflow Through the Agent
  - Task 6: Create a GitHub Repository
  - Task 7: Commit the Summary
  - Task 8: Create a Feature Branch & Add Metadata
  - Task 9: Open a Pull Request
  - Task 10: Commit the X API Script
  - Task 11: Update the README
  - **Activity #1: Multi-Account Comparison Pipeline**

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

>NOTE: This can be done in place of the Main Assignment

### Social Listening → GitHub Issue Triage Pipeline

Build an automated pipeline that bridges **social media signals to GitHub project management** — all orchestrated through your LangGraph agent.

**Requirements:**

1. **Social Listening**: Use the X API tools to search for posts mentioning a specific open-source project or technology of your choice (e.g., `"langchain bug"`, `"langgraph feature request"`, `"#llamaindex issue"`). Retrieve at least 20 posts.

2. **Post Classification**: Instruct the agent to classify each retrieved post into one of these categories:
   - `bug_report` — user is reporting a problem
   - `feature_request` — user is suggesting a new capability
   - `question` — user is asking for help
   - `praise` — user is sharing positive feedback
   - `general` — none of the above

   The agent should output structured reasoning for each classification.

3. **Automated Issue Creation**: For posts classified as `bug_report` or `feature_request`, use the GitHub MCP tools to automatically create issues in a new repository. Each issue should include:
   - A descriptive title summarizing the social media post
   - Body containing: original post text, classification category, the agent's reasoning, and a suggested priority level (`low` / `medium` / `high`)
   - Appropriate label(s) matching the classification

4. **Digest Report**: Ask the agent to generate a `social-listening-digest.md` that summarizes:
   - Total posts analyzed and breakdown by category
   - Issues created (with links)
   - Sentiment overview across all posts
   - Recommended action items for maintainers

5. **Full Git Workflow**: Use the MCP tools to commit the digest on a new branch and open a PR with a meaningful description.

When submitting, provide:
- Your Loom video link demonstrating the pipeline end-to-end
- The GitHub URL to your completed Advanced Build repository (with issues and PRs visible)

Have fun!
</details>





# Ship 🚢

- The completed notebook.
- 5min. Loom Video
- The GitHub URL to your `x-post-summarizer-2026` repository

# Share 🚀

- Walk through your notebook and explain what you've completed in the Loom video
- Make a social media post about your final application and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned

# Submitting Your Homework

### Main Homework Assignment

Follow these steps to prepare and submit your homework:

1. Pull the latest updates from upstream into the main branch of your AIE9 repo:
  - *(You should have completed this process already.)* For your initial repo setup, see [Initial_Setup](https://github.com/AI-Maker-Space/AIE9/tree/main/00_Docs/Prerequisites/Initial_Setup)
    - To get the latest updates from AI Makerspace into your own AIE9 repo, run the following commands:
    ```
    git checkout main
    git pull upstream main
    git push origin main
    ```
2. **IMPORTANT:** Start Cursor from the `14_MCP_Connectors` folder (you can also use the *File -> Open Folder* menu option of an existing Cursor window)
3. Answer Questions 1 - 3 using the `##### Answer:` markdown cell below them in `Session_14_MCP_Connectors.ipynb`
4. Complete Activity #1 in Breakout Room #1 and Activity #1 in Breakout Room #2 in `Session_14_MCP_Connectors.ipynb`
5. Add, commit and push your modified notebooks to your GitHub repository.

When submitting your homework, provide:

- Your Loom video link
- The GitHub URL to the `14_MCP_Connectors` folder on your assignment branch
- The GitHub URL to your `x-post-summarizer-2026` repository

