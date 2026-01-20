<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

<h1 align="center" id="heading">Session 1: Introduction and Vibe Check</h1>

### [Quicklinks](https://github.com/AI-Maker-Space/AIE9/tree/main/00_AIM_Quicklinks)

| 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Vibe Check!](https://github.com/AI-Maker-Space/AIE9/blob/main/00_AIM_Docs/Session_Sheets/01_Vibe_Check.md) | Coming Soon! | Coming Soon! | You are here! | Coming Soon! | Coming Soon! |

## 🏗️ How AIM Does Assignments

> 📅 **Assignments will always be released to students as live class begins.** We will never release assignments early.

Each assignment will have a few of the following categories of exercises:

- ❓ **Questions** – these will be questions that you will be expected to gather the answer to! These can appear as general questions, or questions meant to spark a discussion in your breakout rooms!

- 🏗️ **Activities** – these will be work or coding activities meant to reinforce specific concepts or theory components.

- 🚧 **Advanced Builds (optional)** – Take on a challenge! These builds require you to create something with minimal guidance outside of the documentation. Completing an Advanced Build earns full credit in place of doing the base assignment notebook questions/activities.

### Main Assignment

In the following assignment, you are required to take the app that you created for the AIE9 challenge (from [this repository](https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge)) and conduct what is known, colloquially, as a "vibe check" on the application.

You will be required to submit a link to your GitHub, as well as screenshots of the completed "vibe checks" through the provided Google Form!

> NOTE: This will require you to make updates to your personal class repository, instructions on that process can be found [here](https://github.com/AI-Maker-Space/AIE9/tree/main/00_Initial_Setup)!


#### A Note on Vibe Checking

>"Vibe checking" is an informal term for cursory unstructured and non-comprehensive evaluation of LLM-powered systems. The idea is to loosely evaluate our system to cover significant and crucial functions where failure would be immediately noticeable and severe.
>
>In essence, it's a first look to ensure your system isn't experiencing catastrophic failure.

---

#### 🏗️ Activity #1: General Vibe Checking Evals

Please evaluate your system on the following questions:

1. Explain the concept of object-oriented programming in simple terms to a complete beginner.
   - **Aspect Tested**:
     * does the LLM just regurgitate/repeat information or can it actually show a level of understanding and explain conceptual things in simpler terms
     * can the LLM adjust how it talks when conveying information (you wouldn't explain OOP to a third-year computer science student the same way you would to a complete beginner)
2. Read the following paragraph and provide a concise summary of the key points…
    - **Aspect Tested**:
    - testing it's ability to summarize and give only essentially information (filtering out noise and parts that aren't as important)
    - is it actually summarizing the article you shown or introducing information that wasn't present (hallucinating)?
3. Write a short, imaginative story (100–150 words) about a robot finding friendship in an unexpected place.
    - **Aspect Tested**:
    - can the LLM actually follow instructions (is the story actually within the 100-150 word limit; is the story actually about a robot finding friendship..etc.)
    - can the LLM actually be creative?
4. If a store sells apples in packs of 4 and oranges in packs of 3, how many packs of each do I need to buy to get exactly 12 apples and 9 oranges?
    - **Aspect Tested**:
    - can it do math
    - it's ability to reason, especially mathematical reasoning and showing its thought process behind it
5. Rewrite the following paragraph in a professional, formal tone…
    - **Aspect Tested**:
    - does the LLM understand what a professional "tone" is
    - can it actually change the "tone" without changing the underlying message of the text
#### ❓Question #1:

Do the answers appear to be correct and useful?
##### ✅ Answer:
#### 1
![1/1](./images/1/1_1.png)
![1/2](./images/1/1_3.png)
![1/3](./images/1/1_4.png)
![1/4](./images/1/1_5.png)
![1/5](./images/1/1_6.png)

Yes, I think the response to this first prompt was pretty useful. Comparing objects to lego pieces was a decent analogy for a true beginner. It gives a pretty concrete example for the analogy and doesn't miss out on keys aspects of an object like properties and methods (the ability to perform actions is also a good analogy for methods). It also mentions OOP as a great way to organize code which is a key piece to mention to beginners even though they won't truly understand this until they actually start writing their own programs. I think this LLM was able to pass this vibe check pretty well.

#### 2
In this second prompt, I gave the LLM the first paragraph of the Wikipedia page on the [Barbary Wars](https://en.wikipedia.org/wiki/Barbary_Wars) to summarize. I think it does a decent job keeping key details about which presidents actually were involved in each Barbary War and the outcome of each war. I think it does a good job of giving the most important parts in that paragraph while simultaneously filtering out information that isn't necessary for the summary. It does a good job of keeping a key piece of information about the Barbary Wars being the first major military entanglement for the US outside of the New World. I believe the LLM passed this vibe check.
![2/1](./images/1/2_1.png)
![2/2](./images/1/2_2.png)
![2/3](./images/1/2_3.png)
![2/4](./images/1/2_4.png)
![2/5](./images/1/2_5.png)
![2/6](./images/1/2_6.png)
![2/7](./images/1/2_7.png)
![2/8](./images/1/2_8.png)


#### 3

The LLM passes this vibe check pretty well. It did follow the directions and kept the story between 100-150 words. The story is about definitely about a robot finding friendship (with a sparrow). I'm not sure why, reading it reminded me of the Iron Giant for some reason. It's subjective about whether this was a creative story but I believe it was (sparrows and robots aren't necessarily a common pairing). This definitely passed my vibe check for sure.
![3/1](./images/1/3_1.png)
![3/2](./images/1/3_2.png)
![3/3](./images/1/3_3.png)

#### 4

The LLM passes this vibe with flying colors. It's able to show it's mathematical reasoning and arrive at the correct answer. LLMs can do math!
![4/1](./images/1/4_1.png)
![4/2](./images/1/4_2.png)

#### 5

For this prompt, I chose a post on  a New England Patriots facebook group (I'm a Pats' fan) which was written pretty informally, especially in terms of punctuation and grammar, and I told the LLM to rewrite the following paragraph in a professional tone. This post talks about how the last 2 times the Patriots played on his birthday in the playoffs, they went to the Super Bowl (alluding to the fact that it may happen this year again). 
- Surprisingly, the "deflategate" is spelled incorrectly by the LLM as "deblategate" even though the spelling was correct in the original post. I think this is an indicator that the LLM didn't understand what "deflategate" and for some reason thought changing the word by 1 letter was more "professional"
- at the end of the post, you can see the writer wrote "Go Pats" which the LLM changed to "Go Patriots" which shows it was trying to make the post be more professional.
- it does a good job of fixing information/incorrect punctuation that is present in most online forum posts
- Overall, I think the LLM did a decent job at making the post sound more professional even though it had some errors

![5/1](./images/1/5_1.png)
![5/2](./images/1/5_2.png)
![5/3](./images/1/5_3.png)
![5/4](./images/1/5_4.png)
![5/5](./images/1/5_5.png)
![5/6](./images/1/5_6.png)
![5/7](./images/1/5_7.png)

- I tried another Reddit post that was talking about a NASA mission and the new photos of Pluto this telescope gathered
- Just like in the first example, it does a good job fixing punctuation and phrasing and does use a more professional "tone"
- it still kept the essence of the original post and key details but changed punctuation and the wording (slightly) to make it sound more professional
- for an example, I pasted the references to the articles that were at the end of the Reddit post saying "read more at...links to article"; the LLM changed this phrasing to "for further information, please refer to" which is definitely a more formal tone
- overall, I think these two prompts show that the LLM is decent at changing "tone" of informal writing (blog posts) although it is not perfect (as in with the deflategate error)

![5/8](./images/1/5_8.png)
![5/9](./images/1/5_9.png)
![5/10](./images/1/5_10.png)
![5/11](./images/1/5_11.png)
![5/12](./images/1/5_12.png)
![5/13](./images/1/5_13.png)
![5/14](./images/1/5_14.png)
---

#### 🏗️ Activity #2: Personal Vibe Checking Evals (Your Assistant Can Answer)

Now test your assistant with personal questions it should be able to help with. Try prompts like:

- "Help me think through the pros and cons of [enter decision you're working on making]."
- "What are the pros and cons of [job A] versus [job B] as the next step in my career?"
- "Draft a polite follow-up [email, text message, chat message] to a [enter person details] who hasn't responded."
- "Help me plan a birthday surprise for [person]."
- "What can I cook with [enter ingredients] in fridge."

##### Your Prompts and Results:
1. Prompt: "Help me think through the pros and cons of replacing a transmission on my 2014 Honda Civic versus just buying a new a new car? Note that I live in NYC so I don't really need my car besides driving between NYC and MA (where my parents live)."
- this was an actual decision I had to make a few months prior so I was curious how the LLM would reason about this decision
- it does a good job of making a pros and cons list for each side of the decision
- it came to very reasonable and similar conclusions that I did going throught this entire process myself
- it was able to estimate the costs for replacing the transmission vs just buying a new car
- it does a good job of taking the context in my promt that I don't drive this car too frequently there are cheaper options such as just renting whenever I do need a car for the weekend
- I actually did end up replacing my transmission since it was the cheaper option in the short term and because my car's mileage was only 128k

 **Result:**
![1/1](./images/2/1_1.png)
![1/2](./images/2/1_2.png)
![1/3](./images/2/1_3.png)
![1/4](./images/2/1_4.png)
![1/5](./images/2/1_5.png)
![1/6](./images/2/1_6.png)
![1/7](./images/2/1_7.png)
![1/8](./images/2/1_8.png)
![1/9](./images/2/1_9.png)
![1/10](./images/2/1_10.png)
![1/11](./images/2/1_11.png)
![1/12](./images/2/1_12.png)
![1/13](./images/2/1_13.png)
![1/14](./images/2/1_14.png)

2. Prompt: "What are the pros and cons of working at a MAANG company versus an early stage AI startup as the next step in my career?"
- this was 
-
**Result:**
![2/1](./images/2/2_1.png)
![2/2](./images/2/2_2.png)
![2/3](./images/2/2_3.png)
![2/4](./images/2/2_4.png)
![2/5](./images/2/2_5.png)
![2/6](./images/2/2_6.png)
![2/7](./images/2/2_7.png)
![2/8](./images/2/2_8.png)
![2/9](./images/2/2_9.png)
![2/10](./images/2/2_10.png)
![2/11](./images/2/2_11.png)
![2/12](./images/2/2_12.png)
![2/13](./images/2/2_13.png)
![2/14](./images/2/2_14.png)
![2/15](./images/2/2_15.png)
![2/16](./images/2/2_16.png)
![2/6](./images/2/2_17.png)
![2/6](./images/2/2_18.png)
![2/6](./images/2/2_19.png)
![2/6](./images/2/2_20.png)
![2/6](./images/2/2_21.png)
![2/6](./images/2/2_22.png)
![2/6](./images/2/2_23.png)
![2/6](./images/2/2_24.png)
![2/6](./images/2/2_25.png)
![2/6](./images/2/2_26.png)
![2/6](./images/2/2_27.png)
![2/6](./images/2/2_28.png)
![2/6](./images/2/2_29.png)
![2/6](./images/2/2_30.png)
![2/6](./images/2/2_31.png)
![2/6](./images/2/2_32.png)
![2/6](./images/2/2_33.png)
![2/6](./images/2/2_34.png)
![2/6](./images/2/2_35.png)
![2/6](./images/2/2_36.png)
![2/6](./images/2/2_37.png)
![2/6](./images/2/2_38.png)
![2/6](./images/2/2_39.png)
![2/6](./images/2/2_40.png)
![2/6](./images/2/2_41.png)
![2/6](./images/2/2_42.png)
![2/6](./images/2/2_43.png)


2. Prompt: "Help me plan a birthday surprise for my sister."
**Result:**
![3/1](./images/2/3_1.png)
![3/2](./images/2/3_2.png)
![3/3](./images/2/3_3.png)
![3/4](./images/2/3_4.png)
![3/5](./images/2/3_5.png)
![3/6](./images/2/3_6.png)
![3/7](./images/2/3_7.png)
![3/8](./images/2/3_8.png)
![3/9](./images/2/3_9.png)
![3/10](./images/2/3_10.png)
![3/11](./images/2/3_11.png)
![3/12](./images/2/3_12.png)
![3/13](./images/2/3_13.png)
![3/14](./images/2/3_14.png)
![3/15](./images/2/3_15.png)
![3/16](./images/2/3_16.png)
![3/17](./images/2/3_17.png)
![3/18](./images/2/3_18.png)
![3/19](./images/2/3_19.png)
![3/20](./images/2/3_20.png)
![3/21](./images/2/3_21.png)
![3/22](./images/2/3_22.png)
![3/23](./images/2/3_23.png)
![3/24](./images/2/3_24.png)
![3/25](./images/2/3_25.png)
![3/26](./images/2/3_26.png)
![3/27](./images/2/3_27.png)
![3/28](./images/2/3_28.png)
![3/29](./images/2/3_29.png)
![3/30](./images/2/3_30.png)
![3/31](./images/2/3_31.png)

#### ❓Question #2:

Are the vibes of this assistant's answers aligned with your vibes? Why or why not?
##### ✅ Answer:

---

#### 🏗️ Activity #3: Personal Vibe Checking Evals (Requires Additional Capabilities)

Now test your assistant with questions that would require capabilities beyond basic chat, such as access to external tools, APIs, or real-time data. Try prompts like:

- "What does my schedule look like tomorrow?"
- "What time should I leave for the airport?"

##### Your Prompts and Results:
1. Prompt:
   - Result:
2. Prompt:
   - Result:

#### ❓Question #3:

What are some limitations of your application?
##### ✅ Answer:

---

This "vibe check" now serves as a baseline, of sorts, to help understand what holes your application has.

### 🚧 Advanced Build (OPTIONAL):

Please make adjustments to your application that you believe will improve the vibe check you completed above, then deploy the changes to your Vercel domain [(see these instructions from your Challenge project)](https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge/blob/main/README.md) and redo the above vibe check.

> NOTE: You may reach for improving the model, changing the prompt, or any other method.

#### 🏗️ Activity #1
##### Adjustments Made:
- _describe adjustment(s) here_

##### Results:
1. _Comment here how the change(s) impacted the vibe check of your system_
2.
3.
4.
5.


## Submitting Your Homework
### Main Assignment
Follow these steps to prepare and submit your homework:
1. Pull the latest updates from upstream into the main branch of your AIE9 repo:
    - For your initial repo setup see [00_Initial_Setup](https://github.com/AI-Maker-Space/AIE9/tree/main/00_Initial_Setup)
    - To get the latest updates from AI Makerspace into your own AIE9 repo, run the following commands:
    ```
    git checkout main
    git pull upstream main
    git push origin main
    ```
2. **IMPORTANT:** Start Cursor from the `01_Prototyping_Best_Practices_and_Vibe_Check` folder (you can also use the _File -> Open Folder_ menu option of an existing Cursor window)
3. Edit this `README.md` file (the one in your `AIE9/01_Prototyping_Best_Practices_and_Vibe_Check` folder)
4. Complete all three Activities:
    - **Activity #1:** Evaluate your system using the general vibe checking questions and define the "Aspect Tested" for each
    - **Activity #2:** Test your assistant with personal prompts it should be able to answer
    - **Activity #3:** Test your assistant with prompts requiring additional capabilities
5. Provide answers to all three Questions (`❓Question #1`, `❓Question #2`, `❓Question #3`)
6. Add, commit and push your modified `README.md` to your origin repository's main branch.

When submitting your homework, provide the GitHub URL to your AIE9 repo.

### The Advanced Build:
1. Follow all of the steps (Steps 1 - 6) of the Main Assignment above
2. Document what you changed and the results you saw in the `Adjustments Made:` and `Results:` sections of the Advanced Build
3. Add, commit and push your additional modifications to this `README.md` file to your origin repository.

When submitting your homework, provide the following on the form:
+ The GitHub URL to your AIE9 repo.
+ The public Vercel URL to your updated Challenge project on your AIE9 repo.
