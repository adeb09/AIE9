# Impact of Chunk Size on RAG Performance

## Overview

Chunk size is a critical parameter in RAG systems that significantly affects retrieval quality and answer generation. Here's how it impacts performance across different question types.

## Precision and Recall Metrics

In RAG systems, precision and recall are key metrics for evaluating retrieval quality, and chunk size significantly impacts both:

### Precision
**What it measures:** Of all the chunks you retrieve, how many are actually relevant?

**Chunk size impact:**
- **Smaller chunks** → Higher precision: Each chunk is more focused and specific. When you retrieve a 100-word chunk, most of it is likely on-topic.
- **Larger chunks** → Lower precision: A 1000-word chunk may contain the answer, but also lots of irrelevant context that dilutes the signal.

### Recall
**What it measures:** Of all the relevant information in your corpus, how much did you successfully retrieve?

**Chunk size impact:**
- **Larger chunks** → Higher recall: More likely to capture complete context and related information. If the answer is anywhere in a section, the whole section gets retrieved.
- **Smaller chunks** → Lower recall: You might miss relevant information because:
  - The answer spans multiple chunks
  - Important context got split away
  - The specific chunk with your answer doesn't match the query well enough

### The Precision-Recall Tradeoff

This creates a classic precision-recall tradeoff:

```
Small chunks: High precision, Low recall
↕
Large chunks: Low precision, High recall
```

**In practice:**
- Use **smaller chunks** (128-256 tokens) when you need precise, focused answers
- Use **larger chunks** (512-1024 tokens) when you need comprehensive context
- Many systems use **medium chunks with overlap** or **hierarchical retrieval** to balance both metrics

The optimal chunk size depends on your specific use case, document structure, and whether missing information (low recall) or noisy results (low precision) is more problematic.

## Chunk Size Trade-offs

### Smaller chunks (e.g., 128-256 tokens)
- Higher precision: More focused, relevant content per chunk
- Better semantic matching: Embeddings represent specific concepts
- More granular retrieval: Can pinpoint exact information
- But: May lose context, require more chunks to answer questions

### Larger chunks (e.g., 512-1024 tokens)
- More context per chunk: Preserve relationships between ideas
- Fewer retrieval calls needed
- But: Lower precision, embeddings may be diluted, harder to rank relevance

## Impact on Single-Hop Questions

**Single-hop questions** require information from one source/chunk to answer (e.g., "What is the capital of France?")

- **Smaller chunks work well**: The answer typically lives in a focused passage
- Retrieval can precisely target the relevant chunk
- Less noise in the retrieved context
- Performance is generally good with moderate chunk sizes (256-512 tokens)

## Impact on Multi-Hop Questions

**Multi-hop questions** require synthesizing information across multiple sources/chunks (e.g., "What is the population of the country where the Eiffel Tower is located?")

This is where chunk size becomes critical:

### With smaller chunks
- ✅ Better at retrieving each individual fact (e.g., "Eiffel Tower is in France" + "France population is X")
- ❌ May need top-k=10+ to capture all necessary information
- ❌ Facts might be split across chunks, breaking logical connections
- ❌ LLM must do more synthesis work from disconnected pieces

### With larger chunks
- ✅ More likely to capture related information in a single chunk
- ✅ Preserves causal relationships and reasoning chains
- ❌ May retrieve irrelevant information along with relevant facts
- ❌ Harder for embeddings to match when query relates to small portion of chunk

## Optimal Strategies

For multi-hop questions specifically:

1. **Hierarchical chunking**: Use both small (for precision) and large chunks (for context)
2. **Sliding windows with overlap**: 30-50% overlap helps preserve context across boundaries
3. **Dynamic chunk sizing**: Respect document structure (don't split mid-paragraph or mid-table)
4. **Higher top-k retrieval**: Retrieve 5-10 chunks for multi-hop vs 1-3 for single-hop
5. **Re-ranking**: Use a cross-encoder to re-rank larger candidate sets

## In Your Test Sets

When evaluating with synthetic data:
- **Single-hop tests**: Will be relatively stable across chunk sizes (256-512 works well)
- **Multi-hop tests**: Performance may degrade significantly with small chunks (<256 tokens)
- Sweet spot for balanced performance: **512 tokens with 128-token overlap**
- If multi-hop performance is poor, try increasing to 768-1024 tokens or implement hierarchical retrieval

## Key Insight

**Multi-hop questions expose chunk size limitations** because they require the system to either retrieve multiple disconnected chunks or have chunks large enough to contain reasoning chains.
