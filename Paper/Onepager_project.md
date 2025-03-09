# Writing Style Cloner with RL (SPSA): A Robust Approach to Style Replication

**Author**: Dr. Wai Yan, [https://github.com/WaiYanNyeinNaing]  
**Date**: 2025

---

## Abstract
Current large language models (LLMs) struggle to replicate human-like writing styles across diverse contexts due to unnatural tone, reliance on large datasets, and extensive fine-tuning requirements. The "Writing Style Cloner with RL (SPSA)" introduces a novel solution using reinforcement learning (RL) with Simultaneous Perturbation Stochastic Approximation (SPSA). By analyzing a reference text and optimizing style features—sentence length, lexical density, tone, and structure—the system efficiently clones styles without needing vast training data, offering a flexible and robust approach to style replication.

---

## Introduction
Writing style shapes communication, varying by audience, purpose, and context (e.g., casual blogs vs. formal reports). Existing LLMs face challenges in dynamically adapting styles, often requiring complex prompt engineering or fine-tuning on large datasets. The "Writing Style Cloner with RL (SPSA)" overcomes these limitations by combining NLP-based style analysis with an RL optimization framework, enabling real-time style replication from minimal input.

---

## Methodology
The system is implemented as a Streamlit application with three key components:

### Style Analysis
Using NLTK, the system extracts style features from a reference text (via URL or input):
- Average sentence length
- Lexical density (unique words ratio)
- Tone (via SentimentIntensityAnalyzer)
- Paragraph structure (lines per paragraph)

### Text Generation
The Ollama 'qwen2' model generates styled text based on a system prompt incorporating weighted style parameters, which dictate the influence of each feature.

### Optimization with SPSA
SPSA refines the weights iteratively to maximize a reward function:
- **Reward**: Combines quantitative similarity (40%)—matching style metrics—and qualitative feedback (60%)—LLM-assessed style alignment and originality.
- **Process**: Perturbs weights (`w_plus`, `w_minus`), evaluates rewards, approximates gradients, and updates weights over iterations (e.g., 5).

This ensures the generated text progressively aligns with the reference style.

---

## Results
The tool effectively transforms text styles, such as converting a casual blog post into a formal academic tone, while preserving content. Real-time metrics show reward improvement, highlighting its optimization success.

### Sample Reward Progression
| **Iteration** | **w_plus Reward** | **w_minus Reward** | **Current Reward** | **Best Reward** |
|---------------|-------------------|--------------------|--------------------|-----------------|
| 1             | 0.78              | 0.72               | 0.80               | 0.80            |
| 2             | 0.82              | 0.76               | 0.84               | 0.84            |
| 3             | 0.85              | 0.79               | 0.87               | 0.87            |
| 4             | 0.88              | 0.82               | 0.89               | 0.89            |
| 5             | 0.90              | 0.84               | 0.91               | 0.91            |

The reward rises from 0.80 to 0.91, demonstrating effective style tuning.

---

## Discussion
The SPSA-based approach offers distinct advantages:
- **No Dataset Dependency**: Works with user-provided references.
- **Natural Tone**: Iterative optimization enhances style authenticity.
- **Efficiency**: Reduces reliance on manual prompt adjustments.

Challenges include computational costs from repeated LLM calls and reliance on the 'qwen2' model. Future enhancements could involve faster algorithms or expanded style metrics.

---

## Conclusion
The "Writing Style Cloner with RL (SPSA)" provides a versatile, efficient solution for style replication, addressing LLM shortcomings. Its open-source nature encourages further development, making it a valuable tool for diverse writing applications.