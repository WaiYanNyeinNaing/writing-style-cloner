import streamlit as st
import re
import subprocess
import asyncio
from crawl4ai import AsyncWebCrawler
import nltk
from statistics import mean
import pandas as pd
import random
import emoji
from nltk.sentiment import SentimentIntensityAnalyzer
import sys

# NLTK setup
nltk.download('punkt')
nltk.download('vader_lexicon')

# Fix for Windows asyncio
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

STYLE_REFERENCE_INSTRUCTION = "(for style reference only, do not use its content)"

### Utility Functions
def markdown_to_text(md):
    """Convert markdown text to plain text by removing markdown syntax."""
    return re.sub(r'[#*`\[\]]', '', md).strip()

async def fetch_article_text(url):
    """Fetch and extract text from a URL using AsyncWebCrawler."""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return markdown_to_text(result.markdown) if result.success else None

def count_emojis(text):
    """Count the number of emojis in a text string."""
    return len(emoji.emoji_list(text))

def analyze_style(text):
    """Analyze the style of a given text based on various metrics."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    sentences = nltk.sent_tokenize(text)
    words = text.lower().split()
    total_emojis = sum(count_emojis(sent) for sent in sentences)
    excl_count = sum(1 for sent in sentences if sent.strip().endswith('!'))
    quest_count = sum(1 for sent in sentences if sent.strip().endswith('?'))
    sia = SentimentIntensityAnalyzer()
    
    return {
        "avg_lines_per_paragraph": mean([p.count('\n') + 1 for p in paragraphs]) if paragraphs else 0,
        "avg_sentence_length": mean([len(s.split()) for s in sentences]) if sentences else 0,
        "lexical_density": len(set(words)) / len(words) if words else 0,
        "tone": "positive" if sia.polarity_scores(text)['compound'] > 0.5 else "neutral",
        "avg_emojis_per_sentence": total_emojis / len(sentences) if sentences else 0,
        "prop_excl": excl_count / len(sentences) if sentences else 0,
        "prop_quest": quest_count / len(sentences) if sentences else 0,
        "sample_text": text
    }

def generate_system_prompt(style, weights, input_lines, qual_feedback=None):
    """Generate a system prompt that ensures content is from input text and style from reference."""
    prompt = f"""
    Paraphrase the input text to match the following style characteristics while preserving its original meaning and factual content:
    - Average lines per paragraph: {style['avg_lines_per_paragraph']:.1f} (importance: {weights['paragraph_lines']:.2f})
    - Average sentence length: {style['avg_sentence_length']:.1f} words (importance: {weights['sentence_length']:.2f})
    - Lexical density: {style['lexical_density']:.2f} (importance: {weights['lexical_density']:.2f})
    - Tone: {style['tone']} (importance: {weights['tone']:.2f})
    - Average emojis per sentence: {style['avg_emojis_per_sentence']:.2f} (importance: {weights['emojis']:.2f})
    - Proportion of sentences ending with '!': {style['prop_excl']:.2f} (importance: {weights['excl']:.2f})
    - Proportion of sentences ending with '?': {style['prop_quest']:.2f} (importance: {weights['quest']:.2f})
    - Sample text for style reference only (do not use its content): "{style['sample_text']}"

    Instructions:
    - Rewrite the input text to reflect the above style characteristics.
    - Ensure that the rewritten text is a paraphrase of the input text, maintaining its original meaning and factual content.
    - Do not include any information or phrasing from the sample text in your rewrite.
    - Structure paragraphs to have approximately {style['avg_lines_per_paragraph']:.1f} lines each.
    - Use emojis at a rate of about {style['avg_emojis_per_sentence']:.2f} per sentence.
    - End approximately {style['prop_excl']:.2f} of sentences with '!' and {style['prop_quest']:.2f} with '?'.
    - Use double newlines to separate paragraphs.

    The input text has an average of {input_lines:.1f} lines per paragraph. Adjust the paragraph structure accordingly while keeping the content intact.
    """
    if qual_feedback:
        prompt += f"\nPrevious feedback: {qual_feedback}\nPlease address this feedback in your rewrite."
    return prompt

def run_ollama(prompt, input_text):
    """Run the Ollama language model with the given prompt and input text."""
    try:
        result = subprocess.run(['ollama', 'run', 'deepseek-r1'], input=f"{prompt}\n\n{input_text}", text=True, capture_output=True, timeout=60)
        return result.stdout.strip()
    except Exception as e:
        st.error(f"Error running Ollama: {e}")
        return None

def evaluate_style_similarity(original_style, generated_text):
    """Evaluate how similar the generated text's style is to the target style."""
    gen_style = analyze_style(generated_text)
    features = [
        ("avg_lines_per_paragraph", 0.15),
        ("avg_sentence_length", 0.15),
        ("lexical_density", 0.15),
        ("tone", 0.15),
        ("avg_emojis_per_sentence", 0.15),
        ("prop_excl", 0.1),
        ("prop_quest", 0.1)
    ]
    score = 0
    for feat, weight in features:
        if feat == "tone":
            score += weight * (1 if original_style[feat] == gen_style[feat] else 0)
        else:
            diff = abs(original_style[feat] - gen_style[feat])
            max_val = max(original_style[feat], 1)
            similarity = 1 - diff / max_val
            score += weight * similarity
    return score, gen_style

def llm_qualitative_evaluation(original_input, reference, generated, orig_style, gen_style):
    """Perform qualitative evaluation ensuring content preservation and style matching."""
    prompt = f"""
    Original Input Text:
    {original_input}

    Reference Text (for style only):
    {reference}

    Generated Text:
    {generated}

    First, check if the generated text contains any sentences or phrases directly copied from the reference text.
    If copying is detected, respond with 'COPYING DETECTED' followed by an explanation of which parts are copied, and assign a score of 0.00.

    If no copying is detected:
    1. Evaluate how well the generated text preserves the meaning and factual content of the original input text. Consider if all key points are included and if the information is accurately represented. Assign a content preservation score between 0 and 1.
    2. Evaluate how well the generated text matches the style of the reference text based on the following parameters:
       - Paragraph lines: {orig_style['avg_lines_per_paragraph']:.1f} vs {gen_style['avg_lines_per_paragraph']:.1f}
       - Sentence length: {orig_style['avg_sentence_length']:.1f} vs {gen_style['avg_sentence_length']:.1f} words
       - Lexical density: {orig_style['lexical_density']:.2f} vs {gen_style['lexical_density']:.2f}
       - Tone: {orig_style['tone']} vs {gen_style['tone']}
       - Emojis per sentence: {orig_style['avg_emojis_per_sentence']:.2f} vs {gen_style['avg_emojis_per_sentence']:.2f}
       - Sentences ending with '!': {orig_style['prop_excl']:.2f} vs {gen_style['prop_excl']:.2f}
       - Sentences ending with '?': {orig_style['prop_quest']:.2f} vs {gen_style['prop_quest']:.2f}
       Assign a style match score between 0 and 1.

    Respond with 'NO COPYING DETECTED', followed by 'Content Preservation Score: X.XX', 'Style Match Score: Y.YY', and a brief explanation.
    """
    try:
        result = subprocess.run(['ollama', 'run', 'qwen2'], input=prompt, text=True, capture_output=True, timeout=120)
        response = result.stdout.strip()
        if response.startswith("COPYING DETECTED"):
            copying_detected = True
            feedback = response[len("COPYING DETECTED"):].strip()
            score = 0.00
        else:
            copying_detected = False
            feedback = response
            content_score_match = re.search(r'Content Preservation Score:\s*(\d\.\d{2})', feedback)
            style_score_match = re.search(r'Style Match Score:\s*(\d\.\d{2})', feedback)
            content_score = float(content_score_match.group(1)) if content_score_match else 0.5
            style_score = float(style_score_match.group(1)) if style_score_match else 0.5
            score = (content_score + style_score) / 2  # Average of content and style scores
        return feedback, score, copying_detected
    except Exception as e:
        st.error(f"Qualitative evaluation failed: {e}")
        return "Evaluation failed", None, False

def calculate_reward(quant_score, qual_score):
    """Calculate the combined reward from quantitative and qualitative scores."""
    return (0.4 * quant_score + 0.6 * qual_score) if qual_score is not None else quant_score

### Main Application
def main():
    st.title("Writing Style Cloner with RL (SPSA)")
    st.write("Optimizes style to match reference text while preserving input text content.")

    # Initialize session state variables
    if 'style' not in st.session_state:
        st.session_state.style = None
    if 'reference_text' not in st.session_state:
        st.session_state.reference_text = None
    if 'input_text' not in st.session_state:
        st.session_state.input_text = None
    if 'best_text' not in st.session_state:
        st.session_state.best_text = None
    if 'best_reward' not in st.session_state:
        st.session_state.best_reward = -float('inf')

    #### Step 1: Provide Reference Text
    st.subheader("Step 1: Provide Reference Text (Target Style)")
    source = st.radio("Reference source:", ["URL", "Text"])
    if source == "URL":
        url = st.text_input("Enter URL:")
        if st.button("Analyze URL") and url:
            with st.spinner("Fetching and analyzing text..."):
                text = asyncio.run(fetch_article_text(url))
                if text:
                    st.session_state.reference_text = text
                    st.session_state.style = analyze_style(text)
                    st.json(st.session_state.style)
                else:
                    st.error("Failed to fetch text from URL.")
    else:
        text = st.text_area("Paste reference text:", height=150)
        if st.button("Analyze Text") and text:
            st.session_state.reference_text = text
            st.session_state.style = analyze_style(text)
            st.json(st.session_state.style)

    #### Step 2: Upload Text to Rewrite
    st.subheader("Step 2: Upload Text to Rewrite (Content Source)")
    uploaded_file = st.file_uploader("Upload text file", type=["txt"])
    if uploaded_file and st.session_state.style:
        input_text = uploaded_file.read().decode("utf-8")
        st.session_state.input_text = input_text
        st.session_state.best_text = input_text  # Initialize best_text with original input
        input_lines = analyze_style(input_text)["avg_lines_per_paragraph"]
        st.text_area("Input Text", input_text, height=150)
        st.write(f"Input lines per paragraph: {input_lines:.1f}")

        if st.button("Optimize Style"):
            with st.spinner("Optimizing with SPSA..."):
                # Optimization parameters
                max_iterations = 5
                c = 0.1
                alpha = 0.05
                weights = {
                    "paragraph_lines": 0.5, "sentence_length": 0.5, "lexical_density": 0.5,
                    "tone": 0.5, "emojis": 0.5, "excl": 0.5, "quest": 0.5
                }
                feedback = None
                optimization_data = []
                best_rewards = []
                eval_step = 0
                viz_placeholder = st.empty()

                # Optimization loop with checkpointing
                for iteration in range(1, max_iterations + 1):
                    st.write(f"**Iteration {iteration}**")

                    # Use the best_text as the current input for generation
                    current_input = st.session_state.best_text

                    # Perturb weights for SPSA
                    delta = {k: random.choice([-1, 1]) for k in weights}
                    w_plus = {k: min(max(weights[k] + c * delta[k], 0), 1) for k in weights}
                    w_minus = {k: min(max(weights[k] - c * delta[k], 0), 1) for k in weights}

                    # Evaluate w_plus
                    st.write("Evaluating w_plus...")
                    prompt_plus = generate_system_prompt(st.session_state.style, w_plus, input_lines, feedback)
                    text_plus = run_ollama(prompt_plus, current_input)
                    if text_plus:
                        quant_plus, gen_style_plus = evaluate_style_similarity(st.session_state.style, text_plus)
                        qual_feedback_plus, qual_score_plus, copying_detected_plus = llm_qualitative_evaluation(
                            st.session_state.input_text, st.session_state.reference_text, text_plus, st.session_state.style, gen_style_plus
                        )
                        if copying_detected_plus:
                            reward_plus = -1.0
                        else:
                            reward_plus = calculate_reward(quant_plus, qual_score_plus)
                        st.write(f"Reward: {reward_plus:.2f}")
                        # Update checkpoint if reward improves
                        if reward_plus > st.session_state.best_reward:
                            st.session_state.best_reward = reward_plus
                            st.session_state.best_text = text_plus
                        eval_step += 1
                        optimization_data.append({
                            "Step": eval_step, "Iteration": iteration, "Type": "w_plus",
                            "Reward": reward_plus, "Best Reward": st.session_state.best_reward
                        })
                        best_rewards.append(st.session_state.best_reward)

                    # Evaluate w_minus
                    st.write("Evaluating w_minus...")
                    prompt_minus = generate_system_prompt(st.session_state.style, w_minus, input_lines, feedback)
                    text_minus = run_ollama(prompt_minus, current_input)
                    if text_minus:
                        quant_minus, gen_style_minus = evaluate_style_similarity(st.session_state.style, text_minus)
                        qual_feedback_minus, qual_score_minus, copying_detected_minus = llm_qualitative_evaluation(
                            st.session_state.input_text, st.session_state.reference_text, text_minus, st.session_state.style, gen_style_minus
                        )
                        if copying_detected_minus:
                            reward_minus = -1.0
                        else:
                            reward_minus = calculate_reward(quant_minus, qual_score_minus)
                        st.write(f"Reward: {reward_minus:.2f}")
                        # Update checkpoint if reward improves
                        if reward_minus > st.session_state.best_reward:
                            st.session_state.best_reward = reward_minus
                            st.session_state.best_text = text_minus
                        eval_step += 1
                        optimization_data.append({
                            "Step": eval_step, "Iteration": iteration, "Type": "w_minus",
                            "Reward": reward_minus, "Best Reward": st.session_state.best_reward
                        })
                        best_rewards.append(st.session_state.best_reward)

                    # Gradient approximation and weights update
                    grad = {k: (reward_plus - reward_minus) / (2 * c * delta[k]) if delta[k] != 0 else 0 for k in weights}
                    weights = {k: min(max(weights[k] + alpha * grad[k], 0), 1) for k in weights}

                    # Evaluate updated weights
                    st.write("Evaluating current weights...")
                    prompt_current = generate_system_prompt(st.session_state.style, weights, input_lines, feedback)
                    text_current = run_ollama(prompt_current, current_input)
                    if text_current:
                        quant_current, gen_style_current = evaluate_style_similarity(st.session_state.style, text_current)
                        qual_feedback, qual_score, copying_detected = llm_qualitative_evaluation(
                            st.session_state.input_text, st.session_state.reference_text, text_current, st.session_state.style, gen_style_current
                        )
                        if copying_detected:
                            reward_current = -1.0
                        else:
                            reward_current = calculate_reward(quant_current, qual_score)
                        st.write(f"Reward: {reward_current:.2f}")
                        # Update checkpoint if reward improves
                        if reward_current > st.session_state.best_reward:
                            st.session_state.best_reward = reward_current
                            st.session_state.best_text = text_current
                        eval_step += 1
                        optimization_data.append({
                            "Step": eval_step, "Iteration": iteration, "Type": "current",
                            "Reward": reward_current, "Best Reward": st.session_state.best_reward
                        })
                        best_rewards.append(st.session_state.best_reward)
                        feedback = qual_feedback

                    # Visualization update
                    if optimization_data:
                        df = pd.DataFrame(optimization_data)
                        best_rewards_df = pd.DataFrame({"Best Reward": best_rewards})
                        with viz_placeholder.container():
                            st.write("### Optimization Progress")
                            st.line_chart(best_rewards_df, height=300)
                            st.write("### Optimization Log")
                            st.dataframe(df[["Step", "Iteration", "Type", "Reward", "Best Reward"]])
                    else:
                        st.warning("No optimization data available yet.")

                #### Step 3: Display Final Optimized Text
                st.subheader("Step 3: Final Optimized Text")
                if st.session_state.best_text:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area("Original Text", st.session_state.input_text, height=200)
                    with col2:
                        st.text_area("Optimized Text", st.session_state.best_text, height=200)
                    st.write(f"Best Reward: {st.session_state.best_reward:.2f}")
                else:
                    st.error("Optimization failed to produce a valid result.")

if __name__ == "__main__":
    main()