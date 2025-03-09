# Writing Style Cloner with RL (SPSA)

## 🚀 Overview
"Writing Style Cloner with RL (SPSA)" is an innovative AI tool designed to analyze and replicate the writing style of a given reference text. Using natural language processing (NLP) and a reinforcement learning (RL)-inspired algorithm called Simultaneous Perturbation Stochastic Approximation (SPSA), this project transforms input text to match stylistic features such as sentence length, lexical density, tone, and paragraph structure. Developed by Dr. Wai Yan, this tool is ideal for writers, content creators, and researchers aiming to adapt their writing to specific styles or contexts.

## 📄 Research Paper
This project employs cutting-edge techniques in style analysis and reinforcement learning. While a formal research paper is not yet available, a one-page summary is provided below. Contributions and further research are encouraged.

## 🔍 Abstract
This project presents a novel approach to writing style replication by integrating style analysis with an RL-inspired SPSA algorithm. It extracts stylistic features from a reference text and optimizes the rewriting of input text to align with these features, overcoming challenges in current large language models (LLMs) like unnatural tone generation and context adaptability. The system operates without extensive fine-tuning or large datasets, offering a robust solution for style cloning across diverse occasions and topics.

## 📚 Datasets
The project does not use predefined datasets. Instead, it processes user-provided reference texts (via URL or direct input) and input texts in real-time, making it highly flexible and user-driven.

## 🛠️ Installation
Follow these steps to set up the project locally:

1. Clone the repository:
   git clone https://github.com/WaiYanNyeinNaing/writing-style-cloner.git
   cd writing-style-cloner

2. Install the required Python packages:
   pip install -r requirements.txt

3. Install Playwright browsers (required for web crawling):
   playwright install

4. Install Ollama and download the 'qwen2' model:
   ollama pull qwen2

5. Ensure your system supports Ollama and the language model.

The `requirements.txt` should include:
streamlit
crawl4ai
nltk
pandas
playwright

## ⚙️ Usage
Run the Streamlit app with:
streamlit run app.py

Then:
1. Select a reference source (URL or text input).
2. Click "Analyze" to extract the reference style.
3. Upload a .txt file to rewrite.
4. Click "Generate Styled Text" to optimize and view the result.

The app displays the optimization process and final styled text.

## 📈 Results
The tool provides real-time feedback, including:
- Style similarity metrics (e.g., sentence length, lexical density).
- Qualitative evaluations from the 'qwen2' model.
- Optimization logs showing reward improvements.

Users can see how SPSA enhances style matching iteratively, producing text that closely mirrors the reference style.

## 📖 Citation
Cite this project as:
@misc{waiyan2025,
  title={Writing Style Cloner with RL (SPSA)},
  author={Wai Yan},
  year={2025},
  url={https://github.com/WaiYanNyeinNaing/writing-style-cloner}
}

## 🌟 Contribution
We welcome contributions! Submit issues, fork the repository, or send pull requests via GitHub. For significant changes, please open an issue to discuss first.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact
- [GitHub Profile](https://github.com/WaiYanNyeinNaing)