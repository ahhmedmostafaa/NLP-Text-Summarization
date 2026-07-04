# Extractive Text Summarization (TF-IDF)

An extractive text summarization pipeline built and evaluated on the CNN/Daily Mail news dataset — a standard benchmark for summarization tasks.

## How it works
1. Loads the **CNN/Daily Mail** dataset via Hugging Face `datasets`
2. Splits each article into sentences
3. Builds a **TF-IDF** vector representation of the sentences
4. Computes pairwise **cosine similarity** between sentences to score each one by overall relevance to the article
5. Selects the top-ranked sentences as the extractive summary
6. Processes the full train/test splits and saves original + summarized pairs to CSV

## Dataset
[CNN/Daily Mail (abisee/cnn_dailymail)](https://huggingface.co/datasets/abisee/cnn_dailymail) — a standard benchmark dataset for summarization tasks, containing news articles paired with human-written summaries.

## Tech Stack
- Python
- Hugging Face `datasets`
- Scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
- Pandas, NumPy

## Files
- `text_summarize_project.ipynb` — full pipeline
- `description of Project.docx` — project write-up
- `article *.txt` — sample article outputs for inspection

## How to Run
```bash
pip install datasets scikit-learn pandas numpy
jupyter notebook text_summarize_project.ipynb
```

## Roadmap
- [ ] Add ROUGE score evaluation against reference summaries
- [ ] Compare against an abstractive summarization baseline (e.g., a pretrained transformer)
- [ ] Make summary length (number of sentences) configurable
