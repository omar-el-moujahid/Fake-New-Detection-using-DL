# Multimodal Fake News Detection

Comparative study of fake news detection across text, image, and multimodal approaches using transformer architectures and deep learning.

## Results

| Model | Dataset | F1-Score |
|-------|---------|----------|
| SBERT end-to-end (no softmax) | COVID-19 | **0.98** |
| SBERT end-to-end | ISOT | 0.97 |
| BERT + CNN | ISOT | 0.97 |
| BERT + Random Forest | ISOT | 0.96 |
| BERT + Random Forest | COVID-19 | 0.91 |
| CNN (image only) | Real/Fake Images | 97.1% acc |
| Multimodal BERT + Faster R-CNN | Fakeddit | 0.63 F1 |

## Architecture

Three modalities compared:
- **Text**: BERT+SVM, BERT+CNN, BERT+Random Forest, SBERT end-to-end, LLaMA
- **Image**: CNN from scratch vs ViT-B/16 fine-tuned
- **Multimodal**: BERT + Faster R-CNN with cross-attention fusion (inspired by MFAE)

## Datasets
- ISOT Fake News Dataset
- COVID-19 Fake News Dataset  
- Real/Fake Images Dataset
- Fakeddit (multimodal, Reddit-based)

## Tech Stack
PyTorch · Hugging Face Transformers · scikit-learn · Flask · Docker

## Notebooks
All implementations are in the 
otebooks/ folder.
