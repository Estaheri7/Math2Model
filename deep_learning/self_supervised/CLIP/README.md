# CLIP: Contrastive Language–Image Pre-training

CLIP is a model developed by OpenAI that learns **visual concepts from natural language supervision**. It jointly trains an image encoder and a text encoder to project both modalities into a **shared embedding space**, where **similarity between image and text embeddings is maximized** for matching pairs.

---

## Overview

CLIP learns a **joint embedding space** for images and texts using **contrastive learning**. The idea is to align the image and its corresponding text (caption/label) while pushing apart unrelated image-text pairs.

---

## Architecture

CLIP consists of:

- **Image Encoder**: Usually a Vision Transformer (ViT) or ResNet that encodes an image into a feature vector.
- **Text Encoder**: A Transformer that encodes a textual description into a feature vector.
- **Contrastive Loss**: Cross-modal contrastive loss is used to train the encoders jointly.
