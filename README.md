---
title: Oral Disease Classifier
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
license: mit
short_description: Oral disease diagnosis + RAG medicine recommendations
---

# Oral Disease Classification System

An AI-powered web application for classifying oral diseases using deep learning.

## Features

- 🔍 Classifies 7 types of oral diseases
- 💊 Provides medicine recommendations
- 📊 Shows confidence scores
- ⚡ Fast inference (~1 second)

## Supported Conditions

1. **CaS** - Canker Sores
2. **CoS** - Cold Sores
3. **Gum** - Gingivostomatitis
4. **MC** - Mouth Cancer
5. **OC** - Oral Cancer
6. **OLP** - Oral Lichen Planus
7. **OT** - Oral Thrush

## Model Information

- **Architecture:** InceptionResNetV2
- **Training Accuracy:** 99.51%
- **Dataset:** Custom oral disease dataset (5,143 images)

## Usage

1. Upload an image of an oral condition
2. Click "Analyze"
3. View the diagnosis and recommendations

## Disclaimer

⚠️ This is an AI-powered tool for informational purposes only. It is NOT a substitute for professional medical diagnosis and treatment. Always consult a qualified healthcare professional.

## Citation

Based on research: "Mouth and oral disease classification using InceptionResNetV2 method"