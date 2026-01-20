import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import pretrainedmodels
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import os

# ===============================
# GEMINI API CONFIG (OLD SDK - WORKS ON SPACES)
# ===============================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully")
else:
    client = None
    print("⚠️ GEMINI_API_KEY missing")

# ===============================
# CLASS DEFINITIONS
# ===============================
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

disease_names = {
    'CaS': 'Canker Sores (Aphthous Ulcers)',
    'CoS': 'Cold Sores (Herpes Labialis)',
    'Gum': 'Gingivostomatitis',
    'MC': 'Mouth Cancer (Suspected)',
    'OC': 'Oral Cancer (Suspected)',
    'OLP': 'Oral Lichen Planus',
    'OT': 'Oral Thrush (Oral Candidiasis)'
}

# ===============================
# SAFE KNOWLEDGE BASE (SUMMARY-ONLY)
# ===============================
medical_knowledge_base = [
    {
        "disease": "CaS",
        "document": """
Canker sores are non-contagious shallow ulcers inside the mouth.
Management focuses on reducing irritation, using soothing mouth rinses, 
maintaining oral hygiene, and avoiding spicy or acidic foods. 
Medical review is recommended if sores persist, recur frequently, or become large.
"""
    },
    {
        "disease": "CoS",
        "document": """
Cold sores result from herpes simplex virus infection and appear on or around the lips.
Management includes early use of antiviral creams, keeping lips moisturized,
and avoiding sun exposure or stress triggers. 
Seek medical care if outbreaks are frequent or severe.
"""
    },
    {
        "disease": "Gum",
        "document": """
Gingivostomatitis causes inflammation of gums and oral tissues.
Care includes pain relief, hydration, gentle oral hygiene, and rest.
If symptoms include high fever, difficulty swallowing, or dehydration,
professional medical attention is important.
"""
    },
    {
        "disease": "MC",
        "document": """
Suspected mouth cancer requires urgent professional evaluation.
Diagnosis is performed by specialists through examination and biopsy.
Treatment varies depending on stage and may involve surgery or other therapies.
Early referral is essential.
"""
    },
    {
        "disease": "OC",
        "document": """
Oral cancer is a serious condition requiring specialist assessment.
Management decisions depend on staging and patient factors.
This tool does not provide treatment plans for cancer; early referral is critical.
"""
    },
    {
        "disease": "OLP",
        "document": """
Oral lichen planus is a chronic inflammatory condition.
Management focuses on reducing irritation, maintaining oral hygiene,
and monitoring for long-term changes. Regular follow-up is recommended.
"""
    },
    {
        "disease": "OT",
        "document": """
Oral thrush is a fungal infection caused by Candida.
General care includes antifungal therapy prescribed by a clinician,
good oral hygiene, and addressing risk factors such as inhaler use or diabetes.
Seek medical guidance if symptoms persist.
"""
    }
]

# ===============================
# RAG INITIALIZATION
# ===============================
def initialize_rag_system():
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="oral_disease_knowledge")

    for i, item in enumerate(medical_knowledge_base):
        collection.add(
            documents=[item["document"]],
            metadatas=[{"disease": item["disease"]}],
            ids=[f"doc_{i}"]
        )
    return embedder, collection

# ===============================
# LOAD CLASSIFIER
# ===============================
def load_classification_model():
    model = pretrainedmodels.__dict__['inceptionresnetv2'](pretrained=None)
    model.last_linear = nn.Linear(model.last_linear.in_features, 7)
    model.load_state_dict(torch.load("inceptionresnetv2_teeth.pth", map_location="cpu"))
    model.eval()
    return model

embedder, knowledge_collection = initialize_rag_system()
classification_model = load_classification_model()

# ===============================
# IMAGE TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# RETRIEVAL
# ===============================
def retrieve_knowledge(disease_code):
    results = knowledge_collection.query(
        query_texts=["educational summary"],
        where={"disease": disease_code},
        n_results=1
    )
    return results["documents"][0] if results["documents"] else []

# ===============================
# GENERATION (USING OLD GEMINI SDK)
# ===============================
def generate_with_gemini(disease_name, confidence, docs):
    if not client:
        return "⚠️ Gemini API not configured."

    context = "\n".join(docs)

    prompt = f"""
You are a medical education assistant.

Condition Identified: {disease_name}
Model Confidence: {confidence:.2f}%

Retrieved Medical Summary:
{context}

TASK:
Provide a clear, concise educational overview that includes:
- What the condition generally is
- General management approach
- When to seek medical care

RULES:
- NO medication dosages
- NO prescribing or instructing
- NO specific treatment regimens
- Educational guidance ONLY
- End with a disclaimer
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=800
            )
        )
        text = response.text

        return text + """

---
⚠️ **Disclaimer**
This information is for educational purposes only.
It is NOT a diagnosis or treatment plan.
Always consult a licensed healthcare professional.
"""

    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"


# ===============================
# FULL PIPELINE
# ===============================
def predict_with_full_rag(image):
    if image is None:
        return "⚠️ Upload an image."

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = classification_model(img_tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, 1)

    disease_code = class_names[idx.item()]
    disease_name = disease_names[disease_code]

    docs = retrieve_knowledge(disease_code)
    return generate_with_gemini(disease_name, conf.item() * 100, docs)

# ===============================
# GRADIO UI
# ===============================
with gr.Blocks(title="Oral Disease TRUE RAG") as demo:
    gr.Markdown("# 🦷 Oral Disease Classifier (Educational RAG System)")
    img = gr.Image(type="pil")
    out = gr.Markdown()
    btn = gr.Button("Analyze")

    btn.click(predict_with_full_rag, img, out)

demo.launch()
