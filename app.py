import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import pretrainedmodels
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

# ===============================
# GEMINI CONFIGURATION (FIXED)
# ===============================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    print("✅ Gemini API configured successfully")
else:
    gemini_model = None
    print("⚠️ GEMINI_API_KEY not found")

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
# SAFE MEDICAL KNOWLEDGE BASE
# ===============================
medical_knowledge_base = [
    {
        "disease": "CaS",
        "document": """
Canker sores are non-contagious ulcers inside the mouth. 
Management focuses on symptom relief, reducing inflammation, and avoiding triggers.
Common approaches include topical oral gels, soothing mouth rinses, maintaining oral hygiene,
nutritional balance, and avoiding spicy or acidic foods.
Medical evaluation is advised if ulcers persist, recur frequently, or are unusually large.
"""
    },
    {
        "disease": "CoS",
        "document": """
Cold sores are caused by herpes simplex virus and typically occur on or around the lips.
Management includes early antiviral treatment, keeping the area clean, avoiding touching lesions,
and minimizing known triggers such as stress or sun exposure.
Medical care is recommended for severe, frequent, or non-healing outbreaks.
"""
    },
    {
        "disease": "Gum",
        "document": """
Gingivostomatitis is an inflammatory condition affecting gums and oral tissues.
Care focuses on pain control, maintaining hydration, gentle oral hygiene,
and treating underlying viral or bacterial causes when confirmed by a clinician.
Urgent care is needed if swallowing becomes difficult or dehydration occurs.
"""
    },
    {
        "disease": "MC",
        "document": """
Suspected mouth cancer requires urgent professional evaluation.
Diagnosis typically involves clinical examination and biopsy.
Management is handled by specialists and may include surgery, radiation, or systemic therapy.
Early referral significantly improves outcomes.
"""
    },
    {
        "disease": "OC",
        "document": """
Oral cancer is a serious condition that must be diagnosed and managed by specialists.
Early detection and referral are critical.
Treatment plans depend on staging and individual patient factors.
This tool does not provide treatment plans for cancer.
"""
    },
    {
        "disease": "OLP",
        "document": """
Oral lichen planus is a chronic inflammatory condition.
Management focuses on symptom control, reducing inflammation, monitoring changes,
and avoiding irritants.
Regular follow-up is important due to a small risk of malignant transformation.
"""
    },
    {
        "disease": "OT",
        "document": """
Oral thrush is a fungal infection caused by Candida species.
Management generally includes antifungal therapy, good oral hygiene,
addressing risk factors such as diabetes or inhaled steroids,
and denture care when applicable.
Medical consultation is recommended if symptoms persist or recur.
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
# MODEL LOADING
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
        query_texts=["management and overview"],
        where={"disease": disease_code},
        n_results=1
    )
    return results["documents"][0] if results["documents"] else []

# ===============================
# GENERATION (SAFE RAG)
# ===============================
def generate_with_gemini(disease_name, confidence, docs):
    if not gemini_model:
        return "⚠️ Gemini API not configured."

    context = "\n".join(docs)

    prompt = f"""
You are a medical education assistant.

Condition: {disease_name}
Model Confidence Score: {confidence:.2f}%

Retrieved Knowledge:
{context}

TASK:
Provide a concise, structured educational summary including:
- Brief overview
- General management approach
- When to seek professional care

RULES:
- No medication dosages
- No treatment prescriptions
- Educational tone only
- End with a medical disclaimer
"""

    response = gemini_model.generate_content(prompt)
    return response.text + """

---
⚠️ **Medical Disclaimer**
This information is for educational purposes only.
It does NOT provide diagnosis or treatment.
Always consult a qualified healthcare professional.
"""

# ===============================
# FULL PIPELINE
# ===============================
def predict_with_full_rag(image):
    if image is None:
        return "⚠️ Please upload an image."

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = classification_model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, idx = torch.max(probs, 1)

    disease_code = class_names[idx.item()]
    disease_name = disease_names[disease_code]
    confidence_score = confidence.item() * 100

    docs = retrieve_knowledge(disease_code)
    return generate_with_gemini(disease_name, confidence_score, docs)

# ===============================
# GRADIO UI
# ===============================
with gr.Blocks(title="Oral Disease TRUE RAG Classifier") as demo:
    gr.Markdown("# 🦷 Oral Disease Classification with TRUE RAG")
    image_input = gr.Image(type="pil")
    output = gr.Markdown()
    analyze = gr.Button("Analyze with RAG")

    analyze.click(predict_with_full_rag, image_input, output)

demo.launch()
