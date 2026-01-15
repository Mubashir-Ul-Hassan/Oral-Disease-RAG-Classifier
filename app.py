import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import pretrainedmodels

# Define class names
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

disease_info = {
    'CaS': {
        'name': 'Canker Sores (Aphthous Ulcers)',
        'description': 'Small, painful sores inside the mouth with white or yellow center.',
        'medicines': ['Benzocaine gel', 'Hydrogen peroxide rinse', 'Corticosteroid ointment'],
        'advice': 'Usually heal within 1-2 weeks. Avoid spicy/acidic foods.'
    },
    'CoS': {
        'name': 'Cold Sores (Herpes Labialis)',
        'description': 'Viral infection causing fluid-filled blisters around lips.',
        'medicines': ['Acyclovir cream', 'Valacyclovir', 'Docosanol'],
        'advice': 'Antiviral treatment most effective when started early.'
    },
    'Gum': {
        'name': 'Gingivostomatitis',
        'description': 'Inflammation of gums and mouth lining.',
        'medicines': ['Chlorhexidine mouthwash', 'Ibuprofen', 'Antiviral medication'],
        'advice': 'Maintain good oral hygiene. See dentist for severe cases.'
    },
    'MC': {
        'name': 'Mouth Cancer',
        'description': 'Malignant growth in the mouth.',
        'medicines': ['Requires oncologist consultation'],
        'advice': '⚠️ URGENT: Immediate referral to oncologist required.'
    },
    'OC': {
        'name': 'Oral Cancer',
        'description': 'Cancerous lesions in oral cavity.',
        'medicines': ['Requires oncologist consultation'],
        'advice': '⚠️ URGENT: Immediate referral to oncologist required.'
    },
    'OLP': {
        'name': 'Oral Lichen Planus',
        'description': 'Chronic inflammatory condition affecting mouth lining.',
        'medicines': ['Topical corticosteroids', 'Tacrolimus ointment'],
        'advice': 'Chronic condition requiring regular monitoring.'
    },
    'OT': {
        'name': 'Oral Thrush',
        'description': 'Fungal infection causing white patches in mouth.',
        'medicines': ['Nystatin suspension', 'Fluconazole', 'Clotrimazole lozenges'],
        'advice': 'Antifungal treatment for 7-14 days.'
    }
}

# Load model (removed @st.cache_resource)
def load_model():
    model = pretrainedmodels.__dict__['inceptionresnetv2'](pretrained=None)
    model.last_linear = nn.Linear(model.last_linear.in_features, 7)
    
    # Load trained weights - change filename if needed
    model.load_state_dict(torch.load('inceptionresnetv2_teeth.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image):
    if image is None:
        return "Please upload an image first."
    
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    predicted_class = class_names[prediction.item()]
    confidence_score = confidence.item() * 100
    
    # Get disease information
    info = disease_info[predicted_class]
    
    # Format output
    result = f"""
## 🔍 Diagnosis: {info['name']}

**Confidence:** {confidence_score:.2f}%

**Description:** {info['description']}

### 💊 Recommended Medicines:
"""
    
    for med in info['medicines']:
        result += f"\n- {med}"
    
    result += f"""

### 📋 Medical Advice:
{info['advice']}

---

### ⚠️ **IMPORTANT DISCLAIMER**
This is an AI-powered tool for informational purposes only. It is NOT a substitute for professional medical diagnosis and treatment. Always consult a qualified healthcare professional for proper medical advice, diagnosis, and treatment.
"""
    
    return result

# Create Gradio interface
with gr.Blocks(title="Oral Disease RAG Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    #  Oral Disease RAG Classifier
    
    Upload an image of an oral condition to get AI-powered analysis and medicine recommendations using RAG technology.
    
    **Supported Conditions:**
    - Canker Sores (CaS)
    - Cold Sores (CoS)
    - Gingivostomatitis (Gum)
    - Mouth Cancer (MC)
    - Oral Cancer (OC)
    - Oral Lichen Planus (OLP)
    - Oral Thrush (OT)
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Oral Image")
            submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
        
        with gr.Column():
            output = gr.Markdown(label="Results")
    
    submit_btn.click(fn=predict, inputs=image_input, outputs=output)
    
    gr.Markdown("""
    ---
    **Model Information:**
    - Architecture: InceptionResNetV2
    - Training Accuracy: ~99.5%
    - Technology: RAG (Retrieval-Augmented Generation)
    
    **Disclaimer:** This tool is for educational and informational purposes only.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()