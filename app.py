import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms as T
from timm import create_model
from transformers import pipeline
from ultralytics import YOLO
import os

# ====================== SETUP ======================
st.set_page_config(
    page_title="PneumoAI Medical Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== MODEL LOADING ======================
@st.cache_resource
def load_models():
    """Load all medical models with error handling"""
    models = {}
    
    # Classification Model
    try:
        cls_model_path = 'pneumonia_detection_model.pth'
        if not os.path.exists(cls_model_path):
            st.error("Classification model not found!")
        else:
            model = create_model('deit_base_patch16_224', pretrained=False, num_classes=4)
            model.load_state_dict(torch.load(cls_model_path, map_location='cpu'))
            model.eval()
            models['classification'] = model
    except Exception as e:
        st.error(f"Classification model error: {str(e)}")
    
    # Detection Model
    try:
        det_model_path = "best (1).torchscript"
        if not os.path.exists(det_model_path):
            st.error("Detection model not found!")
        else:
            models['detection'] = YOLO(det_model_path, task='detect')
    except Exception as e:
        st.error(f"Detection model error: {str(e)}")
    
    return models

# ====================== MEDICAL FUNCTIONS ======================
def classify_pneumonia(image, model):
    """Classify pneumonia type"""
    try:
        # Preprocess
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)
        
        # Classify
        with torch.no_grad():
            outputs = model(img_tensor)
        
        diagnosis = ["COVID-19", "Viral Pneumonia", "Lung Opacity", "Normal"][torch.argmax(outputs)]
        severity = torch.softmax(outputs, dim=1)[0][torch.argmax(outputs)].item() * 10
        
        return diagnosis, severity
    
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return None, None

def detect_pneumonia(image, model):
    """Detect pneumonia regions with bounding boxes"""
    try:
        img = np.array(image)
        
        # Handle different image formats
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Process
        img_resized = cv2.resize(img, (1024, 1024))
        results = model.predict(img_resized, conf=0.25, imgsz=1024)
        
        # Draw boxes
        boxes_drawn = 0
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 3)
                boxes_drawn += 1
        
        return cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), boxes_drawn
    
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return None, None

# ====================== MEDICAL CHATBOT ======================
class PneumoniaChatbot:
    def __init__(self):
        self.knowledge_base = {
            "symptoms": {
                "question": "What are the symptoms of pneumonia?",
                "answer": """
                Common symptoms include:
                - Cough (often with green/yellow/rust-colored phlegm)
                - Fever, sweating and chills
                - Shortness of breath
                - Rapid, shallow breathing
                - Sharp chest pain that worsens with deep breathing/coughing
                - Loss of appetite, low energy, fatigue
                - Nausea/vomiting (especially in children)
                - Confusion (especially in elderly patients)
                """
            },
            "treatment": {
                "question": "How is pneumonia treated?",
                "answer": """
                Treatment depends on type and severity:
                
                **Bacterial Pneumonia:**
                - Antibiotics (amoxicillin, azithromycin, etc.)
                - Rest and fluids
                - Fever reducers (acetaminophen/ibuprofen)
                
                **Viral Pneumonia:**
                - Antiviral medications (if severe)
                - Symptom management
                - Same supportive care as bacterial
                
                **Severe Cases:**
                - Hospitalization for oxygen therapy
                - IV antibiotics/fluids
                - Mechanical ventilation if needed
                """
            },
            "prevention": {
                "question": "How can pneumonia be prevented?",
                "answer": """
                Prevention strategies:
                1. Vaccination:
                   - Pneumococcal vaccines (PCV13, PPSV23)
                   - Annual flu vaccine
                   - COVID-19 vaccines
                2. Good hygiene:
                   - Frequent hand washing
                   - Disinfecting surfaces
                   - Covering coughs/sneezes
                3. Healthy lifestyle:
                   - Not smoking
                   - Managing chronic conditions
                   - Proper nutrition and exercise
                """
            }
        }
        
    def generate_response(self, question):
        """Generate medical response with references"""
        lower_q = question.lower()
        
        # Check knowledge base first
        for key in self.knowledge_base:
            if key in lower_q:
                return self.knowledge_base[key]["answer"]
        
        # Use AI for other questions
        generator = pipeline('text-generation', model='microsoft/biogpt')
        response = generator(
            f"Answer this pneumonia-related medical question accurately and concisely: {question}",
            max_length=200,
            do_sample=True,
            temperature=0.7
        )[0]['generated_text']
        
        return response.split(".")[0] + "."

# ====================== STREAMLIT UI ======================
def main():
    st.title("🏥 PneumoAI Medical Diagnosis Suite")
    
    # Load all models at startup
    models = load_models()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio(
        "Select Mode:",
        ["Classification", "Detection", "Medical Chatbot"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About This App:**
    - Classification: Diagnoses pneumonia type
    - Detection: Identifies infected lung regions
    - Chatbot: Answers medical questions
    """)
    
    # Mode selection
    if mode == "Classification":
        st.header("Pneumonia Classification")
        st.markdown("Upload a chest X-ray for diagnosis")
        
        upload = st.file_uploader("Choose X-ray image", type=["jpg", "png", "jpeg"])
        
        if upload and models.get('classification'):
            image = Image.open(upload).convert('RGB')
            st.image(image, caption="Original X-ray", use_column_width=True)
            
            with st.spinner("Analyzing X-ray..."):
                diagnosis, severity = classify_pneumonia(image, models['classification'])
            
            if diagnosis:
                st.success("### Diagnosis Results")
                st.markdown(f"""
                - **Condition:** {diagnosis}
                - **Severity Score:** {severity:.1f}/10
                """)
                
                # Medical interpretation
                if diagnosis == "Normal":
                    st.balloons()
                    st.success("No signs of pneumonia detected!")
                else:
                    st.warning(f"Clinical findings suggest {diagnosis}")
                    st.markdown(f"""
                    **Recommended Actions:**
                    - Consult a pulmonologist
                    - {severity:.1f}/10 severity indicates {'mild' if severity < 4 else 'moderate' if severity < 7 else 'severe'} case
                    - Immediate medical attention recommended for scores >7
                    """)
    
    elif mode == "Detection":
        st.header("Pneumonia Region Detection")
        st.markdown("Upload X-ray to identify infected lung areas")
        
        upload = st.file_uploader("Choose X-ray image", type=["jpg", "png", "jpeg"])
        
        if upload and models.get('detection'):
            image = Image.open(upload)
            
            with st.spinner("Detecting pneumonia regions..."):
                result_img, count = detect_pneumonia(image, models['detection'])
            
            if result_img is not None:
                st.image(result_img, caption=f"Detected {count} pneumonia regions", use_column_width=True)
                
                st.markdown("### Clinical Interpretation")
                if count == 0:
                    st.success("No pneumonia regions detected")
                else:
                    st.warning(f"Found {count} infected region{'s' if count > 1 else ''}")
                    st.markdown("""
                    **Next Steps:**
                    - Quantitative analysis shows affected areas
                    - Urgent care recommended for >3 regions
                    - Compare with follow-up scans to monitor progression
                    """)
    
    elif mode == "Medical Chatbot":
        st.header("Pneumonia Medical Assistant")
        st.markdown("Ask questions about pneumonia diagnosis, treatment, and prevention")
        
        # Initialize chatbot
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = PneumoniaChatbot()
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Common questions
        st.subheader("Quick Questions:")
        cols = st.columns(3)
        for i, (key, item) in enumerate(st.session_state.chatbot.knowledge_base.items()):
            if cols[i%3].button(item["question"]):
                st.session_state.current_question = item["question"]
        
        # Chat interface
        st.subheader("Ask Your Question:")
        user_input = st.text_input(
            "Type your medical question:",
            value=st.session_state.get("current_question", ""),
            key="user_input"
        )
        
        if st.button("Submit") and user_input:
            with st.spinner("Consulting medical resources..."):
                response = st.session_state.chatbot.generate_response(user_input)
            
            # Update chat history
            st.session_state.chat_history.append(("User", user_input))
            st.session_state.chat_history.append(("Dr. PneumoAI", response))
            st.session_state.current_question = ""
            
            # Display conversation
            st.subheader("Conversation History:")
            for speaker, text in st.session_state.chat_history[-6:]:
                if speaker == "User":
                    st.markdown(f"**You:** {text}")
                else:
                    st.markdown(f"**:blue[{speaker}]:** {text}")
                    st.write("")

if __name__ == "__main__":
    main()