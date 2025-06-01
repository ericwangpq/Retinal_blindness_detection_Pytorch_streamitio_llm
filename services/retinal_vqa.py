# Enhanced LLM integration for retinopathy explanation and VQA with RAG support
import requests
import json
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
from .rag_service import RAGService

# Initialize RAG service globally
rag_service = None

def initialize_rag_service():
    """Initialize RAG service for enhanced medical analysis"""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
        # Try to initialize knowledge base
        try:
            rag_service.initialize_knowledge_base()
        except Exception as e:
            print(f"Warning: Could not initialize RAG knowledge base: {e}")
    return rag_service

# this is the old way to load the api key - load from .env file / os env variable
# import os
# from dotenv import load_dotenv

# Load environment variables
# load_dotenv()


# Get API key from configuration file
def get_api_key():
    try:
        with open('config/openai_api_key', 'r') as file:
            api_key = file.read().strip()
        if not api_key:
            raise ValueError("API key not found in config file")
        return api_key
    except Exception as e:
        print(f"Error loading API key: {str(e)}")
        return None

def encode_image_to_base64(image):
    """Convert PIL Image or matplotlib figure to base64 string"""
    if isinstance(image, plt.Figure):
        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        image.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        image_pil = Image.open(buf)
    else:
        image_pil = image
    
    # Convert to base64
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def analyze_retinal_image_and_heatmap(original_image, heatmap_figure, prediction_results, patient_age=None, diabetes_duration=None):
    """Analyze retinal image with heatmap using GPT-4o-mini vision capabilities enhanced with RAG"""
    api_url = "https://api.openai.com/v1/chat/completions"
    api_key = get_api_key()
    
    if not api_key:
        raise ValueError("API key not found")
    
    # Initialize RAG service
    rag = initialize_rag_service()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Encode images to base64
    original_b64 = encode_image_to_base64(original_image)
    heatmap_b64 = encode_image_to_base64(heatmap_figure)
    
    # Construct detailed prompt
    patient_info = ""
    if patient_age and diabetes_duration:
        patient_info = f"\nPatient Information:\n- Age: {patient_age} years\n- Duration of diabetes: {diabetes_duration} years\n"
    
    # Get relevant medical literature context using RAG
    medical_context = ""
    if rag:
        try:
            search_query = f"diabetic retinopathy grade {prediction_results['value']} {prediction_results['class']} analysis heatmap fundus examination"
            medical_context = rag.get_relevant_medical_context(search_query, max_context_length=2000)
        except Exception as e:
            print(f"Warning: Could not retrieve medical context: {e}")
    
    # Enhanced prompt with medical literature context
    base_prompt = f"""You are an expert ophthalmologist AI assistant analyzing retinal images for diabetic retinopathy.

{patient_info}
AI Model Results:
- Predicted Class: {prediction_results['class']}
- Severity Grade: {prediction_results['value']}
- Confidence: {prediction_results['probability']:.2%}

I'm showing you two images:
1. The original retinal fundus photograph
2. A GradCAM heatmap visualization showing which areas the AI model focused on for its prediction"""

    if medical_context:
        enhanced_prompt = f"""{base_prompt}

RELEVANT MEDICAL LITERATURE:
{medical_context}

Please provide a comprehensive analysis that incorporates insights from the current medical literature above, including:

1. **Clinical Assessment**: Explain what the AI prediction means in medical terms, referencing relevant literature
2. **Heatmap Analysis**: Describe what the highlighted areas represent and their clinical significance based on current research
3. **Key Findings**: Identify specific retinal features visible in the image that support the diagnosis, citing literature when relevant
4. **Evidence-Based Patient Explanation**: Provide a clear, patient-friendly explanation supported by research findings
5. **Current Guidelines Recommendations**: Suggest appropriate next steps based on the latest clinical guidelines
6. **Monitoring Protocol**: Advise on follow-up frequency and warning signs based on evidence-based practices

**IMPORTANT FORMATTING INSTRUCTIONS:**
- When referencing literature or research findings, use the format: ***According to the literature, [finding]*** or ***Research indicates that [finding]*** or ***Studies show that [finding]***
- Make all literature citations bold and italic using ***text*** format
- This will help patients easily identify evidence-based information
- Example: ***According to recent studies, GradCAM highlighted areas typically indicate microaneurysms which are early signs of diabetic retinopathy***

When relevant, cite the medical literature to support your analysis and recommendations."""
    else:
        enhanced_prompt = f"""{base_prompt}

Please provide a comprehensive analysis including:

1. **Clinical Assessment**: Explain what the AI prediction means in medical terms
2. **Heatmap Analysis**: Describe what the highlighted areas in the heatmap represent and their clinical significance
3. **Key Findings**: Identify specific retinal features visible in the image that support the diagnosis
4. **Patient Explanation**: Provide a clear, patient-friendly explanation of the findings
5. **Recommendations**: Suggest appropriate next steps based on the severity level
6. **Monitoring**: Advise on follow-up frequency and warning signs to watch for

**FORMATTING NOTE**: Use clear section headers and bullet points to make the analysis easy to read.

Be thorough but accessible, and highlight any areas of concern that require immediate attention."""

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert ophthalmologist specializing in diabetic retinopathy analysis. You have access to current medical literature and provide detailed, accurate medical insights while being accessible to patients. Always use ***bold italic*** formatting when citing literature or research findings to make evidence-based information clearly visible."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": enhanced_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{original_b64}"}
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{heatmap_b64}"}
                    }
                ]
            }
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error analyzing images: {response.status_code}, {response.text}"

def answer_retinal_question(question, context_analysis, prediction_results, patient_age=None, diabetes_duration=None):
    """Answer specific questions about the retinal analysis using RAG-enhanced responses"""
    
    # Initialize RAG service
    rag = initialize_rag_service()
    
    # Prepare patient context
    patient_info = ""
    if patient_age and diabetes_duration:
        patient_info = f"Patient: {patient_age} years old, diabetes for {diabetes_duration} years. "
    
    patient_context = f"{patient_info}Current AI Results: {prediction_results['class']} (Grade {prediction_results['value']}, {prediction_results['probability']:.2%} confidence)"
    
    # Use RAG-enhanced question answering if available
    if rag:
        try:
            return rag.enhanced_question_answering(
                question=question,
                previous_analysis=context_analysis,
                patient_context=patient_context,
                prediction_results=prediction_results
            )
        except Exception as e:
            print(f"Warning: RAG-enhanced answering failed, falling back to basic method: {e}")
    
    # Fallback to original method
    api_url = "https://api.openai.com/v1/chat/completions"
    api_key = get_api_key()
    
    if not api_key:
        raise ValueError("API key not found")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are an expert ophthalmologist answering questions about a diabetic retinopathy analysis.

Context from previous analysis:
{context_analysis}

{patient_context}

Patient Question: {question}

Please provide a clear, accurate answer based on the analysis context and your medical expertise. If the question is outside the scope of the retinal analysis, politely redirect to relevant information.

**FORMATTING INSTRUCTION**: When referencing medical knowledge or established practices, use ***bold italic*** format to highlight key medical insights. Example: ***According to ophthalmology guidelines, regular screening is essential for diabetic patients***"""

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert ophthalmologist providing patient education about diabetic retinopathy. Be accurate, empathetic, and helpful. Use ***bold italic*** formatting to highlight key medical insights and established practices."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 800
    }
    
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error answering question: {response.status_code}, {response.text}"

def get_rag_service_info():
    """Get information about RAG service status"""
    rag = initialize_rag_service()
    if rag:
        try:
            return rag.get_knowledge_base_stats()
        except Exception as e:
            return {"error": str(e)}
    return {"status": "RAG service not available"}

def search_medical_literature(query: str, k: int = 5):
    """Search medical literature directly"""
    rag = initialize_rag_service()
    if rag:
        try:
            return rag.search_medical_literature(query, k)
        except Exception as e:
            return [{"error": str(e)}]
    return [{"error": "RAG service not available"}]

def get_llm_explanation(dr_grade, patient_age, diabetes_duration):
    """Legacy function for basic explanations - kept for backward compatibility"""
    api_url = "https://api.openai.com/v1/chat/completions"
    api_key = get_api_key()
    
    if not api_key:
        raise ValueError("API key not found")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    You are a medical AI assistant helping to explain diabetic retinopathy findings.
    
    Patient information:
    - Age: {patient_age} years
    - Duration of diabetes: {diabetes_duration} years
    
    AI Detection Result:
    - Diabetic Retinopathy Grade: {dr_grade}
    
    Please provide:
    1. A simple explanation of what this grade means in patient-friendly language
    2. What this finding suggests about their diabetes management
    3. Recommended next steps
    4. When they should seek immediate medical attention
    """
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a medical assistant specializing in ophthalmology."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
patient_explanation = get_llm_explanation(
    dr_grade=2, 
    patient_age=62, 
    diabetes_duration=15
)