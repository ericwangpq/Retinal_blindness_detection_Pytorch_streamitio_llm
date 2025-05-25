# Enhanced LLM integration for retinopathy explanation and VQA
import requests
import json
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt

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
    """Analyze retinal image with heatmap using GPT-4o-mini vision capabilities"""
    api_url = "https://api.openai.com/v1/chat/completions"
    api_key = get_api_key()
    
    if not api_key:
        raise ValueError("API key not found")
    
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
    
    prompt = f"""You are an expert ophthalmologist AI assistant analyzing retinal images for diabetic retinopathy.

{patient_info}
AI Model Results:
- Predicted Class: {prediction_results['class']}
- Severity Grade: {prediction_results['value']}
- Confidence: {prediction_results['probability']:.2%}

I'm showing you two images:
1. The original retinal fundus photograph
2. A GradCAM heatmap visualization showing which areas the AI model focused on for its prediction

Please provide a comprehensive analysis including:

1. **Clinical Assessment**: Explain what the AI prediction means in medical terms
2. **Heatmap Analysis**: Describe what the highlighted areas in the heatmap represent and their clinical significance
3. **Key Findings**: Identify specific retinal features visible in the image that support the diagnosis
4. **Patient Explanation**: Provide a clear, patient-friendly explanation of the findings
5. **Recommendations**: Suggest appropriate next steps based on the severity level
6. **Monitoring**: Advise on follow-up frequency and warning signs to watch for

Be thorough but accessible, and highlight any areas of concern that require immediate attention."""

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert ophthalmologist specializing in diabetic retinopathy analysis. Provide detailed, accurate medical insights while being accessible to patients."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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
        "max_tokens": 1500
    }
    
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error analyzing images: {response.status_code}, {response.text}"

def answer_retinal_question(question, context_analysis, prediction_results, patient_age=None, diabetes_duration=None):
    """Answer specific questions about the retinal analysis"""
    api_url = "https://api.openai.com/v1/chat/completions"
    api_key = get_api_key()
    
    if not api_key:
        raise ValueError("API key not found")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    patient_info = ""
    if patient_age and diabetes_duration:
        patient_info = f"Patient: {patient_age} years old, diabetes for {diabetes_duration} years. "
    
    prompt = f"""You are an expert ophthalmologist answering questions about a diabetic retinopathy analysis.

Context from previous analysis:
{context_analysis}

{patient_info}Current AI Results: {prediction_results['class']} (Grade {prediction_results['value']}, {prediction_results['probability']:.2%} confidence)

Patient Question: {question}

Please provide a clear, accurate answer based on the analysis context and your medical expertise. If the question is outside the scope of the retinal analysis, politely redirect to relevant information."""

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert ophthalmologist providing patient education about diabetic retinopathy. Be accurate, empathetic, and helpful."
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