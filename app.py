import streamlit as st
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

# Import services
from services.model_service import ModelService
from services.visualization_service import VisualizationService
from services.retinal_vqa import analyze_retinal_image_and_heatmap, answer_retinal_question

# Initialize services
model_service = ModelService()
visualization_service = VisualizationService()

# Initialize session state for VQA
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = ""
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {}
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'heatmap_figure' not in st.session_state:
    st.session_state.heatmap_figure = None
if 'patient_age' not in st.session_state:
    st.session_state.patient_age = None
if 'diabetes_duration' not in st.session_state:
    st.session_state.diabetes_duration = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

st.title("🔬 AI-Powered Diabetic Retinopathy Screening & Analysis")
st.markdown("### Advanced screening tool with AI-powered visual question answering")

with st.sidebar:
    st.header("📋 Patient Information")
    st.markdown("*Optional: Provide patient details for personalized analysis*")
    
    patient_age = st.number_input(
        "Patient Age", 
        min_value=18, 
        max_value=100, 
        value=st.session_state.patient_age if st.session_state.patient_age else 50,
        help="Patient's current age"
    )
    
    diabetes_duration = st.number_input(
        "Diabetes Duration (years)", 
        min_value=0, 
        max_value=50, 
        value=st.session_state.diabetes_duration if st.session_state.diabetes_duration else 5,
        help="How long the patient has had diabetes"
    )
    
    # Store in session state
    st.session_state.patient_age = patient_age
    st.session_state.diabetes_duration = diabetes_duration
    
    st.divider()
    st.header("ℹ️ About")
    st.markdown("""
    **Features:**
    - AI-powered retinal image analysis
    - GradCAM visualization
    - Expert AI explanations
    - Interactive Q&A system
    - Personalized recommendations
    """)

col1, col2 = st.columns(2)
with col1:
    st.subheader('🎯 Benefits')
    st.markdown('- Early detection of retinopathy')
    st.markdown('- Accessible screening in primary care')
    st.markdown('- AI-powered explanations')
    st.markdown('- Interactive patient education')

with col2:
    st.subheader('⚙️ How It Works')
    st.markdown('1. Upload a retinal image')
    st.markdown('2. AI analyzes image & generates heatmap')
    st.markdown('3. Get comprehensive AI explanation')
    st.markdown('4. Ask questions about your results')

st.header('📤 Upload Retinal Image')
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Retinal Image', width=300)
    st.write("✅ Image successfully uploaded!")
    
    # Store original image in session state
    st.session_state.original_image = Image.open(uploaded_file).convert('RGB')

if st.button("🔍 Analyze Image", type="primary"):
    if uploaded_file is None:
        st.error("❌ Please upload an image first!")
    else:
        try:
            with st.spinner("🔄 Processing image and generating analysis..."):
                # Load model and make prediction
                model = model_service.load_model("classifier.pt")
                results = model_service.predict_image(uploaded_file)
                st.session_state.prediction_results = results
                
                # Generate heatmap
                fig = visualization_service.generate_gradcam_visualization(model, uploaded_file)
                st.session_state.heatmap_figure = fig
                
                # Get AI analysis of images
                with st.spinner("🤖 AI is analyzing the images and heatmap..."):
                    ai_analysis = analyze_retinal_image_and_heatmap(
                        st.session_state.original_image,
                        fig,
                        results,
                        patient_age,
                        diabetes_duration
                    )
                    st.session_state.ai_analysis = ai_analysis
                    st.session_state.analysis_complete = True
                
                st.success("✅ Analysis complete!")
                
        except FileNotFoundError:
            st.error("❌ Model file 'classifier.pt' not found. Please ensure the model file is in the same directory.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")

# Display results if analysis is complete
if st.session_state.analysis_complete:
    st.header("📊 Analysis Results")
    
    tab1, tab2, tab3 = st.tabs(["🎯 AI Analysis", "📈 Technical Results", "🔥 Heatmap Visualization"])
    
    with tab1:
        st.subheader("🤖 Expert AI Analysis")
        st.markdown(st.session_state.ai_analysis)
    
    with tab2:
        st.subheader("📋 Technical Results")
        results = st.session_state.prediction_results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Classification", results['class'])
        with col2:
            st.metric("Severity Grade", results['value'])
        with col3:
            st.metric("Confidence", f"{results['probability']:.2%}")
        
        # Severity explanation
        severity_colors = {
            0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "🟣"
        }
        severity_descriptions = {
            0: "No diabetic retinopathy detected",
            1: "Mild non-proliferative diabetic retinopathy",
            2: "Moderate non-proliferative diabetic retinopathy", 
            3: "Severe non-proliferative diabetic retinopathy",
            4: "Proliferative diabetic retinopathy"
        }
        
        grade = results['value']
        st.info(f"{severity_colors[grade]} **Grade {grade}**: {severity_descriptions[grade]}")
    
    with tab3:
        st.subheader("🔥 GradCAM Heatmap Analysis")
        st.markdown("*This heatmap shows which areas of the retina the AI model focused on when making its prediction.*")
        if st.session_state.heatmap_figure:
            st.pyplot(st.session_state.heatmap_figure)
            plt.close(st.session_state.heatmap_figure)
    
    # Interactive Q&A Section
    st.header("💬 Ask Questions About Your Results")
    st.markdown("*Ask our AI expert any questions about your retinal analysis, diagnosis, or recommendations.*")
    
    # Display previous Q&A history
    if st.session_state.qa_history:
        st.subheader("📝 Previous Questions & Answers")
        for i, (question, answer) in enumerate(st.session_state.qa_history):
            with st.expander(f"Q{i+1}: {question[:50]}..."):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
    
    # New question input
    user_question = st.text_input(
        "Ask a question about your results:",
        placeholder="e.g., What does this mean for my vision? Should I be worried? What should I do next?",
        key="user_question"
    )
    
    if st.button("🤔 Ask AI Expert", type="secondary"):
        if user_question.strip():
            with st.spinner("🤖 AI expert is thinking..."):
                try:
                    answer = answer_retinal_question(
                        user_question,
                        st.session_state.ai_analysis,
                        st.session_state.prediction_results,
                        patient_age,
                        diabetes_duration
                    )
                    
                    # Add to history
                    st.session_state.qa_history.append((user_question, answer))
                    
                    # Display the answer
                    st.subheader("🎯 AI Expert Response")
                    st.markdown(answer)
                    
                    # Clear the input
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error getting AI response: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question first!")

if st.button("🔄 Reset Analysis"):
    # Clear all session state
    for key in ['analysis_complete', 'ai_analysis', 'prediction_results', 
                'original_image', 'heatmap_figure', 'qa_history']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Custom CSS for better styling
chat_plh_style = """
        <style>
            div[data-testid='stVerticalBlock']:has(div#chat_inner):not(:has(div#chat_outer)) {
                background-color: #2d425d;
                border-radius: 10px;
                padding: 10px;
            }
            .stMetric {
                background-color: #f0f2f6;
                padding: 10px;
                border-radius: 5px;
                border-left: 4px solid #1f77b4;
            }
        </style>
        """

st.markdown(chat_plh_style, unsafe_allow_html=True)
