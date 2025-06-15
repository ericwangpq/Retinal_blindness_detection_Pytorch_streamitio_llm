import streamlit as st
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

# Import services
from services.model_service import ModelService
from services.visualization_service import VisualizationService
from services.retinal_vqa import analyze_retinal_image_and_heatmap, answer_retinal_question, get_rag_service_info, search_medical_literature

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
st.markdown("### Advanced screening tool with AI-powered visual question answering and medical literature support")

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
    
    # RAG Knowledge Base Status
    st.header("📚 Medical Knowledge Base")
    rag_info = get_rag_service_info()
    
    if 'error' in rag_info:
        st.error(f"⚠️ Knowledge base not available: {rag_info['error']}")
    else:
        total_vectors = rag_info.get('total_vectors', 0)
        if total_vectors > 0:
            st.success(f"✅ {total_vectors} medical documents indexed")
            sources = rag_info.get('sources', [])
            if sources:
                st.caption(f"Sources: {len(sources)} medical papers")
                with st.expander("View Sources"):
                    for source in sources[:10]:  # Show first 10 sources
                        st.write(f"• {source}")
                    if len(sources) > 10:
                        st.write(f"... and {len(sources) - 10} more")
        else:
            st.warning("⚠️ Knowledge base empty")
    
    st.divider()
    st.header("ℹ️ About")
    st.markdown("""
    **Enhanced Features:**
    - AI-powered retinal image analysis
    - GradCAM visualization
    - Literature-supported AI explanations
    - Interactive Q&A with medical evidence
    - Personalized recommendations
    - Access to medical research database
    """)

col1, col2 = st.columns(2)
with col1:
    st.subheader('🎯 Benefits')
    st.markdown('- Early detection of retinopathy')
    st.markdown('- Accessible screening in primary care')
    st.markdown('- Evidence-based AI explanations')
    st.markdown('- Medical literature support')
    st.markdown('- Interactive patient education')

with col2:
    st.subheader('⚙️ How It Works')
    st.markdown('1. Upload a retinal image')
    st.markdown('2. AI analyzes image & generates heatmap')
    st.markdown('3. Get literature-supported AI explanation')
    st.markdown('4. Ask evidence-based questions')
    st.markdown('5. Search medical literature directly')

st.header('📤 Upload Retinal Image')
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Retinal Image', width=300)
    st.write("✅ Image successfully uploaded!")
    
    # Store original image and uploaded file in session state
    st.session_state.original_image = Image.open(uploaded_file).convert('RGB')
    st.session_state.uploaded_file = uploaded_file

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
                
                # Store model in session state for multi-layer analysis
                st.session_state.loaded_model = model
                
                # Generate heatmap
                fig = visualization_service.generate_gradcam_visualization(model, uploaded_file)
                st.session_state.heatmap_figure = fig
                
                # Get AI analysis of images with RAG enhancement
                with st.spinner("🤖 AI is analyzing the images with medical literature support..."):
                    ai_analysis = analyze_retinal_image_and_heatmap(
                        st.session_state.original_image,
                        fig,
                        results,
                        patient_age,
                        diabetes_duration
                    )
                    st.session_state.ai_analysis = ai_analysis
                    st.session_state.analysis_complete = True
                
                st.success("✅ Analysis complete with medical literature support!")
                
        except FileNotFoundError:
            st.error("❌ Model file 'classifier.pt' not found. Please ensure the model file is in the same directory.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")

# Display results if analysis is complete
if st.session_state.analysis_complete:
    st.header("📊 Analysis Results")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI Analysis", "📈 Technical Results", "🔥 Heatmap Visualization", "📚 Literature Search"])
    
    with tab1:
        st.subheader("🤖 Evidence-Based AI Analysis")
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
        st.subheader("🔥 Enhanced GradCAM Heatmap Analysis")
        st.markdown("*These improved heatmaps show which areas of the retina the AI model focused on for its prediction.*")
        
        # Add visualization options
        viz_option = st.radio(
            "Choose visualization type:",
            ["Enhanced Standard View", "Multi-layer Analysis"],
            key="viz_type"
        )
        
        if st.session_state.heatmap_figure:
            if viz_option == "Enhanced Standard View":
                st.markdown("**Standard Enhanced GradCAM** - Shows original image, pure heatmap, and overlay with applied improvements:")
                st.pyplot(st.session_state.heatmap_figure)
                plt.close(st.session_state.heatmap_figure)
                

            
            elif viz_option == "Multi-layer Analysis":
                st.markdown("**Multi-layer GradCAM Analysis** - Shows how different layers of the AI model focus on different features:")
                
                with st.spinner("🔄 Generating multi-layer analysis..."):
                    try:
                        # First try to get model from session state
                        model = st.session_state.get('loaded_model', None)
                        
                        # If not available, try to reload it
                        if model is None:
                            try:
                                model = model_service.load_model("classifier.pt")
                                st.session_state.loaded_model = model
                            except Exception as load_error:
                                st.error(f"Failed to load model: {str(load_error)}")
                                model = None
                        
                        if model is not None:
                            multi_fig = visualization_service.generate_multi_layer_gradcam(model, st.session_state.uploaded_file)
                            st.pyplot(multi_fig)
                            plt.close(multi_fig)
                            
                            st.info("💡 **Interpretation Guide:**\n"
                                   "- **Layer 2**: Low-level features (edges, textures)\n"
                                   "- **Layer 3**: Mid-level features (shapes, patterns)\n"
                                   "- **Layer 4**: High-level features (complex structures, pathological signs)")
                        else:
                            st.error("Model not available for multi-layer analysis")
                    except Exception as e:
                        st.error(f"Error generating multi-layer analysis: {str(e)}")
                        # Fallback to standard view
                        st.markdown("**Falling back to standard view:**")
                        st.pyplot(st.session_state.heatmap_figure)
                        plt.close(st.session_state.heatmap_figure)
    
    with tab4:
        st.subheader("📚 Search Medical Literature")
        st.markdown("*Search through the medical literature database for specific information about diabetic retinopathy.*")
        
        search_query = st.text_input(
            "Search medical literature:",
            placeholder="e.g., 'proliferative diabetic retinopathy treatment', 'retinal screening guidelines'",
            key="literature_search"
        )
        
        if st.button("🔍 Search Literature", key="search_lit_btn"):
            if search_query.strip():
                with st.spinner("🔍 Searching medical literature..."):
                    search_results = search_medical_literature(search_query, k=5)
                    
                    if search_results and not any('error' in result for result in search_results):
                        st.subheader("📑 Search Results")
                        
                        for i, result in enumerate(search_results):
                            similarity_score = result.get('similarity_score', 0)
                            source_file = result.get('source_file', 'Unknown')
                            text = result.get('text', '')
                            
                            with st.expander(f"Result {i+1}: {source_file} (Similarity: {similarity_score:.2%})"):
                                st.markdown(f"**Source:** {source_file}")
                                st.markdown(f"**Relevance Score:** {similarity_score:.2%}")
                                st.markdown("**Content:**")
                                st.write(text[:500] + "..." if len(text) > 500 else text)
                    else:
                        st.warning("No relevant results found or search service unavailable.")
            else:
                st.warning("⚠️ Please enter a search query!")
    
    # Interactive Q&A Section
    st.header("💬 Ask Questions About Your Results")
    st.markdown("*Ask our AI expert any questions about your retinal analysis - now powered by medical literature for evidence-based answers.*")
    
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
            with st.spinner("🤖 AI expert is researching your question with medical literature..."):
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
                    st.subheader("🎯 Evidence-Based AI Expert Response")
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
                'original_image', 'heatmap_figure', 'qa_history', 'uploaded_file', 'loaded_model']:
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
