import streamlit as st
import requests
import time
import pandas as pd
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c7c4c;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c7c4c;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8f2;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2c7c4c;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .treatment-box {
        background-color: #f5faff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #2c7c4c;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #666;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Define the URL of the FastAPI service
API_URL = 'http://localhost:8080/predict/'

# Sidebar content
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/plant-under-sun.png", width=100)
    st.markdown("## Plant Disease Detector")
    st.markdown("### How to use")
    st.markdown("""
    1. Upload an image of a plant leaf
    2. Wait for analysis
    3. View detailed results and treatment options
    """)
    
    st.markdown("### Problem Statement")
    st.markdown("""
    Urban strawberry cultivation faces challenges due to limited space, environmental fluctuations, and increased risk of plant diseases. Early detection and effective treatment are crucial for maximizing yield and ensuring healthy plants in urban environments.
    """)
    
    st.markdown("### Project Objectives")
    st.markdown("""
    Our project aims to help people grow strawberries in urban areas by providing an easy way to detect diseases early and recommend treatments.
    
    <span style="color:#2c7c4c;">
    🍓 For best results, keep temperature and humidity at optimal levels for strawberry growth.
    </span>
    """, unsafe_allow_html=True)

    st.markdown("### Group Members")
    st.markdown("""
    - DISSANAYAKE DMDSA - EN22179012  
    - MARLON KVA - EN22556066  
    - THARINDU ABEYSINGHE - EN22175816  
    """)

    st.markdown("### Supervisor")
    st.markdown("Ms. Shehani Jayasinghe")

    st.markdown("### About")
    st.markdown("""
    This application uses deep learning to identify plant diseases from leaf images. 
    It provides both chemical and natural treatment recommendations based on the detected disease.
    """)
    
    st.markdown("### Supported Plants")
    supported_plants = ["Berry", "Strawberry", "Apple", "Tomato", "Potato", "Corn", "Grape"]
    for plant in supported_plants:
        st.markdown(f"- {plant}")

# Main content
st.markdown("<h1 class='main-header'>Strawberry Detection and Treatment Advisor</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <div class='info-box' style="text-align:center;">
        <span style="font-size:2.5rem;">🍓🍓🍓</span><br>
        <span style="font-size:1.3rem; color:#d72660;"><b>Welcome to the Strawberry Disease Detector!</b></span><br>
        <span style="font-size:1.1rem; color:#2c7c4c;">
            Upload a clear image of a <b>strawberry leaf</b> showing symptoms.<br>
            <span style="font-size:1.5rem;">🍓</span> The system will analyze the image and provide<br>
            <b>disease identification</b> along with <b>treatment recommendations</b>.<br>
            <span style="font-size:1.5rem;">🍓🍃🍓</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for upload and preview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 class='sub-header'>Upload Image</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"], help="Upload a clear image of the affected plant leaf")
        
        if uploaded_file:
            # Display additional options
            image_preview = Image.open(uploaded_file)
            
            # Add analyze button
            analyze_button = st.button("Analyze Image", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("<h3 class='sub-header'>Image Preview</h3>", unsafe_allow_html=True)
        if uploaded_file:
            st.image(image_preview, caption='Uploaded Image', use_container_width=True)
        else:
            st.markdown("No image uploaded yet.")

# Results section
if 'uploaded_file' in locals() and uploaded_file is not None and 'analyze_button' in locals() and analyze_button:
    st.markdown("<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
    
    # Show loading spinner while processing
    with st.spinner('Analyzing leaf image... Please wait'):
        # Reset the file pointer
        uploaded_file.seek(0)
        
        # Prepare the file for API request
        files = {'file': uploaded_file.getvalue()}
        
        try:
            # Make API request
            response = requests.post(API_URL, files=files, timeout=15)
            
            # Process response
            if response.status_code == 200:
                prediction = response.json()
                if 'predicted_class' in prediction:
                    # Show success message
                    st.success("Analysis complete!")
                    
                    # Display results in a nice format
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.markdown(f"### Diagnosis: {prediction['predicted_class']}")
                        
                        # Confidence score with progress bar
                        confidence = prediction['confidence']
                        st.markdown(f"**Confidence Level:** {confidence:.1f}%")
                        st.progress(confidence / 100)
                        
                        # Disease information
                        st.markdown("#### Disease Information")
                        
                        # Create info based on the disease name
                        disease_info = {
                            "Healthy": "The plant appears healthy with no visible signs of disease.",
                            "Bacterial": "Bacterial diseases are caused by bacteria and often result in spots, blights, and wilts.",
                            "Viral": "Viral diseases are systemic and can cause mosaic patterns, yellowing, and stunting.",
                            "Fungal": "Fungal diseases are common and can cause spots, powdery mildew, and various rots."
                        }
                        
                        disease_type = "Unknown"
                        for key in disease_info:
                            if key.lower() in prediction['predicted_class'].lower():
                                disease_type = key
                                break
                        
                        st.markdown(disease_info.get(disease_type, "No specific information available for this disease."))
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        # Plant health meter
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.markdown("### Plant Health Meter")
                        
                        # Determine health score based on confidence and disease type
                        health_score = 95 if "healthy" in prediction['predicted_class'].lower() else 100 - confidence
                        
                        # Color coding based on health
                        if health_score > 80:
                            meter_color = "green"
                            health_status = "Good"
                        elif health_score > 50:
                            meter_color = "orange"
                            health_status = "Moderate"
                        else:
                            meter_color = "red"
                            health_status = "Poor"
                            
                        st.markdown(f"**Status: {health_status}**")
                        st.markdown(
                            f"""
                            <div style="margin-top:10px; margin-bottom:20px; width:100%; height:20px; background-color:#f0f0f0; border-radius:10px;">
                                <div style="width:{health_score}%; height:100%; background-color:{meter_color}; border-radius:10px;"></div>
                            </div>
                            <p style="text-align:center;">{health_score:.1f}%</p>
                            """, 
                            unsafe_allow_html=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Treatment options in tabs
                    st.markdown("<h3 class='sub-header'>Treatment Options</h3>", unsafe_allow_html=True)
                    
                    tab1, tab2 = st.tabs(["Chemical Treatment", "Natural/Organic Treatment"])
                    
                    with tab1:
                        st.markdown("<div class='treatment-box'>", unsafe_allow_html=True)
                        st.markdown("### Chemical Solution")
                        st.markdown(prediction.get('chemical_solution', 'No chemical treatment information available.'))
                        
                        # Add application instructions if available
                        if 'application_method' in prediction:
                            st.markdown("#### Application Method")
                            st.markdown(prediction['application_method'])
                            
                        st.markdown("#### Precautions")
                        st.markdown("- Always wear protective gear when applying chemicals")
                        st.markdown("- Follow manufacturer's instructions for dosage and application")
                        st.markdown("- Keep chemicals away from children and pets")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    with tab2:
                        st.markdown("<div class='treatment-box'>", unsafe_allow_html=True)
                        st.markdown("### Natural/Organic Solution")
                        st.markdown(prediction.get('natural_solution', 'No natural treatment information available.'))
                        
                        st.markdown("#### Benefits of Natural Treatment")
                        st.markdown("- Environmentally friendly")
                        st.markdown("- Safe for beneficial insects")
                        st.markdown("- No chemical residues on produce")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                else:
                    st.error("The prediction service returned an incomplete response. Please try again.")
            else:
                st.error(f"Error: Unable to get a prediction. Status code: {response.status_code}")
                st.markdown("Please check if the API service is running and accessible.")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the prediction service: {e}")
            st.markdown("""
            **Troubleshooting tips:**
            1. Make sure the API server is running
            2. Check that the URL is correct
            3. Verify your network connection
            """)

# Footer
st.markdown("""
<div class='footer'>
© 2025 Plant Disease Detector | Developed by Group 13 – Tharindu Ravihara, Sasmitha Dissanayake, Marlon Avishka, Dinupa Devinda, and Kusal Punchihewa<br>
<b>Group 18 Members:</b><br>
DISSANAYAKE DMDSA - EN22179012<br>
MARLON KVA - EN22556066<br>
THARINDU ABEYSINGHE - EN22175816<br>
<b>Supervised by Ms. Shehani Jayasinghe</b>
</div>
""", unsafe_allow_html=True)