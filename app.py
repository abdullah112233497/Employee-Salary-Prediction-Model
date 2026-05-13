import gradio as gr
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('salary_model.pkl')

# Define the prediction function
def predict_salary(job_title, education_level, experience_years, skills_count, industry, company_size, location, remote_work, certifications):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'job_title': [job_title],
        'education_level': [education_level],
        'experience_years': [experience_years],
        'skills_count': [skills_count],
        'industry': [industry],
        'company_size': [company_size],
        'location': [location],
        'remote_work': [remote_work],
        'certifications': [certifications]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    return f"${prediction[0]:,.2f}"

# Define the UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown(
        """
        # 💰 Employee Salary Predictor
        ### Predict your potential salary based on job title, experience, and skills.
        """
    )
    
    with gr.Row():
        with gr.Column():
            job_title = gr.Dropdown(
                label="Job Title",
                choices=["AI Engineer", "Backend Developer", "Business Analyst", "Cloud Engineer", "Cybersecurity Analyst", "Data Analyst", "Data Scientist", "DevOps Engineer", "Frontend Developer", "Machine Learning Engineer", "Product Manager", "Software Engineer"]
            )
            education_level = gr.Dropdown(
                label="Education Level",
                choices=["Bachelor", "Diploma", "High School", "Master", "PhD"]
            )
            industry = gr.Dropdown(
                label="Industry",
                choices=["Consulting", "Education", "Finance", "Government", "Healthcare", "Manufacturing", "Media", "Retail", "Technology", "Telecom"]
            )
            company_size = gr.Dropdown(
                label="Company Size",
                choices=["Enterprise", "Large", "Medium", "Small", "Startup"]
            )
        
        with gr.Column():
            location = gr.Dropdown(
                label="Location",
                choices=["Australia", "Canada", "Germany", "India", "Netherlands", "Remote", "Singapore", "Sweden", "UK", "USA"]
            )
            remote_work = gr.Radio(
                label="Remote Work",
                choices=["Hybrid", "No", "Yes"],
                value="Yes"
            )
            experience_years = gr.Slider(
                label="Years of Experience",
                minimum=0,
                maximum=40,
                step=1,
                value=5
            )
            skills_count = gr.Slider(
                label="Number of Skills",
                minimum=0,
                maximum=50,
                step=1,
                value=10
            )
            certifications = gr.Slider(
                label="Number of Certifications",
                minimum=0,
                maximum=20,
                step=1,
                value=2
            )
            
    predict_btn = gr.Button("Calculate Expected Salary", variant="primary")
    
    output = gr.Textbox(label="Predicted Annual Salary", placeholder="Prediction will appear here...")
    
    predict_btn.click(
        fn=predict_salary,
        inputs=[job_title, education_level, experience_years, skills_count, industry, company_size, location, remote_work, certifications],
        outputs=output
    )

    gr.Markdown(
        """
        ---
        *Note: This model is for educational purposes and provides estimates based on historical data.*
        """
    )

if __name__ == "__main__":
    demo.launch()
