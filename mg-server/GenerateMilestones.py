from flask import Flask, request, render_template, jsonify
import pandas as pd
import ollama
import concurrent.futures
from flask_cors import CORS
import re
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)

# --------------------------------------------
# Initialize Firebase Admin
# --------------------------------------------
cred = credentials.Certificate("awjplatform-f9f40-firebase-adminsdk-fbsvc-6cd760e165.json")
firebase_admin.initialize_app(cred, {
    'projectId': 'awjplatform-f9f40',
    'databaseURL': 'https://awjplatform-f9f40.firebaseio.com' 
})
db = firestore.client()

# --------------------------------------------
# Caching dictionaries for model outputs
# --------------------------------------------
project_milestones_cache = {}
milestone_details_cache = {}

# --------------------------------------------------
# Step 0: Load the dataset and cache preview from JSONL
# --------------------------------------------------
file_path = "finetune_dataset.jsonl"
try:
    data = pd.read_json(file_path, lines=True)
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: The dataset file was not found. Check the file path!")
    exit()
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

# Cache the dataset preview (computed only once)
dataset_preview = data.to_string()

# --------------------------------------------------
# Step 2: Generate Project Milestones (Milestone Names)
# --------------------------------------------------
def generate_project_milestones(company_description, project_name, project_description, attempts=0):
    global dataset_preview
    # Create a cache key based on inputs
    cache_key = (company_description, project_name, project_description, dataset_preview)
    if cache_key in project_milestones_cache:
        return project_milestones_cache[cache_key]
    
    prompt = f"""
Based on the following company and project overview:

Company Description: {company_description}
Project Name: {project_name}
Project Description: {project_description}

And using the dataset preview below as context:
{dataset_preview}

Generate a comma-separated list of realistic, sequential and unique milestone names that reflect the key stages of this project: {project_description}. 
Do not include numbering; just output the milestone names.

**Output format:**
Milestones: <milestone1>, <milestone2>, <milestone3>[, ...]
    """
    response = ollama.chat(model="mistral", messages=[
        {"role": "system", "content": "You are an AI that generates realistic project milestones."},
        {"role": "user", "content": prompt}
    ])
    generated_text = response.message.content.strip()

    milestones = []
    for line in generated_text.splitlines():
        if line.lower().startswith("milestones:"):
            milestones_str = line.split(":", 1)[1]
            milestones = [m.strip() for m in milestones_str.split(",") if m.strip()]
            break

    if not milestones:
        if attempts < 3:
            print(f"Retrying milestone generation for project '{project_name}' (attempt {attempts+1})...")
            return generate_project_milestones(company_description, project_name, project_description, attempts+1)
        else:
            print(f"Warning: Could not generate milestones for project '{project_name}' after several attempts. Using default milestones.")
            milestones = [
    "Product Backlog Creation",
    "Sprint Planning",
    "Design & Prototyping",
    "Sprint Development Cycles",
    "Continuous Testing & QA",
    "Sprint Reviews & Retrospectives",
    "Final Integration & UAT (User Acceptance Testing)",
    "Release & Deployment",
    "Post-Release Support & Maintenance"
]
    
    project_milestones_cache[cache_key] = milestones
    return milestones

# ---------------------------------------------------------------------
# Step 3: Generate Milestone Details (Tasks, Time, Resources, KPIs, etc.)
# ---------------------------------------------------------------------
def generate_milestone_details_ollama(project_name, project_description, milestone):
    global dataset_preview
    # Create a cache key based on inputs
    cache_key = (project_name, project_description, milestone, dataset_preview)
    if cache_key in milestone_details_cache:
        return milestone_details_cache[cache_key]
    
    prompt = f"""
Based on the following project overview:

Project Name: {project_name}
Project Description: {project_description}
Milestone: {milestone}

Use the following dataset preview as context:
{dataset_preview}

For this milestone, generate:
- Task: multiple comma-separated tasks that reflect the milestone: {milestone} of this project: {project_description}.
- Time Estimate (Days): numeric value (no extra text) representing the total estimated time in days for the milestone: {milestone}.
- KPI: exactly one KPI that represents the  milestone: {milestone} as a whole (e.g., "Model accuracy > 85%").
- Risk Factors: milestone-specific risk for the milestone: {milestone}.
- Risk Indicator: output only one value: "Low", "Medium", or "High" with no extra explanation for the milestone: {milestone}.

**Important Instruction:**
- Do not output any placeholder text such as "<To be determined>", "TBD", "to be determined", "n/a", or similar.

**Output format:**
Milestone: {milestone}
Task: <task1>, <task2>, <task3>
Time Estimate (Days): <estimate>
KPI: <kpi>
Risk Factors: <risk>
Risk Indicator: <indicator>
    """
    response = ollama.chat(model="mistral", messages=[
        {"role": "system", "content": "You are an AI that generates structured project data."},
        {"role": "user", "content": prompt}
    ])
    generated_text = response.message.content.strip()

    # Parse the generated text into a dictionary
    milestone_data = {}
    expected_keys = [
        "Milestone", "Task", "Time Estimate (Days)", "KPI", "Risk Factors", "Risk Indicator"
    ]
    for line in generated_text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if key in expected_keys:
                milestone_data[key] = value.strip()
            else:
                print(f"Warning: Unexpected column '{key}' encountered in milestone generation. Ignoring.")
    
    if "Milestone" not in milestone_data or not milestone_data["Milestone"]:
        milestone_data["Milestone"] = milestone
    else:
        milestone_data["Milestone"] = milestone  # enforce the milestone name

    milestone_details_cache[cache_key] = milestone_data
    return milestone_data

# ----------------------------------------------------
# Helper: Check if a string has balanced parentheses
# ----------------------------------------------------
def is_balanced(s):
    return s.count("(") == s.count(")")

# ---------------------------------------------------------------------
# Process milestones in parallel
# ---------------------------------------------------------------------
def process_milestone(milestone, project_name, project_description):
    print("Processing milestone:", milestone)
    milestone_details = generate_milestone_details_ollama(project_name, project_description, milestone)
    milestone_details["Milestone"] = milestone  # enforce the milestone name
    new_row = {
        "Project Name": project_name,
        "Project Description": project_description,
    }
    new_row.update(milestone_details)
    return new_row

# ----------------------------------------------------
# Run Model: Receives input data and returns output data
# ----------------------------------------------------
def run_model(company_description, project_name, project_description):
    # Generate project milestones.
    milestone_names = generate_project_milestones(company_description, project_name, project_description)
    print(f"Milestones: {milestone_names}")

    # Fix the milestone list (merge split milestones if needed).
    fixed_milestones = []
    i = 0
    while i < len(milestone_names):
        combined = milestone_names[i]
        while not is_balanced(combined) and i < len(milestone_names)-1:
            i += 1
            combined += ", " + milestone_names[i]
        fixed_milestones.append(combined)
        i += 1
    print("Fixed Milestones:", fixed_milestones)

    # Process milestones concurrently while preserving order.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(
            lambda milestone: process_milestone(milestone, project_name, project_description),
            fixed_milestones
        ))
    augmented_data = results

    return augmented_data

# (Optional) If you need to expose the dataset elsewhere.
def get_dataset():
    return data

# ---------------------------------------------
# Flask Routes
# ---------------------------------------------
@app.route('/generate-milestones', methods=['POST'])
def index():
    # Get form data
    project_name = request.form['project_name']
    project_description = request.form['project_description']
    UserID = request.form['user_id']
    company_id = request.form['company_id']
    ProjectID = request.form['project_id']

    # Retrieve user document
    user_doc = db.collection("User").document(UserID).get()
    if not user_doc.exists:
        return "User not found", 404
    user_data = user_doc.to_dict()
    company_ref_value = user_data.get("CompanyID")
    if not company_ref_value:
        return "Company ID not found for user", 400
    if isinstance(company_ref_value, firestore.DocumentReference):
        company_id = company_ref_value.id
    else:
        company_id = company_ref_value

    # Retrieve company details
    company_ref = db.collection("Company").document(company_id)
    company_doc = company_ref.get()
    if not company_doc.exists:
        return "Company not found!!", 404
    company_data = company_doc.to_dict()
    company_description = company_data.get("CompDescription", "")

    # Generate milestones using your model
    print("start generating")
    milestones_generated = run_model(company_description, project_name, project_description)
    print(f"Generated Milestones: {milestones_generated}")

    # Retrieve the saved project (assumed saved by client) from Firestore
    project_collection = db.collection("Project")
    query_projects = project_collection.where("ProjectName", "==", project_name).stream()
    project_doc = None
    for doc in query_projects:
        project_doc = doc
        break
    if project_doc is None:
        return "Project not found", 404
    project_data = project_doc.to_dict()
    print(f"Retrieved project with ID: {ProjectID}")

    # Check if milestones already exist for this project
    milestone_collection = db.collection("Milestones")
    query_milestones = milestone_collection.where("ProjectID", "==", ProjectID).stream()
    existing_milestones = [doc.to_dict() for doc in query_milestones]

    if not existing_milestones:
        # Determine next available milestone ID
        milestone_docs = milestone_collection.get()
        existing_milestone_numbers = []
        for doc in milestone_docs:
            m_match = re.match(r"m(\d{3})$", doc.id)
            if m_match:
                existing_milestone_numbers.append(int(m_match.group(1)))
        existing_milestone_numbers.sort()
        next_milestone_number = 1
        for num in existing_milestone_numbers:
            if num != next_milestone_number:
                break
            next_milestone_number += 1

        # Save generated milestones to Firestore
        for milestone in milestones_generated:
            milestone_id = f"m{next_milestone_number:03d}"
            next_milestone_number += 1
            milestone_data = {
                "ProjectID": ProjectID,
                "Milestone": milestone.get("Milestone", ""),
                "MilestoneID": milestone_id,
                "TimeEstimate": milestone.get("Time Estimate (Days)", ""),
                "Task": milestone.get("Task", ""),
                "KPI": milestone.get("KPI", ""),
                "RiskFactors": milestone.get("Risk Factors", ""),
                "RiskIndicator": milestone.get("Risk Indicator", ""),
            }
            milestone_collection.document(milestone_id).set(milestone_data)
            print(f"Saved milestone '{milestone.get('Milestone', '')}' with ID: {milestone_id}")
        query_milestones = milestone_collection.where("ProjectID", "==", ProjectID).stream()
        milestones = [doc.to_dict() for doc in query_milestones]
    else:
        milestones = existing_milestones

    return jsonify({
        "project_name": project_name,
        "project_id": ProjectID,
        "milestones": milestones
    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host="0.0.0.0")