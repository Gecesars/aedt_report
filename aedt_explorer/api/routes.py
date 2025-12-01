from flask import jsonify, request, current_app, send_from_directory
from . import api
from ..services.aedt_client import AedtClient
from ..services.jobs import job_manager
import os

aedt_client = AedtClient()

@api.route('/health')
def health():
    return jsonify({"status": "ok"})

# --- Project & Design Endpoints ---

@api.route('/projects', methods=['GET'])
def list_projects():
    try:
        projects = aedt_client.list_projects()
        return jsonify(projects)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/projects/<project_id>/designs', methods=['GET'])
def list_designs(project_id):
    # In our simple implementation, project_id is the filename
    # We need to reconstruct the full path
    base_dir = current_app.config.get("AEDT_PROJECTS_BASE_DIR")
    project_path = os.path.join(base_dir, project_id)
    
    try:
        designs = aedt_client.list_designs(project_path)
        return jsonify(designs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/projects/<project_id>/designs/<design_name>/metadata', methods=['GET'])
def get_design_metadata(project_id, design_name):
    base_dir = current_app.config.get("AEDT_PROJECTS_BASE_DIR")
    project_path = os.path.join(base_dir, project_id)
    try:
        metadata = aedt_client.get_design_metadata(project_path, design_name)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/projects/<project_id>/designs/<design_name>/reports', methods=['GET'])
def list_reports(project_id, design_name):
    base_dir = current_app.config.get("AEDT_PROJECTS_BASE_DIR")
    project_path = os.path.join(base_dir, project_id)
    try:
        reports = aedt_client.list_reports(project_path, design_name)
        return jsonify(reports)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/projects/<project_id>/designs/<design_name>/sparameters', methods=['POST'])
def get_sparameters(project_id, design_name):
    base_dir = current_app.config.get("AEDT_PROJECTS_BASE_DIR")
    project_path = os.path.join(base_dir, project_id)
    data = request.json
    report_name = data.get('report_name')
    traces = data.get('traces', [])
    
    try:
        result = aedt_client.get_sparameters(project_path, design_name, report_name, traces)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/projects/<project_id>/designs/<design_name>/3d-image', methods=['GET'])
def get_3d_image(project_id, design_name):
    base_dir = current_app.config.get("AEDT_PROJECTS_BASE_DIR")
    project_path = os.path.join(base_dir, project_id)
    
    # Output dir for images (static folder)
    static_dir = os.path.join(current_app.root_path, 'static', 'images')
    os.makedirs(static_dir, exist_ok=True)
    
    try:
        filename = aedt_client.export_3d_model_image(project_path, design_name, static_dir)
        # Return URL
        return jsonify({"url": f"/static/images/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Job Endpoints ---

@api.route('/jobs', methods=['POST'])
def create_job():
    data = request.json
    job_type = data.get('type')
    params = data.get('params', {})
    
    # Add project path to params
    project_id = data.get('project_id')
    if project_id:
        base_dir = current_app.config.get("AEDT_PROJECTS_BASE_DIR")
        params['project_path'] = os.path.join(base_dir, project_id)
    
    params['design_name'] = data.get('design_name')
    
    target_callable = None
    if job_type == 'field_animation':
        target_callable = aedt_client.run_field_animation_job
    elif job_type == 'parametric_sweep':
        target_callable = aedt_client.run_parametric_sweep_job
    else:
        return jsonify({"error": "Invalid job type"}), 400
        
    job_id = job_manager.create_job(job_type, params, target_callable)
    return jsonify({"job_id": job_id, "status": "pending"})

@api.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@api.route('/jobs/<job_id>/frames', methods=['GET'])
def get_job_frames(job_id):
    job = job_manager.get_job(job_id)
    if not job or job['status'] != 'done':
        return jsonify({"error": "Job not done or not found"}), 404
    
    result = job.get('result', {})
    frames = result.get('frames', [])
    # Convert file paths to URLs if they are in static
    # Assuming frames are saved in static/images/jobs/<job_id>
    # We need to ensure the job saves them there.
    
    return jsonify({"frames": frames})
