import os
import logging
import glob
from pyaedt import Desktop, Hfss
from flask import current_app

logger = logging.getLogger(__name__)

class AedtClient:
    _desktop = None
    _desktop_session = None

    @classmethod
    def get_desktop(cls):
        """
        Ensures a single instance of Desktop is running.
        """
        if cls._desktop is None:
            version = current_app.config.get("AEDT_VERSION", "2025.2")
            logger.info(f"Initializing AEDT Desktop version {version}...")
            try:
                # non_graphical=False means GUI will be visible
                # new_desktop_session=False tries to attach to existing session
                cls._desktop = Desktop(
                    specified_version=version,
                    non_graphical=False,
                    new_desktop_session=False,
                    close_on_exit=False,
                    student_version=False
                )
                logger.info("AEDT Desktop initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize AEDT Desktop: {e}")
                raise e
        return cls._desktop

    def list_projects(self, base_dir: str = None) -> list[dict]:
        """
        Lists .aedt projects in the base directory.
        """
        if base_dir is None:
            base_dir = current_app.config.get("AEDT_PROJECTS_BASE_DIR")
        
        if not os.path.exists(base_dir):
            logger.warning(f"Project directory not found: {base_dir}")
            return []

        projects = []
        # Find all .aedt files
        aedt_files = glob.glob(os.path.join(base_dir, "*.aedt"))
        
        for path in aedt_files:
            name = os.path.basename(path)
            projects.append({
                "id": name, # Using filename as ID for simplicity
                "name": name,
                "path": path
            })
        return projects

    def open_project(self, project_path: str):
        """
        Opens a project in the active Desktop session.
        """
        desktop = self.get_desktop()
        try:
            logger.info(f"Opening project: {project_path}")
            # load_project returns the project object
            project = desktop.load_project(project_path)
            return project
        except Exception as e:
            logger.error(f"Error opening project {project_path}: {e}")
            raise

    def list_designs(self, project_path: str) -> list[dict]:
        """
        Lists designs in a project.
        """
        # We need to open/get the project first
        # Note: In a real app, we might want to cache the project object
        # but PyAEDT handles project management well.
        project_name = os.path.basename(project_path).replace(".aedt", "")
        desktop = self.get_desktop()
        
        # Check if project is already open in AEDT
        project_list = desktop.project_list()
        if project_name not in project_list:
             self.open_project(project_path)
        
        # Get the project object via PyAEDT (using Hfss just to access project list or generic app)
        # A cleaner way is to use desktop.project_list and then get the object
        # But PyAEDT apps like Hfss(projectname, designname) are the standard way.
        # We'll use a generic approach to list designs.
        
        # We can use the project object returned by load_project or get it from desktop
        # Since PyAEDT 0.7+, desktop.projects is a dictionary-like object? 
        # Actually desktop.project_list() returns names.
        
        # Let's try to instantiate an app to get design list, or use desktop.odesktop.GetProjects()
        # But we want to stick to PyAEDT API.
        
        # Workaround: Open the project and inspect it.
        # Hfss(projectname) attaches to the first HFSS design or creates one.
        # We want to LIST them.
        
        # Using internal OProject object from PyAEDT if possible, or just iterating.
        # desktop.active_project gives the active project.
        
        # Let's assume we set the project as active.
        desktop.odesktop.SetActiveProject(project_name)
        oproject = desktop.odesktop.GetActiveProject()
        
        designs = []
        if oproject:
            for design in oproject.GetTopDesignList():
                # design is a name string
                # We can try to guess type or get it
                odesign = oproject.GetDesign(design)
                design_type = odesign.GetDesignType()
                designs.append({
                    "name": design,
                    "type": design_type
                })
        
        return designs

    def get_design_metadata(self, project_path: str, design_name: str) -> dict:
        """
        Gets metadata (setups, sweeps, ports) for a specific design.
        """
        project_name = os.path.basename(project_path).replace(".aedt", "")
        
        # Initialize HFSS to interact with the design
        # This assumes it's an HFSS design. If it's Maxwell, we might need Maxwell3d.
        # For the MVP we assume HFSS as per requirements.
        try:
            app = Hfss(projectname=project_name, designname=design_name, specified_version=current_app.config.get("AEDT_VERSION"))
        except Exception as e:
            logger.error(f"Could not attach to design {design_name}: {e}")
            raise

        metadata = {
            "setups": app.setup_names,
            "sweeps": [], # Will populate below
            "ports": [p.name for p in app.port_list] if hasattr(app, 'port_list') else [], # PyAEDT property
            "variables": app.variable_manager.variable_names
        }
        
        # Get sweeps for each setup
        for setup in app.setup_names:
            sweeps = app.get_sweeps(setup)
            for sweep in sweeps:
                metadata["sweeps"].append(f"{setup}:{sweep}")
                
        return metadata

    def list_reports(self, project_path: str, design_name: str) -> list[dict]:
        project_name = os.path.basename(project_path).replace(".aedt", "")
        app = Hfss(projectname=project_name, designname=design_name, specified_version=current_app.config.get("AEDT_VERSION"))
        
        reports = []
        all_reports = app.post.reports_by_category
        # all_reports is a generic accessor, let's use app.post.all_report_names
        report_names = app.post.all_report_names
        
        for r in report_names:
            # We could try to get report type, but PyAEDT might not expose it easily without parsing
            reports.append({"name": r})
            
        return reports

    def get_sparameters(self, project_path: str, design_name: str, report_name: str, traces: list[str] = None) -> dict:
        """
        Gets S-parameter data from a report.
        """
        project_name = os.path.basename(project_path).replace(".aedt", "")
        app = Hfss(projectname=project_name, designname=design_name, specified_version=current_app.config.get("AEDT_VERSION"))
        
        # If report_name is provided, we try to get data from it.
        # PyAEDT's get_solution_data_per_report is useful.
        
        try:
            # This returns a SolutionData object
            solution_data = app.post.get_solution_data(expressions=traces, report_name=report_name)
            
            if not solution_data:
                return {"error": "No data found"}
            
            # Format for frontend: freq vs value
            # Assuming frequency domain
            freqs = solution_data.primary_sweep_values
            
            results = {
                "frequency": freqs,
                "traces": {}
            }
            
            for trace in traces:
                # data_magnitude or data_db depending on what user wants?
                # Usually we want complex or dB. Let's send dB for now as it's common for S-params
                # But the report might already be in dB.
                # If the report is a plot, it has processed data.
                # solution_data.data_real, data_imag, data_magnitude, data_db
                
                # Let's check the unit of the trace in the report if possible, or just send what we get.
                # solution_data.values_real gives the raw values if they are real, or we can compute.
                
                # For safety, let's send magnitude (dB) if it's S-parameter-like
                if "dB" in trace or "S" in trace:
                     vals = solution_data.data_db(trace)
                else:
                     vals = solution_data.data_real(trace)
                
                results["traces"][trace] = vals
                
            return results

        except Exception as e:
            logger.error(f"Error retrieving S-parameters: {e}")
            raise

    def export_3d_model_image(self, project_path: str, design_name: str, output_dir: str) -> str:
        project_name = os.path.basename(project_path).replace(".aedt", "")
        app = Hfss(projectname=project_name, designname=design_name, specified_version=current_app.config.get("AEDT_VERSION"))
        
        filename = f"{design_name}_3d.png"
        full_path = os.path.join(output_dir, filename)
        
        # Export model image
        app.post.export_model_picture(full_name=full_path, show_axis=True, show_grid=False, show_ruler=False)
        
        return filename

    # --- Job Targets (Async) ---
    # These methods are static or class methods because they run in a separate thread 
    # and might need to instantiate their own PyAEDT app wrapper if not careful,
    # BUT since we are sharing the Desktop session, we can reuse the logic.
    # However, we must be careful about thread safety with COM.
    # PyAEDT is generally thread-safe-ish if we don't conflict on the same object.
    
    def run_field_animation_job(self, params: dict) -> dict:
        """
        Generates frames for field animation.
        Params: project_path, design_name, setup_name, solution_type, variations
        """
        project_path = params.get('project_path')
        design_name = params.get('design_name')
        # Example params: {"phase_start": 0, "phase_stop": 180, "phase_step": 10}
        # Or generic variations list
        
        # We need to be careful with threading and COM
        import pythoncom
        pythoncom.CoInitialize()
        
        try:
            project_name = os.path.basename(project_path).replace(".aedt", "")
            # We instantiate Hfss again in this thread
            app = Hfss(projectname=project_name, designname=design_name, specified_version=current_app.config.get("AEDT_VERSION"))
            
            output_dir = os.path.join(current_app.root_path, 'static', 'images', 'jobs', params.get('job_id', 'temp'))
            os.makedirs(output_dir, exist_ok=True)
            
            frames = []
            
            # Example logic: Sweep phase of a source or just a generic animation
            # For MVP, let's assume we are varying a variable that affects the field (like a phase variable)
            # or we are just exporting a field plot that is already set up for animation?
            # PyAEDT has animate_fields_from_aedtplt but that might be complex.
            # Let's do a simple parameter sweep that affects the field.
            
            param_name = params.get('param_name', 'Phase')
            start = float(params.get('start', 0))
            stop = float(params.get('stop', 360))
            step = float(params.get('step', 10))
            
            current_val = start
            frame_idx = 0
            
            while current_val <= stop:
                # Update variable
                app[param_name] = f"{current_val}deg"
                
                # We might need to resolve if it's a post-processing variable or requires re-simulation
                # If it's just a post-processing variable (like in Edit Sources), we don't need analyze.
                # If it's a geometry variable, we need analyze.
                
                # Assuming post-processing for "animation" usually.
                # But if user wants parametric sweep, that's the other job.
                
                # Export plot
                # We need a plot name.
                plot_name = params.get('plot_name')
                if plot_name:
                    filename = f"frame_{frame_idx:03d}.png"
                    full_path = os.path.join(output_dir, filename)
                    app.post.export_field_plot(plot_name, full_path)
                    
                    # Convert to URL
                    frames.append(f"/static/images/jobs/{params.get('job_id', 'temp')}/{filename}")
                
                current_val += step
                frame_idx += 1
                
            return {"status": "done", "frames": frames}
            
        except Exception as e:
            logger.error(f"Field animation job failed: {e}")
            raise
        finally:
            pythoncom.CoUninitialize()

    def run_parametric_sweep_job(self, params: dict) -> dict:
        """
        Runs parametric sweep and exports plots.
        """
        project_path = params.get('project_path')
        design_name = params.get('design_name')
        
        import pythoncom
        pythoncom.CoInitialize()
        
        try:
            project_name = os.path.basename(project_path).replace(".aedt", "")
            app = Hfss(projectname=project_name, designname=design_name, specified_version=current_app.config.get("AEDT_VERSION"))
            
            output_dir = os.path.join(current_app.root_path, 'static', 'images', 'jobs', params.get('job_id', 'temp'))
            os.makedirs(output_dir, exist_ok=True)
            
            frames = []
            values = []
            
            param_name = params.get('param_name')
            start = float(params.get('start'))
            stop = float(params.get('stop'))
            step = float(params.get('step'))
            
            current_val = start
            frame_idx = 0
            
            while current_val <= stop:
                # Update geometry/setup parameter
                app[param_name] = f"{current_val}mm" # Assuming length/distance
                
                # Run simulation
                # This is blocking and takes time
                setup_name = params.get('setup_name', app.setup_names[0] if app.setup_names else 'Setup1')
                app.analyze_setup(setup_name)
                
                # Export result (e.g. Radiation Pattern)
                filename = f"frame_{frame_idx:03d}.png"
                full_path = os.path.join(output_dir, filename)
                
                # Example: Export 3D Plot or 2D Report
                # For radiation pattern, we might want a far field plot
                # app.post.create_fieldplot_surface... or export_model_picture if we want to see geometry change
                # Or export a report image.
                
                # Let's assume we want to see the Far Field 3D plot if it exists
                # Or just the model 3D view to see geometry change?
                # User said: "plotar o diagrama e então captar essa imagem" -> Radiation pattern.
                
                # We can create a report and export it as image
                # Or if there is a 3D polar plot in the modeler window.
                
                # Let's try to export the model window which might contain the 3D pattern if visible.
                app.post.export_model_picture(full_name=full_path)
                
                frames.append(f"/static/images/jobs/{params.get('job_id', 'temp')}/{filename}")
                values.append(current_val)
                
                current_val += step
                frame_idx += 1
                
            return {"status": "done", "frames": frames, "values": values}
            
        except Exception as e:
            logger.error(f"Parametric sweep job failed: {e}")
            raise
        finally:
            pythoncom.CoUninitialize()
