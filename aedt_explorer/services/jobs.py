import uuid
import threading
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class JobManager:
    def __init__(self, max_workers=1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs = {}

    def create_job(self, job_type: str, params: dict, target_callable) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "type": job_type,
            "status": "pending",
            "params": params,
            "result": None,
            "error": None
        }
        
        logger.info(f"Created job {job_id} of type {job_type}")
        self.executor.submit(self._run_job, job_id, target_callable, params)
        return job_id

    def _run_job(self, job_id, target_callable, params):
        logger.info(f"Starting job {job_id}")
        self.jobs[job_id]["status"] = "running"
        try:
            result = target_callable(params)
            self.jobs[job_id]["result"] = result
            self.jobs[job_id]["status"] = "done"
            logger.info(f"Job {job_id} completed successfully")
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self.jobs[job_id]["error"] = str(e)
            self.jobs[job_id]["status"] = "error"

    def get_job(self, job_id: str) -> dict:
        return self.jobs.get(job_id)

# Singleton instance
job_manager = JobManager()
