'''
runpod | serverless | pod_worker.py
Called to convert a container into a worker pod for the runpod serverless platform.
'''

import os
import shutil

from .modules import lifecycle, job
from .modules.logging import log


def start_worker():
    '''
    Starts the worker.
    '''
    worker_life = lifecycle.LifecycleManager()
    worker_life.heartbeat_ping()

    while True:
        next_job = job.get(worker_life.worker_id)

        if next_job is not None:
            worker_life.job_id = next_job['id']

            try:
                output_urls, job_duration_ms = job.run(next_job['id'], next_job['input'])
                job.post(worker_life.worker_id, next_job['id'], output_urls, job_duration_ms)
            except (ValueError, RuntimeError) as err:
                job.error(worker_life.worker_id, next_job['id'], str(err))

            # -------------------------------- Job Cleanup ------------------------------- #
            shutil.rmtree("input_objects", ignore_errors=True)
            shutil.rmtree("output_objects", ignore_errors=True)

            if os.path.exists('output.zip'):
                os.remove('output.zip')

            worker_life.job_id = None

        if os.environ.get('WEBHOOK_GET_WORK', None) is None:
            log("Local testing complete, exiting.")
            break