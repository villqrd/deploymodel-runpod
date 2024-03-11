""" Used to launch the FastAPI web server when worker is running in API mode. """

# pylint: disable=too-few-public-methods, line-too-long

import os
import uuid
import threading
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any

import uvicorn
import requests
from fastapi import FastAPI, APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse

from .rp_handler import is_generator
from .rp_job import run_job, run_job_generator
from .worker_state import Jobs
from .rp_ping import Heartbeat
from ...version import __version__ as runpod_version

import pydantic

RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", None)


DESCRIPTION = """"""

# Add CLI tool suggestion if RUNPOD_PROJECT_ID is not set.
# if os.environ.get("RUNPOD_PROJECT_ID", None) is None:
#     DESCRIPTION += """

#     ℹ️ | Consider developing with our CLI tool to streamline your worker development process.

#     >_  wget -qO- cli.runpod.net | sudo bash
#     >_  runpodctl project create
#     """

RUN_DESCRIPTION = """
Initiates processing jobs, returning a unique job ID.
"""

RUNSYNC_DESCRIPTION = """
Executes processing jobs synchronously, returning the job's output directly.

This endpoint is ideal for tasks where immediate result retrieval is necessary,
streamlining the execution process by eliminating the need for subsequent
status or result checks.
"""

STREAM_DESCRIPTION = """
Continuously aggregates the output of a processing job, returning the full output once the job is complete.

This endpoint is especially useful for jobs where the complete output needs to be accessed at once. It provides a consolidated view of the results post-completion, ensuring that users can retrieve the entire output without the need to poll multiple times or manage partial results.

**Parameters:**
- **job_id** (string): The unique identifier of the job for which output is being requested. This ID is used to track the job's progress and aggregate its output.

**Returns:**
- **output** (Any): The aggregated output from the job, returned as a single entity once the job has concluded. The format of the output will depend on the nature of the job and how its results are structured.
"""

STATUS_DESCRIPTION = """
Checks the completion status of a processing job and returns its output if the job is complete.

This endpoint is invaluable for monitoring the progress of a job and obtaining the output only after the job has fully completed. It simplifies the process of querying job completion and retrieving results, eliminating the need for continuous polling or result aggregation.
"""
# **Note:** The availability of the `output` field is contingent on the job's completion status. If the job is still in progress, this field may be omitted or contain partial results, depending on the implementation.

# ------------------------------ Initializations ----------------------------- #
job_list = Jobs()
heartbeat = Heartbeat()


# ------------------------------- Input Objects ------------------------------ #
@dataclass
class Job:
    """Represents a job."""

    id: str
    input: Union[dict, list, str, int, float, bool]


@dataclass
class TestJob:
    """Represents a test job.
    input can be any type of data.
    """

    id: Optional[str] = None
    input: Optional[Union[dict, list, str, int, float, bool]] = None
    webhook: Optional[str] = None


@dataclass
class DefaultRequest:
    """Represents a test input."""

    input: Dict[str, Any]
    webhook: Optional[str] = None


# ------------------------------ Output Objects ------------------------------ #
@dataclass
class JobOutput:
    """Represents the output of a job."""

    id: str
    status: str
    output: Optional[Union[dict, list, str, int, float, bool]] = None
    error: Optional[str] = None


@dataclass
class StreamOutput:
    """Stream representation of a job."""

    id: str
    status: str = "IN_PROGRESS"
    stream: Optional[Union[dict, list, str, int, float, bool]] = None
    error: Optional[str] = None


class BaseResponse(pydantic.BaseModel):
    id: str
    status: str


# ------------------------------ Webhook Sender ------------------------------ #
def _send_webhook(url: str, payload: Dict[str, Any]) -> bool:
    """
    Sends a webhook to the provided URL.

    Args:
        url (str): The URL to send the webhook to.
        payload (Dict[str, Any]): The JSON payload to send.

    Returns:
        bool: True if the request was successful, False otherwise.
    """
    with requests.Session() as session:
        try:
            response = session.post(url, json=payload, timeout=10)
            response.raise_for_status()  # Raises exception for 4xx/5xx responses
            return True
        except requests.RequestException as err:
            print(f"WEBHOOK | Request to {url} failed: {err}")
            return False


def get_input_model(handler):
    input_model = handler.__annotations__.get("input")
    if not input_model:
        raise ValueError(
            f"Handler {handler.__name__} must have an input type annotation"
        )
    return input_model


def get_response_model(handler):
    response_model = handler.__annotations__.get("return")
    if not response_model:
        raise ValueError(
            f"Handler {handler.__name__} must have a return type annotation"
        )
    # if not issubclass(response_model, BaseModel):
    #     raise ValueError(f"Handler {handler.__name__} return type must be a deploymodel BaseModel. got {response_model}")
    return response_model


# ---------------------------------------------------------------------------- #
#                                  API Worker                                  #
# ---------------------------------------------------------------------------- #
class WorkerAPI:
    """Used to launch the FastAPI web server when the worker is running in API mode."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the WorkerAPI class.
        1. Starts the heartbeat thread.
        2. Initializes the FastAPI web server.
        3. Sets the handler for processing jobs.
        """
        # Start the heartbeat thread.
        heartbeat.start_ping()

        self.config = config

        root_path = os.environ.get("DEPLOYMODEL_API_ROOT")
        if not root_path:
            raise ValueError("DEPLOYMODEL_API_ROOT environment variable is not set")
        # Initialize the FastAPI web server.

        model_title = os.environ.get(
            "DEPLOYMODEL_MODEL_TITLE", "<DEPLOYMODEL_MODEL_TITLE>"
        )
        self.rp_app = FastAPI(
            title=f"{model_title} | DeployModel",
            description=DESCRIPTION,
            version=runpod_version,
            docs_url="/",
            root_path=root_path,
        )

        # Create an APIRouter and add the route for processing jobs.
        api_router = APIRouter()

        # Docs Redirect /docs -> /
        api_router.add_api_route(
            "/docs", lambda: RedirectResponse(url="/"), include_in_schema=False
        )

        if RUNPOD_ENDPOINT_ID:
            api_router.add_api_route(
                f"/{RUNPOD_ENDPOINT_ID}/realtime", self._realtime, methods=["POST"]
            )

        handler = self.config["handler"]
        input_model = get_input_model(handler)
        input_model = dataclass(input_model)

        def handler_wrapper(job):
            return handler(job["input"])

        from pydantic import Field

        self.config["handler"] = handler_wrapper
        response_model = get_response_model(handler)
        response_model = pydantic.create_model(
            "ResponseModel", output=(response_model, Field(...)), __base__=BaseResponse
        )

        async def _sim_run_wrapper(input: input_model = Depends()):
            job_request = DefaultRequest(input=input)
            return await self._sim_run(job_request)

        # Simulation endpoints.
        api_router.add_api_route(
            "/run",
            _sim_run_wrapper,
            methods=["POST"],
            response_model_exclude_none=True,
            summary="Run job asynchronously.",
            # description=RUN_DESCRIPTION,
        )

        async def _sim_runsync_wrapper(input: input_model = Depends()):
            job_request = DefaultRequest(input=input)
            return await self._sim_runsync(job_request)

        api_router.add_api_route(
            "/runsync",
            _sim_runsync_wrapper,
            methods=["POST"],
            response_model=response_model,
            response_model_exclude_none=True,
            summary="Run job synchronously.",
            # description=RUNSYNC_DESCRIPTION,
        )

        # async def _sim_stream_wrapper(input: input_model = Depends()):
        #     job_request = DefaultRequest(input=input)
        #     return await self._sim_stream(job_request)

        # api_router.add_api_route(
        #     "/stream/{job_id}",
        #     _sim_stream_wrapper,
        #     methods=["POST"],
        #     response_model_exclude_none=True,
        #     summary="Mimics the behavior of the stream endpoint.",
        #     description=STREAM_DESCRIPTION,
        #     tags=["Check Job Results"],
        # )
        api_router.add_api_route(
            "/status/{job_id}",
            self._sim_status,
            methods=["GET"],
            response_model_exclude_none=True,
            response_model=response_model,
            summary="Checks status of job.",
            description=STATUS_DESCRIPTION,
        )

        # Include the APIRouter in the FastAPI application.
        self.rp_app.include_router(api_router)

    def start_uvicorn(self, api_host="localhost", api_port=8000, api_concurrency=1):
        """
        Starts the Uvicorn server.
        """
        uvicorn.run(
            self.rp_app,
            host=api_host,
            port=int(api_port),
            workers=int(api_concurrency),
            log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
            access_log=False,
        )

    # ----------------------------- Realtime Endpoint ---------------------------- #
    async def _realtime(self, job: Job):
        """
        Performs model inference on the input data using the provided handler.
        If handler is not provided, returns an error message.
        """
        job_list.add_job(job.id)

        # Process the job using the provided handler, passing in the job input.
        job_results = await run_job(self.config["handler"], job.__dict__)

        job_list.remove_job(job.id)

        # Return the results of the job processing.
        return jsonable_encoder(job_results)

    # ---------------------------------------------------------------------------- #
    #                             Simulation Endpoints                             #
    # ---------------------------------------------------------------------------- #

    # ------------------------------------ run ----------------------------------- #
    async def _sim_run(self, job_request: DefaultRequest) -> JobOutput:
        """Development endpoint to simulate run behavior."""
        assigned_job_id = f"test-{uuid.uuid4()}"
        job_list.add_job(assigned_job_id, job_request.input, job_request.webhook)
        return jsonable_encoder({"id": assigned_job_id, "status": "IN_PROGRESS"})

    # ---------------------------------- runsync --------------------------------- #
    async def _sim_runsync(self, job_request: DefaultRequest) -> JobOutput:
        """Development endpoint to simulate runsync behavior."""
        assigned_job_id = f"test-{uuid.uuid4()}"
        job = TestJob(id=assigned_job_id, input=job_request.input)

        if is_generator(self.config["handler"]):
            generator_output = run_job_generator(self.config["handler"], job.__dict__)
            job_output = {"output": []}
            async for stream_output in generator_output:
                job_output["output"].append(stream_output["output"])
        else:
            job_output = await run_job(self.config["handler"], job.__dict__)

        if job_output.get("error", None):
            return jsonable_encoder(
                {"id": job.id, "status": "FAILED", "error": job_output["error"]}
            )

        if job_request.webhook:
            thread = threading.Thread(
                target=_send_webhook,
                args=(job_request.webhook, job_output),
                daemon=True,
            )
            thread.start()

        return jsonable_encoder(
            {"id": job.id, "status": "COMPLETED", "output": job_output["output"]}
        )

    # ---------------------------------- stream ---------------------------------- #
    async def _sim_stream(self, job_id: str) -> StreamOutput:
        """Development endpoint to simulate stream behavior."""
        stashed_job = job_list.get_job(job_id)
        if stashed_job is None:
            return jsonable_encoder(
                {"id": job_id, "status": "FAILED", "error": "Job ID not found"}
            )

        job = TestJob(id=job_id, input=stashed_job.input)

        if is_generator(self.config["handler"]):
            generator_output = run_job_generator(self.config["handler"], job.__dict__)
            stream_accumulator = []
            async for stream_output in generator_output:
                stream_accumulator.append({"output": stream_output["output"]})
        else:
            return jsonable_encoder(
                {
                    "id": job_id,
                    "status": "FAILED",
                    "error": "Stream not supported, handler must be a generator.",
                }
            )

        job_list.remove_job(job.id)

        if stashed_job.webhook:
            thread = threading.Thread(
                target=_send_webhook,
                args=(stashed_job.webhook, stream_accumulator),
                daemon=True,
            )
            thread.start()

        return jsonable_encoder(
            {"id": job_id, "status": "COMPLETED", "stream": stream_accumulator}
        )

    # ---------------------------------- status ---------------------------------- #
    async def _sim_status(self, job_id: str) -> JobOutput:
        """Development endpoint to simulate status behavior."""
        stashed_job = job_list.get_job(job_id)
        if stashed_job is None:
            return jsonable_encoder(
                {"id": job_id, "status": "FAILED", "error": "Job ID not found"}
            )

        job = TestJob(id=stashed_job.id, input=stashed_job.input)

        if is_generator(self.config["handler"]):
            generator_output = run_job_generator(self.config["handler"], job.__dict__)
            job_output = {"output": []}
            async for stream_output in generator_output:
                job_output["output"].append(stream_output["output"])
        else:
            job_output = await run_job(self.config["handler"], job.__dict__)

        job_list.remove_job(job.id)

        if job_output.get("error", None):
            return jsonable_encoder(
                {"id": job_id, "status": "FAILED", "error": job_output["error"]}
            )

        if stashed_job.webhook:
            thread = threading.Thread(
                target=_send_webhook,
                args=(stashed_job.webhook, job_output),
                daemon=True,
            )
            thread.start()

        return jsonable_encoder(
            {"id": job_id, "status": "COMPLETED", "output": job_output["output"]}
        )
