"""Industrial pipeline demo UI for workflow diagnostics and Copilot reasoning."""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import socket
import uuid
import urllib.parse
import urllib.request
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)
BACKEND_BASE_URL = os.getenv("PIPELINE_API_URL", "http://127.0.0.1:8001")


def _api_get_json(path: str) -> Any:
    try:
        url = f"{BACKEND_BASE_URL}{path}"
        req = urllib.request.Request(url, headers={"User-Agent": "IndustrialPipelineDemo/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("API GET %s failed: %s", path, exc, exc_info=True)
        return {"error": str(exc)}


def _api_post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        url = f"{BACKEND_BASE_URL}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "IndustrialPipelineDemo/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=40) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("API POST %s failed: %s", path, exc, exc_info=True)
        return {"error": str(exc)}


def _api_post_form(path: str, form_data: dict[str, str]) -> dict[str, Any]:
    try:
        url = f"{BACKEND_BASE_URL}{path}"
        data = urllib.parse.urlencode(form_data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "IndustrialPipelineDemo/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("API FORM POST %s failed: %s", path, exc, exc_info=True)
        return {"error": str(exc)}


def format_json(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, default=str)
    except Exception:
        return str(value)


def run_text_workflow(text_value: str) -> str:
    if not text_value or not text_value.strip():
        return "Enter plain text to submit to the workflow."

    result = _api_post_form(
        "/api/v1/workflows/analyze",
        {
            "text": text_value,
            "file_name": "workflow-text.txt",
            "mime_type": "text/plain",
        },
    )
    if isinstance(result, dict) and result.get("error"):
        return f"Workflow submission failed: {result['error']}"
    return format_json(result)


def copilot_chat(question: str, use_web: bool, job_id: str) -> tuple[str, str, str, str]:
    if not question or not question.strip():
        return (
            "Enter a question for the industrial Copilot.",
            "",
            "",
            "",
        )

    payload = {
        "question": question.strip(),
        "job_id": job_id.strip() or None,
        "use_web": bool(use_web),
    }
    result = _api_post_json("/api/v1/copilot/chat", payload)
    if isinstance(result, dict) and result.get("error"):
        error_text = f"Copilot API error: {result['error']}"
        return error_text, "", "", error_text

    answer = result.get("answer") or result.get("summary") or "No answer returned."
    reasoning = format_json(result.get("reasoning", {}))
    references = format_json(result.get("references", []))
    tool_output = (
        f"Agent: {result.get('agent', 'unknown')}\n"
        f"Web lookup: {result.get('used_web_search', False)}\n"
        f"Confidence: {result.get('confidence', 'n/a')}"
    )
    return answer, reasoning, references, tool_output


def copilot_tool_action(job_id: str, tool_name: str) -> str:
    if not job_id or not job_id.strip():
        return "Enter a job id to run Copilot tools."

    endpoint = {
        "RCA": "rca",
        "Maintenance": "maintenance",
        "Compliance": "compliance",
        "Risk": "risk",
    }.get(tool_name, "rca")

    result = _api_get_json(f"/api/v1/copilot/{endpoint}/{job_id.strip()}")
    if isinstance(result, dict) and result.get("error"):
        return f"Copilot tool error: {result['error']}"
    return format_json(result)


def refresh_job_list() -> tuple[dict[str, Any], str]:
    result = _api_get_json("/api/v1/jobs")
    if isinstance(result, dict) and result.get("error"):
        return gr.update(choices=[]), f"Job list failed: {result['error']}"

    job_ids = [job.get("job_id") for job in (result or []) if isinstance(job, dict) and job.get("job_id")]
    return (
        gr.update(choices=job_ids, value=job_ids[0] if job_ids else None),
        f"Loaded {len(job_ids)} saved jobs.",
    )


def load_job_bundle(job_id: str) -> tuple[str, str]:
    if not job_id:
        return "Select a saved job to view details.", ""

    result = _api_get_json(f"/api/v1/workflows/{job_id.strip()}/bundle?include_raw=true")
    if isinstance(result, dict) and result.get("error"):
        return f"Job load failed: {result['error']}", ""
    return f"Loaded bundle for job {job_id}", format_json(result)


def _api_post_multipart(path: str, fields: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    boundary = uuid.uuid4().hex
    body_parts: list[bytes] = []

    def _add_field(name: str, value: Any) -> None:
        body_parts.append(f"--{boundary}".encode("utf-8"))
        body_parts.append(f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"))
        body_parts.append(b"")
        body_parts.append(str(value).encode("utf-8"))

    for name, value in fields.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        _add_field(name, value)

    for name, filepath in files.items():
        if not filepath:
            continue
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        filename = os.path.basename(filepath)
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body_parts.append(f"--{boundary}".encode("utf-8"))
        body_parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode("utf-8")
        )
        body_parts.append(f"Content-Type: {content_type}".encode("utf-8"))
        body_parts.append(b"")
        with open(filepath, "rb") as file_obj:
            body_parts.append(file_obj.read())

    body_parts.append(f"--{boundary}--".encode("utf-8"))
    body_parts.append(b"")
    body = b"\r\n".join(body_parts)

    request = urllib.request.Request(
        f"{BACKEND_BASE_URL}{path}",
        data=body,
        headers={
            "User-Agent": "IndustrialPipelineDemo/1.0",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def submit_source(
    source_file: Any,
    source_text: str,
    source_name: str,
    source_type: str,
    source_format: str,
    parsed_source: bool,
    source_metadata: str,
) -> tuple[str, str, str]:
    if not source_file and not source_text:
        return "Provide a source file or parsed source text to ingest.", "", ""

    result = None
    if source_file:
        filepath = None
        if isinstance(source_file, str):
            filepath = source_file
        elif isinstance(source_file, dict):
            filepath = source_file.get("tmp_path") or source_file.get("name")
        elif isinstance(source_file, (list, tuple)) and source_file:
            first = source_file[0]
            if isinstance(first, str):
                filepath = first
            elif isinstance(first, dict):
                filepath = first.get("tmp_path") or first.get("name")
            else:
                filepath = getattr(first, "tmp_path", None) or getattr(first, "name", None)
        else:
            filepath = getattr(source_file, "tmp_path", None) or getattr(source_file, "name", None)

        if not filepath or not os.path.exists(filepath):
            return "Uploaded file not found.", "", ""

        actual_name = source_name or os.path.basename(filepath)
        guessed_mime = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
        result = _api_post_multipart(
            "/api/v1/sources/ingest",
            {
                "file_name": actual_name,
                "mime_type": source_format or guessed_mime,
                "parsed_source": json.dumps(bool(parsed_source)),
                "source_name": actual_name,
                "source_type": source_type or "local",
                "source_format": source_format or guessed_mime,
                "source_metadata": source_metadata or "",
            },
            {"file": filepath},
        )
    else:
        actual_name = source_name or "parsed-source.txt"
        result = _api_post_form(
            "/api/v1/sources/ingest",
            {
                "text": source_text,
                "file_name": actual_name,
                "mime_type": source_format or "text/plain",
                "parsed_source": json.dumps(bool(parsed_source)),
                "source_name": actual_name,
                "source_type": source_type or "text",
                "source_format": source_format or "text",
                "source_metadata": source_metadata or "",
            },
        )

    if isinstance(result, dict) and result.get("error"):
        return f"Source ingest failed: {result['error']}", "", format_json(result)

    job_id = result.get("job_id", "")
    message = result.get("message", "Source ingested and queued.")
    return f"{message}", job_id, format_json(result)


def refresh_source_progress(job_id: str) -> tuple[str, str]:
    if not job_id or not job_id.strip():
        return "No active job to poll.", ""
    return poll_job_progress(job_id)


def poll_job_progress(job_id: str) -> tuple[str, str]:
    if not job_id or not job_id.strip():
        return "No active job.", ""
    result = _api_get_json(f"/api/v1/jobs/{job_id}/progress")
    if isinstance(result, dict) and result.get("error"):
        return f"Progress lookup failed: {result['error']}", ""

    status_line = f"{result.get('status', 'unknown').upper()} - {result.get('message', '')}"
    stage_lines: list[str] = []
    for stage in result.get("stage_status", []):
        stage_lines.append(
            f"[{stage.get('timestamp','')}] {stage.get('stage','')} -> {stage.get('status','')}: {stage.get('message','')}"
        )
    for output in result.get("stage_outputs", []):
        stage_lines.append(
            f"[{output.get('timestamp','')}] OUTPUT {output.get('stage','')} -> {output.get('status','')}: {output.get('message','')}"
        )

    if not stage_lines:
        stage_lines.append("No stage progress has been emitted yet.")

    return status_line, "\n".join(stage_lines)


def refresh_runtime_status() -> tuple[str, str, str]:
    catalog = _api_get_json("/api/v1/workflows/catalog")
    status = _api_get_json("/api/v1/models/status")
    if isinstance(catalog, dict) and catalog.get("error"):
        return "", "", f"Runtime catalog failed: {catalog['error']}"
    if isinstance(status, dict) and status.get("error"):
        return "", "", f"Model status failed: {status['error']}"
    return format_json(catalog), format_json(status), "Runtime status refreshed successfully."


def launch_app() -> None:
    """Launch the industrial pipeline Gradio demo UI."""

    custom_css = """
    .main-header {
        background: linear-gradient(135deg, #081c2c 0%, #0f4c72 55%, #071c2f 100%);
        border-radius: 18px; padding: 24px; margin-bottom: 22px;
        text-align: center; color: #f8fafc !important; box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    }
    .main-header h1 { font-size: 2.4em; margin: 0; font-weight: 700; }
    .main-header p { opacity: 0.88; margin-top: 10px; font-size: 1.05em; }
    .section-card { border-radius: 16px; border: 1px solid rgba(255,255,255,0.07); background: rgba(255,255,255,0.04); padding: 18px; margin-bottom: 14px; }
    .footer { text-align: center; padding: 12px; margin-top: 18px; background: #f8fafc; border-radius: 12px; color: #1f2937; font-size: 0.95em; }
    .button-primary { background: linear-gradient(135deg, #0f766e, #14b8a6) !important; color: white !important; }
    """

    with gr.Blocks(
        title="Industrial Pipeline Copilot Demo",
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="teal", neutral_hue="slate"),
    ) as demo:
        gr.HTML(
            """
            <div class=\"main-header\">
                <h1>Industrial Pipeline Copilot Demo</h1>
                <p>Inspect workflow jobs, run text analysis, and query the Copilot for root-cause, maintenance, and compliance guidance.</p>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("🚀 Workflow"):
                gr.Markdown("Use backend workflow analysis to process plain text, incident notes, or document summaries.")
                workflow_text = gr.Textbox(
                    lines=10,
                    label="Workflow input text",
                    placeholder="Paste document content, incident notes, or workflow summary...",
                )
                workflow_submit_btn = gr.Button("Run workflow", variant="primary", elem_classes=["button-primary"])
                workflow_response = gr.Textbox(lines=10, label="Workflow response", interactive=False)

            with gr.Tab("📥 Source Ingestor"):
                gr.Markdown(
                    "Upload a local source file or paste parsed content directly. The backend will ingest it into the industrial pipeline and show realtime stage logs. Supports a broad range of source formats via LlamaParse-style extraction, including documents, tables, images, and text-first sources."
                )
                with gr.Row():
                    source_file = gr.File(label="Local source file", file_count="single")
                    source_text = gr.Textbox(
                        lines=8,
                        label="Parsed source text",
                        placeholder="Paste already-parsed source text here (if you have content already extracted)",
                    )
                with gr.Row():
                    source_name = gr.Textbox(label="Source name", placeholder="Optional source name or filename")
                    source_type = gr.Textbox(label="Source type", placeholder="e.g. local, url, database", value="local")
                with gr.Row():
                    source_format = gr.Textbox(label="Source format", placeholder="e.g. text, html, csv, pdf", value="text")
                    parsed_source = gr.Checkbox(label="Parsed source content", value=True)
                source_metadata = gr.Textbox(
                    lines=3,
                    label="Source metadata (JSON)",
                    placeholder='Optional JSON metadata like {"origin":"local","source_id":"ABC123"}',
                )
                source_submit_btn = gr.Button("Ingest source", variant="primary", elem_classes=["button-primary"])
                source_message = gr.Markdown("Ready to ingest a source. Click ingest, then refresh progress to view backend logs.")
                source_job_id = gr.Textbox(label="Ingest job ID", interactive=False)
                source_job_details = gr.Textbox(lines=6, label="Ingest response", interactive=False)
                source_progress_status = gr.Markdown("No backend progress yet. Refresh to poll latest stage logs.")
                source_progress_log = gr.Textbox(lines=10, label="Backend progress log", interactive=False)
                source_refresh_btn = gr.Button("Refresh progress", size="sm")

            with gr.Tab("🤖 Copilot"):
                gr.Markdown("Ask the Copilot to explain a job, diagnose a failure, or generate maintenance recommendations.")
                copilot_question = gr.Textbox(
                    lines=2,
                    label="Copilot question",
                    placeholder="Example: Why did stage 18 fail for job 12345?",
                )
                use_web_search = gr.Checkbox(value=True, label="Allow web-backed reasoning")
                copilot_job_id = gr.Textbox(lines=1, label="Job ID (optional)", placeholder="Enter job ID to ground the answer")
                copilot_send_btn = gr.Button("Ask Copilot", variant="primary", elem_classes=["button-primary"])
                copilot_answer = gr.Textbox(lines=6, label="Copilot answer", interactive=False)
                with gr.Accordion("Copilot evidence and tool actions", open=False):
                    copilot_reasoning = gr.Textbox(lines=5, label="Reasoning / summary", interactive=False)
                    copilot_references = gr.Textbox(lines=5, label="Evidence & references", interactive=False)
                    with gr.Row():
                        rca_btn = gr.Button("RCA", size="sm")
                        maintenance_btn = gr.Button("Maintenance", size="sm")
                        compliance_btn = gr.Button("Compliance", size="sm")
                        risk_btn = gr.Button("Risk", size="sm")
                    copilot_tool_output = gr.Textbox(lines=5, label="Copilot tool output", interactive=False)

            with gr.Tab("📦 Job Dashboard"):
                gr.Markdown("Browse saved workflow jobs and inspect backend bundles.")
                with gr.Row():
                    job_selector = gr.Dropdown(choices=[], label="Select saved job")
                    refresh_jobs_btn = gr.Button("Refresh jobs", size="sm")
                job_summary = gr.Markdown("No jobs loaded yet.")
                job_bundle_output = gr.Textbox(lines=16, label="Job bundle details", interactive=False)

            with gr.Tab("📊 Runtime"):
                gr.Markdown("View available workflow catalog and model/runtime health.")
                refresh_status_btn = gr.Button("Refresh runtime status", variant="secondary")
                workflow_catalog = gr.Textbox(lines=10, label="Workflow catalog", interactive=False)
                model_status = gr.Textbox(lines=10, label="Model status", interactive=False)
                runtime_status = gr.Markdown("Press refresh to load runtime metadata and health.")

        gr.HTML(
            f"""
            <div class=\"footer\">
                Industrial Pipeline AI • Backend: <code>{BACKEND_BASE_URL}</code>
            </div>
            """
        )

        workflow_submit_btn.click(
            fn=run_text_workflow,
            inputs=[workflow_text],
            outputs=[workflow_response],
            show_progress="full",
        )

        copilot_send_btn.click(
            fn=copilot_chat,
            inputs=[copilot_question, use_web_search, copilot_job_id],
            outputs=[copilot_answer, copilot_reasoning, copilot_references, copilot_tool_output],
            show_progress="full",
        )

        rca_btn.click(
            fn=lambda job_id: copilot_tool_action(job_id, "RCA"),
            inputs=[copilot_job_id],
            outputs=[copilot_tool_output],
        )
        maintenance_btn.click(
            fn=lambda job_id: copilot_tool_action(job_id, "Maintenance"),
            inputs=[copilot_job_id],
            outputs=[copilot_tool_output],
        )
        compliance_btn.click(
            fn=lambda job_id: copilot_tool_action(job_id, "Compliance"),
            inputs=[copilot_job_id],
            outputs=[copilot_tool_output],
        )
        risk_btn.click(
            fn=lambda job_id: copilot_tool_action(job_id, "Risk"),
            inputs=[copilot_job_id],
            outputs=[copilot_tool_output],
        )

        refresh_jobs_btn.click(
            fn=refresh_job_list,
            inputs=None,
            outputs=[job_selector, job_summary],
        )

        job_selector.change(
            fn=load_job_bundle,
            inputs=[job_selector],
            outputs=[job_summary, job_bundle_output],
        )

        source_submit_btn.click(
            fn=submit_source,
            inputs=[
                source_file,
                source_text,
                source_name,
                source_type,
                source_format,
                parsed_source,
                source_metadata,
            ],
            outputs=[source_message, source_job_id, source_job_details],
        )

        source_refresh_btn.click(
            fn=refresh_source_progress,
            inputs=[source_job_id],
            outputs=[source_progress_status, source_progress_log],
        )

        refresh_status_btn.click(
            fn=refresh_runtime_status,
            inputs=None,
            outputs=[workflow_catalog, model_status, runtime_status],
        )

    default_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    port = default_port
    if os.getenv("GRADIO_SERVER_PORT") is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if test_socket.connect_ex(("127.0.0.1", default_port)) == 0:
                logger.warning("Port %d is in use, falling back to port %d", default_port, default_port + 1)
                port = default_port + 1

    launch_url = f"http://127.0.0.1:{port}"
    logger.info("Launching Industrial Pipeline demo UI on %s", launch_url)
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    launch_app()
