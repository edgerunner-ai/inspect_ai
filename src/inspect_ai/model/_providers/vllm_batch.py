"""vLLM offline batch inference provider.

Uses `vllm run-batch` to process inference requests offline, avoiding the
overhead of running a persistent vLLM server. Designed for two-phase eval
flows where all generation happens first (evaluated model on all GPUs),
then all scoring (judge model on all GPUs).
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any

from typing_extensions import override

from inspect_ai._util.content import ContentText
from inspect_ai._util.local_server import configure_devices
from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    Logprob,
    Logprobs,
    ModelOutput,
    ModelUsage,
    StopReason,
    TopLogprob,
)
from inspect_ai.model._openai import (
    messages_to_openai,
    openai_chat_tools,
    openai_completion_params,
)
from inspect_ai.tool._tool_call import ToolCall
from inspect_ai.tool._tool_choice import ToolChoice
from inspect_ai.tool._tool_info import ToolInfo

from ._vllm_lora import parse_vllm_model

logger = logging.getLogger(__name__)


class _SharedBatchState:
    """Shared batch queue for all VLLMBatchAPI instances with the same model.

    When multiple VLLMBatchAPI instances target the same base_model (e.g.,
    10 task scorers all using the same 70B judge), their requests funnel
    into a single queue and are processed in one ``vllm run-batch`` call
    instead of each instance launching its own.
    """

    def __init__(self) -> None:
        self.pending_requests: list[dict[str, Any]] = []
        self.pending_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.lock = asyncio.Lock()
        self.worker_running = False


_shared_batchers: dict[str, _SharedBatchState] = {}


def _openai_tool_choice(tool_choice: ToolChoice) -> Any:
    if isinstance(tool_choice, str):
        return tool_choice
    if hasattr(tool_choice, "name"):
        return {"type": "function", "function": {"name": tool_choice.name}}
    return tool_choice


class VLLMBatchAPI(ModelAPI):
    """Model provider that uses vLLM offline batch inference.

    Instead of starting a persistent vLLM server, this provider collects
    generation requests and processes them via ``vllm run-batch``, which
    loads the model once, processes all requests, and exits.

    Args:
        model_name: HuggingFace model ID or local path.
        config: Generation configuration.
        **model_args: Additional arguments forwarded to ``vllm run-batch``.
            Notable keys:
            - ``tensor_parallel_size`` (int): Number of GPUs for TP.
            - ``gpu_memory_utilization`` (float): GPU memory fraction.
            - ``max_model_len`` (int): Maximum context length.
            - ``dtype`` (str): Model dtype (e.g. "bfloat16").
            - ``revision`` (str): Model revision/commit hash.
            - ``device`` / ``devices`` (str): GPU devices (CUDA_VISIBLE_DEVICES).
    """

    def __init__(
        self,
        model_name: str,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name=model_name, config=config)

        self.base_model, self.adapter_path, self.adapter_name = parse_vllm_model(
            model_name
        )

        _non_vllm_keys = {"base_url", "api_key", "api_key_vars"}
        self._model_args = {
            k: v for k, v in model_args.items() if k not in _non_vllm_keys and v is not None
        }
        self._batch_send_delay = float(self._model_args.pop("batch_send_delay", 30))

    @override
    def collapse_user_messages(self) -> bool:
        return True

    @override
    def collapse_assistant_messages(self) -> bool:
        return True

    @property
    def _shared(self) -> _SharedBatchState:
        key = self.base_model
        if key not in _shared_batchers:
            _shared_batchers[key] = _SharedBatchState()
        return _shared_batchers[key]

    @override
    def max_connections(self) -> int:
        return 100000

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, "ModelCall"]:
        openai_messages = await messages_to_openai(input)

        params = openai_completion_params(
            model=self.base_model,
            config=config,
            tools=len(tools) > 0,
        )

        if len(tools) > 0:
            params["tools"] = openai_chat_tools(tools)
            params["tool_choice"] = _openai_tool_choice(tool_choice)

        if config.max_tokens is not None and "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")

        request_body: dict[str, Any] = {
            "model": self.base_model,
            "messages": [dict(m) for m in openai_messages],
            **{k: v for k, v in params.items() if k != "model"},
        }

        custom_id = str(uuid.uuid4())
        batch_line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request_body,
        }

        shared = self._shared
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()

        async with shared.lock:
            shared.pending_requests.append(batch_line)
            shared.pending_futures[custom_id] = future

            if not shared.worker_running:
                shared.worker_running = True
                asyncio.ensure_future(self._batch_worker())

        response = await future
        return _parse_batch_response(self.base_model, response)

    async def _batch_worker(self) -> None:
        """Collect requests with a delay, then run vllm run-batch."""
        shared = self._shared
        send_delay = self._batch_send_delay

        while True:
            await asyncio.sleep(send_delay)

            async with shared.lock:
                if not shared.pending_requests:
                    shared.worker_running = False
                    return

                requests = shared.pending_requests.copy()
                futures = shared.pending_futures.copy()
                shared.pending_requests.clear()
                shared.pending_futures.clear()

            try:
                results = await self._run_batch(requests)
            except Exception as e:
                for custom_id, future in futures.items():
                    if not future.done():
                        future.set_exception(e)
                continue

            for custom_id, future in futures.items():
                if custom_id in results:
                    if not future.done():
                        future.set_result(results[custom_id])
                else:
                    if not future.done():
                        future.set_exception(
                            RuntimeError(
                                f"No result for request {custom_id} in vllm run-batch output"
                            )
                        )

            async with shared.lock:
                if not shared.pending_requests:
                    shared.worker_running = False
                    return

    async def _run_batch(
        self, requests: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Write requests to JSONL, run vllm run-batch, parse output."""
        model_args = dict(self._model_args)

        revision = model_args.pop("revision", None)

        model_args, env_vars = configure_devices(
            model_args, parallel_size_param="tensor_parallel_size"
        )

        with (
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, prefix="vllm_batch_in_"
            ) as input_file,
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, prefix="vllm_batch_out_"
            ) as output_file,
        ):
            input_path = input_file.name
            output_path = output_file.name

            for req in requests:
                input_file.write(json.dumps(req) + "\n")

        cmd = [
            "vllm",
            "run-batch",
            "-i",
            input_path,
            "-o",
            output_path,
            "--model",
            self.base_model,
        ]

        if revision:
            cmd.extend(["--revision", str(revision)])

        for key, value in model_args.items():
            if value is None:
                continue
            cli_key = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(cli_key)
            else:
                cmd.extend([cli_key, str(value)])

        env = os.environ.copy()
        env.update(env_vars)

        logger.info(
            f"Running vllm run-batch with {len(requests)} requests: {' '.join(cmd)}"
        )
        start_time = time.monotonic()

        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # inherit parent stderr so vLLM progress is visible
        )

        stdout, _ = await process.communicate()

        elapsed = time.monotonic() - start_time
        logger.info(
            f"vllm run-batch completed in {elapsed:.1f}s "
            f"(exit code: {process.returncode}, requests: {len(requests)})"
        )

        if process.returncode != 0:
            stderr_text = ""
            stdout_text = stdout.decode() if stdout else ""
            raise RuntimeError(
                f"vllm run-batch failed with exit code {process.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {stderr_text}\n"
                f"Stdout: {stdout_text}"
            )

        results: dict[str, dict[str, Any]] = {}
        try:
            with open(output_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")
                    if custom_id:
                        results[custom_id] = result
        finally:
            try:
                os.unlink(input_path)
            except OSError:
                pass
            try:
                os.unlink(output_path)
            except OSError:
                pass

        return results


def _parse_batch_response(model: str, result: dict[str, Any]) -> ModelOutput:
    """Parse a vllm run-batch output line into a ModelOutput."""
    if result.get("error"):
        error = result["error"]
        return ModelOutput.from_content(
            model,
            content=f"Batch error: {error}",
            stop_reason="unknown",
        )

    response = result.get("response", {})
    body = response.get("body", {})

    choices: list[ChatCompletionChoice] = []
    for choice_data in body.get("choices", []):
        message = choice_data.get("message", {})
        content = message.get("content", "") or ""
        role = message.get("role", "assistant")

        stop_reason: StopReason = "stop"
        finish_reason = choice_data.get("finish_reason", "stop")
        if finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_calls"
        elif finish_reason == "content_filter":
            stop_reason = "content_filter"

        tool_calls_data = message.get("tool_calls")
        tool_calls = None
        if tool_calls_data:
            tool_calls = []
            for tc in tool_calls_data:
                func = tc.get("function", {})
                arguments = func.get("arguments", "{}")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", str(uuid.uuid4())),
                        function=func.get("name", ""),
                        arguments=arguments,
                        type="function",
                    )
                )

        logprobs_data = choice_data.get("logprobs")
        logprobs = None
        if logprobs_data and logprobs_data.get("content"):
            logprob_list = []
            for lp in logprobs_data["content"]:
                top_logprobs_list = None
                if lp.get("top_logprobs"):
                    top_logprobs_list = [
                        TopLogprob(
                            token=tlp["token"],
                            logprob=tlp["logprob"],
                            bytes=tlp.get("bytes"),
                        )
                        for tlp in lp["top_logprobs"]
                    ]
                logprob_list.append(
                    Logprob(
                        token=lp["token"],
                        logprob=lp["logprob"],
                        bytes=lp.get("bytes"),
                        top_logprobs=top_logprobs_list,
                    )
                )
            logprobs = Logprobs(content=logprob_list)

        choices.append(
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=[ContentText(text=content)] if content else [],
                    tool_calls=tool_calls,
                ),
                stop_reason=stop_reason,
                logprobs=logprobs,
            )
        )

    usage_data = body.get("usage", {})
    usage = ModelUsage(
        input_tokens=usage_data.get("prompt_tokens", 0),
        output_tokens=usage_data.get("completion_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
    ) if usage_data else None

    return ModelOutput(
        model=body.get("model", model),
        choices=choices,
        usage=usage,
    )
