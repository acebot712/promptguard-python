"""Every integration names the framework's real Conversation in ``session_id``.

The Guard API remembers a Conversation's turns when a scan names it in
``context.session_id``, so the id on the wire has to be the framework's own
conversation id: the same on every turn of one conversation, different for
another, and absent when the framework has none. A per-call run or event id --
what the LangChain and LlamaIndex handlers used to send -- makes every scan a
conversation of its own.

The Guard API is faked at the HTTP layer (an ``httpx.MockTransport`` behind the
real ``GuardClient``), so each assertion reads the JSON body that would have
been sent.
"""

import asyncio
import importlib.machinery
import json
import sys
import types
import uuid
from typing import Any

import httpx
import pytest

from promptguard.guard import GuardClient

NO_SESSION = object()


class FakeGuardApi:
    """``POST /api/v1/guard``, allowing everything and recording each body."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"decision": "allow", "event_id": "evt"})

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)

    @property
    def session_ids(self) -> list[Any]:
        """The ``context.session_id`` of every scan, ``NO_SESSION`` where absent."""
        return [b.get("context", {}).get("session_id", NO_SESSION) for b in self.bodies]


@pytest.fixture
def guard_api(monkeypatch: pytest.MonkeyPatch) -> FakeGuardApi:
    fake = FakeGuardApi()
    monkeypatch.setenv("PROMPTGUARD_API_KEY", "pg_test_session")

    def _sync(self: GuardClient) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(fake.handler))

    def _async(self: GuardClient) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(fake.async_handler))

    monkeypatch.setattr(GuardClient, "_ensure_sync_client", _sync)
    monkeypatch.setattr(GuardClient, "_ensure_async_client", _async)
    return fake


# ---------------------------------------------------------------------------
# LangChain callback handler
# ---------------------------------------------------------------------------


def _langchain_turn(handler: Any, text: str, metadata: dict[str, Any] | None) -> None:
    """One chat-model call as langchain-core reports it to a callback handler:
    a start with the call's metadata, then an end that carries none."""
    run_id = uuid.uuid4()
    handler.on_chat_model_start(
        {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
        [[{"role": "user", "content": text}]],
        run_id=run_id,
        parent_run_id=uuid.uuid4(),
        metadata=metadata,
    )
    response = types.SimpleNamespace(generations=[[types.SimpleNamespace(text=f"re: {text}")]])
    handler.on_llm_end(response, run_id=run_id, parent_run_id=None)


class TestLangChainHandler:
    def _handler(self, **kwargs: Any) -> Any:
        from promptguard.integrations.langchain import PromptGuardCallbackHandler

        return PromptGuardCallbackHandler(**kwargs)

    def test_two_calls_in_one_thread_send_the_same_session_id(self, guard_api):
        handler = self._handler()
        _langchain_turn(handler, "hello", {"thread_id": "thread-a"})
        _langchain_turn(handler, "and again", {"thread_id": "thread-a"})

        assert guard_api.session_ids == ["thread-a"] * 4

    def test_another_thread_is_another_session(self, guard_api):
        handler = self._handler()
        _langchain_turn(handler, "hello", {"thread_id": "thread-a"})
        _langchain_turn(handler, "hello", {"thread_id": "thread-b"})

        assert guard_api.session_ids == ["thread-a", "thread-a", "thread-b", "thread-b"]

    def test_a_call_with_no_thread_sends_no_session_id(self, guard_api):
        handler = self._handler()
        _langchain_turn(handler, "hello", None)
        _langchain_turn(handler, "hello", {"ls_provider": "openai"})

        assert guard_api.session_ids == [NO_SESSION] * 4

    @pytest.mark.parametrize("key", ["session_id", "conversation_id"])
    def test_the_other_langsmith_thread_keys_name_the_session(self, guard_api, key):
        handler = self._handler()
        _langchain_turn(handler, "hello", {key: "conv-1"})
        _langchain_turn(handler, "again", {key: "conv-1"})

        assert guard_api.session_ids == ["conv-1"] * 4

    def test_an_integer_thread_id_is_sent_as_a_string(self, guard_api):
        handler = self._handler()
        _langchain_turn(handler, "hello", {"thread_id": 42})

        assert guard_api.session_ids == ["42", "42"]

    def test_an_explicit_session_id_applies_when_the_call_names_none(self, guard_api):
        handler = self._handler(session_id="user-chat-7")
        _langchain_turn(handler, "hello", None)
        _langchain_turn(handler, "hello", {"thread_id": "thread-a"})

        # The call's own thread is more specific than the handler's default.
        assert guard_api.session_ids == ["user-chat-7"] * 2 + ["thread-a"] * 2

    def test_an_empty_explicit_session_id_is_refused(self, guard_api):
        with pytest.raises(ValueError, match="session_id"):
            self._handler(session_id="  ")

    def test_a_tool_call_carries_its_thread(self, guard_api):
        handler = self._handler()
        run_id = uuid.uuid4()
        handler.on_tool_start(
            {"name": "search"}, "query", run_id=run_id, metadata={"thread_id": "thread-a"}
        )
        handler.on_tool_end("result", run_id=run_id)

        assert guard_api.session_ids == ["thread-a", "thread-a"]

    def test_the_run_is_forgotten_once_it_ends(self, guard_api):
        handler = self._handler()
        _langchain_turn(handler, "hello", {"thread_id": "thread-a"})

        assert handler._run_sessions == {}


class TestLangChainRunnable:
    def _guard(self, **kwargs: Any) -> Any:
        from promptguard.integrations.langchain import PromptGuardRunnable

        return PromptGuardRunnable(**kwargs)

    def test_the_configured_thread_names_the_session(self, guard_api):
        guard = self._guard()
        config = {"configurable": {"thread_id": "thread-a"}}
        guard.guard_value("hello", config=config)
        guard.guard_value("again", config=config)
        guard.guard_value("elsewhere", config={"metadata": {"session_id": "thread-b"}})

        assert guard_api.session_ids == ["thread-a", "thread-a", "thread-b"]

    def test_no_thread_sends_no_session_id(self, guard_api):
        guard = self._guard()
        guard.guard_value("hello")
        guard.guard_value("hello", config={"configurable": {}})

        assert guard_api.session_ids == [NO_SESSION, NO_SESSION]

    def test_an_explicit_session_id_applies_when_the_call_names_none(self, guard_api):
        guard = self._guard(session_id="user-chat-7")
        guard.guard_value("hello")

        assert guard_api.session_ids == ["user-chat-7"]

    def test_as_a_runnable_it_reads_the_invoke_config(self, guard_api):
        pytest.importorskip("langchain_core")
        runnable = self._guard().as_runnable()
        runnable.invoke("hello", config={"configurable": {"thread_id": "thread-a"}})
        runnable.invoke("again", config={"configurable": {"thread_id": "thread-a"}})
        runnable.invoke("other", config={"configurable": {"thread_id": "thread-b"}})

        assert guard_api.session_ids == ["thread-a", "thread-a", "thread-b"]


class TestUnderRealLangChain:
    """The handler attached to a real LangGraph graph and a real chat model."""

    @pytest.fixture(autouse=True)
    def _needs_langchain(self):
        callbacks = pytest.importorskip("langchain_core.callbacks")
        pytest.importorskip("langgraph")
        from promptguard.integrations.langchain import PromptGuardCallbackHandler

        if not issubclass(PromptGuardCallbackHandler, callbacks.BaseCallbackHandler):
            pytest.skip(
                "the handler runs under langchain-core only once it subclasses "
                "BaseCallbackHandler (fix/langchain-handler-runs-under-langchain-core)"
            )

    def _graph(self) -> Any:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import START, MessagesState, StateGraph

        model = FakeListChatModel(responses=["one", "two", "three", "four"])

        def respond(state: MessagesState) -> dict[str, Any]:
            return {"messages": [model.invoke(state["messages"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("respond", respond)
        graph.add_edge(START, "respond")
        return graph.compile(checkpointer=InMemorySaver())

    def test_a_langgraph_thread_is_one_session(self, guard_api):
        from promptguard.integrations.langchain import PromptGuardCallbackHandler

        handler = PromptGuardCallbackHandler(scan_responses=False)
        app = self._graph()

        def turn(text: str, thread: str) -> None:
            app.invoke(
                {"messages": [("user", text)]},
                config={"callbacks": [handler], "configurable": {"thread_id": thread}},
            )

        turn("hello", "thread-a")
        turn("and again", "thread-a")
        turn("hello", "thread-b")

        assert guard_api.session_ids == ["thread-a", "thread-a", "thread-b"]

    def test_a_plain_chat_model_call_names_its_session_in_metadata(self, guard_api):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        from promptguard.integrations.langchain import PromptGuardCallbackHandler

        handler = PromptGuardCallbackHandler(scan_responses=False)
        model = FakeListChatModel(responses=["one", "two", "three"])
        model.invoke("hi", config={"callbacks": [handler], "metadata": {"thread_id": "t-1"}})
        model.invoke("hi", config={"callbacks": [handler], "metadata": {"thread_id": "t-1"}})
        model.invoke("hi", config={"callbacks": [handler]})

        assert guard_api.session_ids == ["t-1", "t-1", NO_SESSION]


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


def _llamaindex_turn(handler: Any, text: str) -> None:
    """One LLM event as LlamaIndex's CallbackManager reports it: a fresh event
    id per event, and nothing in the payload that names a conversation."""
    event_id = handler.on_event_start(
        "llm", {"messages": [{"role": "user", "content": text}]}, event_id=str(uuid.uuid4())
    )
    handler.on_event_end("llm", {"response": f"re: {text}"}, event_id=event_id)


class TestLlamaIndex:
    def test_the_event_id_is_never_sent_as_a_session(self, guard_api):
        from promptguard.integrations.llamaindex import PromptGuardCallbackHandler

        handler = PromptGuardCallbackHandler()
        _llamaindex_turn(handler, "hello")
        _llamaindex_turn(handler, "again")

        assert guard_api.session_ids == [NO_SESSION] * 4

    def test_an_explicit_session_id_names_every_turn(self, guard_api):
        from promptguard.integrations.llamaindex import PromptGuardCallbackHandler

        chat_a = PromptGuardCallbackHandler(session_id="chat-a")
        chat_b = PromptGuardCallbackHandler(session_id="chat-b")
        _llamaindex_turn(chat_a, "hello")
        _llamaindex_turn(chat_a, "again")
        _llamaindex_turn(chat_b, "hello")

        assert guard_api.session_ids == ["chat-a"] * 4 + ["chat-b"] * 2

    def test_a_query_event_carries_the_session(self, guard_api):
        from promptguard.integrations.llamaindex import PromptGuardCallbackHandler

        handler = PromptGuardCallbackHandler(session_id="chat-a")
        event_id = handler.on_event_start("query", {"query_str": "q"}, event_id="e1")
        handler.on_event_end("query", {"response": "r"}, event_id=event_id)

        assert guard_api.session_ids == ["chat-a", "chat-a"]

    def test_the_query_guard_takes_an_explicit_session(self, guard_api):
        from promptguard.integrations.llamaindex import PromptGuardQueryGuard

        PromptGuardQueryGuard().guard_query("hello")
        PromptGuardQueryGuard(session_id="chat-a").guard_query("hello")

        assert guard_api.session_ids == [NO_SESSION, "chat-a"]

    def test_an_empty_explicit_session_id_is_refused(self, guard_api):
        from promptguard.integrations.llamaindex import (
            PromptGuardCallbackHandler,
            PromptGuardQueryGuard,
        )

        with pytest.raises(ValueError, match="session_id"):
            PromptGuardCallbackHandler(session_id="")
        with pytest.raises(ValueError, match="session_id"):
            PromptGuardQueryGuard(session_id="")


# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------


class TestCrewAI:
    def test_a_crew_sends_no_session_by_default(self, guard_api):
        from promptguard.integrations.crewai import PromptGuardGuardrail

        guardrail = PromptGuardGuardrail()
        guardrail.before_kickoff({"topic": "hello"})
        guardrail.after_kickoff("result")

        assert guard_api.session_ids == [NO_SESSION, NO_SESSION]

    def test_an_explicit_session_id_names_every_hook(self, guard_api):
        from promptguard.integrations.crewai import PromptGuardGuardrail

        guardrail = PromptGuardGuardrail(session_id="crew-chat-1")
        guardrail.before_kickoff({"topic": "hello"})
        guardrail.scan_task_output("draft", task_name="write")
        guardrail.after_kickoff("result")
        PromptGuardGuardrail(session_id="crew-chat-2").before_kickoff({"topic": "hi"})

        assert guard_api.session_ids == ["crew-chat-1"] * 3 + ["crew-chat-2"]


# ---------------------------------------------------------------------------
# ASGI / FastAPI request-edge middleware
# ---------------------------------------------------------------------------


class TestFastAPIMiddleware:
    def _post(self, header: str | None, headers: list[dict[str, str]]) -> None:
        pytest.importorskip("fastapi")
        from fastapi import FastAPI

        from promptguard.integrations.fastapi import PromptGuardMiddleware

        app = FastAPI()

        @app.post("/chat")
        async def chat() -> dict[str, str]:
            return {"ok": "yes"}

        app.add_middleware(PromptGuardMiddleware, session_id_header=header)

        async def run() -> None:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
                for h in headers:
                    response = await client.post("/chat", json={"prompt": "hello"}, headers=h)
                    assert response.status_code == 200, response.text

        asyncio.run(run())

    def test_the_named_header_is_the_session(self, guard_api):
        self._post(
            "X-Conversation-Id",
            [
                {"X-Conversation-Id": "conv-a"},
                {"X-Conversation-Id": "conv-a"},
                {"X-Conversation-Id": "conv-b"},
                {},
            ],
        )

        assert guard_api.session_ids == ["conv-a", "conv-a", "conv-b", NO_SESSION]

    def test_without_a_named_header_no_session_is_sent(self, guard_api):
        self._post(None, [{"X-Conversation-Id": "conv-a"}])

        assert guard_api.session_ids == [NO_SESSION]


# ---------------------------------------------------------------------------
# Auto-instrumentation: the OpenAI Responses API's own conversation
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_openai_responses(monkeypatch: pytest.MonkeyPatch):
    """A minimal ``openai`` module tree with ``responses.create``."""

    def make_module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        return module

    openai_mod = make_module("openai")
    resources = make_module("openai.resources")
    responses_mod = make_module("openai.resources.responses")

    class Responses:
        def create(self, **kwargs: Any) -> dict[str, Any]:
            return {"output": []}

    class AsyncResponses:
        async def create(self, **kwargs: Any) -> dict[str, Any]:
            return {"output": []}

    responses_mod.Responses = Responses
    responses_mod.AsyncResponses = AsyncResponses
    openai_mod.resources = resources
    resources.responses = responses_mod
    for module in (openai_mod, resources, responses_mod):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    import promptguard.auto as auto
    import promptguard.patches.openai_patch as openai_patch

    monkeypatch.setattr(auto, "_guard_client", GuardClient(api_key="pg_test_session"))
    monkeypatch.setattr(auto, "_mode", "enforce")
    monkeypatch.setattr(auto, "_fail_open", True)
    monkeypatch.setattr(auto, "_scan_responses", False)
    # The fake classes are thrown away with the test, so a fresh record of
    # originals is all the clean-up there is (revert() acts only after apply()).
    monkeypatch.setattr(openai_patch, "_originals", {})
    assert openai_patch._apply_responses_patch() is True
    return Responses


class TestOpenAIResponsesConversation:
    def test_the_conversation_names_the_session(self, guard_api, fake_openai_responses):
        responses = fake_openai_responses()
        responses.create(model="gpt-5-nano", input="hello", conversation="conv_a")
        responses.create(model="gpt-5-nano", input="again", conversation={"id": "conv_a"})
        responses.create(model="gpt-5-nano", input="hello", conversation="conv_b")

        assert guard_api.session_ids == ["conv_a", "conv_a", "conv_b"]

    def test_no_conversation_sends_no_session(self, guard_api, fake_openai_responses):
        responses = fake_openai_responses()
        # previous_response_id changes every turn, so it names no conversation.
        responses.create(model="gpt-5-nano", input="hello", previous_response_id="resp_1")

        assert guard_api.session_ids == [NO_SESSION]
