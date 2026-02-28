"""
SDK patches for auto-instrumentation.

Each module patches a specific LLM SDK's ``create()`` method so that
PromptGuard scans content before and (optionally) after the LLM call.
"""
