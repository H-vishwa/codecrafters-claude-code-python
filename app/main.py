import argparse
import os
import sys
import json
from typing import Any, cast
from openai import OpenAI
from .utils import bash, get_tools, read_file, write_file

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    tools = get_tools()
    messages: list[Any] = [{"role": "user", "content": args.p}]

    model = "anthropic/claude-haiku-4.5"

    while True:
        chat = client.chat.completions.create(
            model=model, messages=messages, max_tokens=1000, tools=tools
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        message = chat.choices[0].message

        messages.append(
            {"role": "assistant", "content": None, "tool_calls": message.tool_calls}
        )

        if not message.tool_calls:
            print(message.content)
            break

        for tool_call in message.tool_calls:
            # some runtime types expose the tool name under different attributes; be defensive.
            tool_name = getattr(tool_call.function, "name", None) or getattr(
                tool_call, "name", None
            )
            if tool_name == "Read":
                tool = read_file
            elif tool_name == "Write":
                tool = write_file
            elif tool_name == "Bash":
                tool = bash
            else:
                raise RuntimeError(f"Unknown tool: {tool_name}")
            tool_args = getattr(tool_call.function, "arguments", None) or getattr(
                tool_call, "arguments", None
            )
            # tool_args may be None or not a str; ensure we pass a str to json.loads
            if tool_args is None:
                tool_kwargs = {}
            else:
                # some runtimes may provide arguments as bytes; decode if needed
                if isinstance(tool_args, (bytes, bytearray)):
                    tool_args = tool_args.decode()
                tool_kwargs = json.loads(tool_args)
            result = tool(**tool_kwargs)
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )


if __name__ == "__main__":
    main()
    