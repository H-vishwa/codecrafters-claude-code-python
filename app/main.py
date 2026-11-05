import argparse
import os
import re
import sys
import json
from typing import Any
from openai import OpenAI
from .utils import bash, get_tools, read_file, write_file

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def load_skill(skill_name: str) -> tuple[str, str, str] | None:
    """Return a skill's name, description, and instruction body."""
    skill_path = os.path.join(".claude", "skills", skill_name, "SKILL.md")
    if not os.path.isfile(skill_path):
        return None

    with open(skill_path, encoding="utf-8") as skill_file:
        content = skill_file.read()

    # Skill files start with YAML frontmatter. The body begins after its closing
    # delimiter; split only twice so delimiters inside the body are preserved.
    parts = content.split("---", 2)
    if len(parts) == 3 and parts[0].strip() == "":
        frontmatter = parts[1]
        body = parts[2].strip()
    else:
        frontmatter = ""
        body = content.strip()

    description = ""
    for line in frontmatter.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "description":
            description = value.strip().strip("\"'")
            break
    return skill_name, description, body


def load_invoked_skills(prompt: str) -> tuple[list[str], str] | None:
    """Load consecutive known skill commands and return their shared arguments."""
    if not prompt.startswith("/"):
        return None

    tokens = prompt.split()
    skill_names: list[str] = []
    for token in tokens:
        if not token.startswith("/") or len(token) == 1:
            break
        skill_name = token[1:]
        if not os.path.isfile(
            os.path.join(".claude", "skills", skill_name, "SKILL.md")
        ):
            break
        skill_names.append(skill_name)

    if not skill_names:
        return None

    arguments = " ".join(tokens[len(skill_names) :])
    return skill_names, arguments


def substitute_skill_arguments(body: str, arguments: str) -> str:
    """Substitute full and positional skill argument placeholders."""
    argument_list = arguments.split()
    body = body.replace("$ARGUMENTS", arguments)

    def replace_positional(match: re.Match[str]) -> str:
        index = int(match.group(1) or match.group(2))
        return argument_list[index] if index < len(argument_list) else ""

    return re.sub(r"\$ARGUMENTS\[(\d+)\]|\$(\d+)\b", replace_positional, body)


def discover_skills() -> list[tuple[str, str]]:
    """Read skill names and descriptions without loading their instruction bodies."""
    skills_dir = os.path.join(".claude", "skills")
    if not os.path.isdir(skills_dir):
        return []

    skills: list[tuple[str, str]] = []
    for skill_name in sorted(os.listdir(skills_dir)):
        skill_path = os.path.join(skills_dir, skill_name, "SKILL.md")
        if not os.path.isfile(skill_path):
            continue

        with open(skill_path, encoding="utf-8") as skill_file:
            content = skill_file.read()
        parts = content.split("---", 2)
        if len(parts) != 3 or parts[0].strip():
            continue

        description = ""
        for line in parts[1].splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "description":
                description = value.strip().strip("\"'")
                break
        skills.append((skill_name, description))
    return skills


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    tools = get_tools()
    available_skills = discover_skills()
    invocation = load_invoked_skills(args.p)
    skill_lines = "\n".join(
        f"- {skill_name}: {description}"
        for skill_name, description in available_skills
    )
    system_message = {
        "role": "system",
        "content": (
            "You have access to the following skills:\n\n"
            + skill_lines
            + (
                "\n\nIf a skill matches the user's request, call the Skill tool "
                "with its name and follow the instructions it returns."
                if available_skills
                else ""
            )
        ),
    }
    if invocation:
        skill_names, skill_arguments = invocation
        messages: list[Any] = [
            system_message,
        ]
        for skill_name in skill_names:
            skill = load_skill(skill_name)
            if skill is None:
                continue
            _, _, skill_body = skill
            skill_body = substitute_skill_arguments(skill_body, skill_arguments)
            skill_folder = os.path.join(".claude", "skills", skill_name).replace(
                os.sep, "/"
            )
            skill_instructions = (
                f"Skill: {skill_name} (located at {skill_folder})\n"
                "Paths in the instructions below are relative to that folder.\n\n"
                f"{skill_body}"
            )
            messages.append({"role": "user", "content": skill_instructions})
    else:
        messages = [system_message, {"role": "user", "content": args.p}]

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
            if tool_name == "Skill":
                tool_args = getattr(tool_call.function, "arguments", None) or getattr(
                    tool_call, "arguments", None
                )
                if tool_args is None:
                    tool_kwargs = {}
                else:
                    if isinstance(tool_args, (bytes, bytearray)):
                        tool_args = tool_args.decode()
                    tool_kwargs = json.loads(tool_args)
                requested_name = tool_kwargs.get("name", "")
                if requested_name not in {name for name, _ in available_skills}:
                    result = f"Unknown skill: {requested_name}"
                else:
                    skill = load_skill(requested_name)
                    if skill is None:
                        result = f"Unknown skill: {requested_name}"
                    else:
                        _, _, skill_body = skill
                        skill_arguments = tool_kwargs.get("args", "")
                        skill_body = substitute_skill_arguments(
                            skill_body, skill_arguments
                        )
                        skill_folder = os.path.join(
                            ".claude", "skills", requested_name
                        ).replace(os.sep, "/")
                        result = (
                            f"Skill: {requested_name} (located at {skill_folder})\n"
                            "Paths in the instructions below are relative to that folder.\n\n"
                            f"{skill_body}"
                        )
            elif tool_name == "Read":
                tool = read_file
            elif tool_name == "Write":
                tool = write_file
            elif tool_name == "Bash":
                tool = bash
            elif tool_name not in ("Skill",):
                raise RuntimeError(f"Unknown tool: {tool_name}")
            if tool_name != "Skill":
                tool_args = getattr(tool_call.function, "arguments", None) or getattr(
                    tool_call, "arguments", None
                )
                # tool_args may be None or not a str; ensure we pass a str to json.loads
                if tool_args is None:
                    tool_kwargs = {}
                else:
                    if isinstance(tool_args, (bytes, bytearray)):
                        tool_args = tool_args.decode()
                    tool_kwargs = json.loads(tool_args)
                try:
                    result = tool(**tool_kwargs)
                except (
                    IsADirectoryError,
                    FileNotFoundError,
                    PermissionError,
                    OSError,
                ) as error:
                    result = f"Tool error: {error}"
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )


if __name__ == "__main__":
    main()
    
