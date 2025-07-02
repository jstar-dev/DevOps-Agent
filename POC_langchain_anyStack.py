import os
import time
from typing import Any
import requests
from urllib.parse import urlparse
import base64

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI
import httpx
from openai import RateLimitError
import yaml

# Tool 1: Generate azure-pipelines.yml
class YAMLGenerator:
    def __call__(self, tech_stack: str) -> str:
        prompt = f"""
Generate a valid Azure DevOps YAML pipeline for a {tech_stack} project that:
- Runs on push to main
- Installs dependencies
- Runs tests

Use appropriate tools based on the tech stack.
Only output valid YAML (no markdown or extra explanations).
Ensure it contains:
  - 'trigger' block
  - 'pool' block
  - steps to install dependencies and run tests
"""
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-15-preview",
            http_client=httpx.Client(verify=False),
            max_tokens=300
        )

        for i in range(3):
            try:
                yaml_output = llm.invoke(prompt).content
                if yaml_output.startswith("```yaml"):
                    yaml_output = yaml_output.strip("`yaml\n")
                if isinstance(yaml_output, str):
                    parsed = yaml.safe_load(yaml_output)
                    formatted_yaml = yaml.dump(parsed, sort_keys=False, default_flow_style=False)
                    print("\nGenerated YAML:\n", formatted_yaml)
                    return formatted_yaml
            except RateLimitError:
                print(f"Rate limit hit. Retrying ({i+1}/3)...")
                time.sleep(2 ** i)
            except Exception as e:
                print(f"Error parsing YAML: {e}\nRaw output:\n{yaml_output}")
                continue
        raise RuntimeError("Azure OpenAI API quota exceeded after retries.")

# Tool 2: Commit YAML to GitHub
class GitHubCommitter:
    def __call__(self, input_str: str) -> Any:
        repo_url, branch, file_content = input_str.split("::", 2)
        token = os.getenv("GITHUB_TOKEN")
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }

        owner_repo = urlparse(repo_url).path.strip("/")
        file_path = "azure-pipelines.yml"
        url = f"https://api.github.com/repos/{owner_repo}/contents/{file_path}"

        # Step 1: Check if the file already exists to get its SHA
        sha = None
        get_resp = requests.get(url + f"?ref={branch}", headers=headers, verify=False)
        if get_resp.status_code == 200:
            sha = get_resp.json().get("sha")

        # Step 2: Prepare PUT request to create or update file
        data = {
            "message": "Add or update azure-pipelines.yml",
            "content": base64.b64encode(file_content.encode("utf-8")).decode("utf-8"),
            "branch": branch,
        }
        if sha:
            data["sha"] = sha

        response = requests.put(url, headers=headers, json=data, verify=False)
        return response.json()

# Step 3: LangChain Agent Setup
if __name__ == "__main__":
    os.environ["AZURE_OPENAI_API_KEY"] = "4MzeGh8DkcRdFF1t1wdw08JxqvWNHrnmbVdcKw7vlywqydMD7WBMJQQJ99BGACYeBjFXJ3w3AAABACOGdTLH"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://poc-openai-1.openai.azure.com"
    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o-mini"
    os.environ["GITHUB_TOKEN"] = "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"


    tools = [
        Tool(
            name="GeneratePipeline",
            func=YAMLGenerator(),
            description="Generate azure-pipelines.yml for a tech stack"
        ),
        Tool(
            name="CommitToRepo",
            func=GitHubCommitter(),
            description="Commits a file to GitHub. Input should be formatted as '<repo_url>::<branch>::<file_content>'"
        )
    ]

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-02-15-preview",
        http_client=httpx.Client(verify=False),
        temperature=0.3,
        max_tokens=300
    )

    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Example natural language instruction
    user_input = "Generate a pipeline for Python and commit to GitHub repo https://github.com/jstar-dev/Sample on main branch"
    agent_executor.run(user_input)
