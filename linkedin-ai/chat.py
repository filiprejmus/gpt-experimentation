from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import AgentExecutor
from langchain.callbacks import get_openai_callback
import os, yaml
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
import yaml
from langchain.requests import RequestsWrapper
from langchain.agents.agent_toolkits.openapi import planner


llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()
user_query = "Use the search endpoint find my meeting with Max and summarize it"

with open("notion_api.yaml", "r") as f:
    notion_yaml = yaml.safe_load(f)

notion_spec = reduce_openapi_spec(notion_yaml)


def construct_notion_auth_headers():
    key = os.environ.get("NOTION_API_KEY")
    return {
        "Authorization": f"Bearer {key}",
        "accept": "application/json",
        "Notion-Version": "2022-06-28",
    }


headers = construct_notion_auth_headers()
requests_wrapper = RequestsWrapper(headers=headers)


notion_agent = planner.create_openapi_agent(notion_spec, requests_wrapper, llm)


with get_openai_callback() as cb:
    notion_agent.run(user_query)
    print(cb)
