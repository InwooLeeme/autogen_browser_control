import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.teams import RoundRobinGroupChat, MagenticOneGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
    
async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key = API_KEY)
    # The web surfer will open a Chromium browser window to perform web browsing tasks.
    web_surfer = MultimodalWebSurfer("web_surfer", model_client, description="""A helpful assistant with access to a web browser.
    Ask them to perform web searches, open pages, and interact with content (e.g., clicking links, scrolling the viewport, filling in form fields, etc.).
    It can also be asked to sleep and wait for pages to load, in cases where the page seems not yet fully loaded.""", headless=False,start_page="https://www.naver.com/",
        browser_channel="chrome", to_save_screenshots=False)
    # The user proxy agent is used to get user input after each step of the web surfer.
    # NOTE: you can skip input by pressing Enter.
    user_proxy = UserProxyAgent("user_proxy")
    # The termination condition is set to end the conversation when the user types 'exit'.
    termination = TextMentionTermination("exit", sources=["user_proxy"])
    # Web surfer and user proxy take turns in a round-robin fashion.
    team = MagenticOneGroupChat([web_surfer], termination_condition=termination, model_client=model_client)
    # Start the team and wait for it to terminate.
    await Console(team.run_stream(task="유튜브에서 카페음악을 검색하고 3번째 영상을 틀어줘"))
    await web_surfer.close()

asyncio.run(main())