from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool,create_react_agent
from langchain_groq import ChatGroq
# from langchain_community.tools import TavilySearch
from langchain_tavily import TavilySearch
import datetime
from langchain import hub
import getpass
import os



load_dotenv()
import os


llm=ChatGroq(model_name="Gemma2-9b-It",groq_api_key=os.getenv("GROQ_API_KEY"))
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))

search_tool = TavilySearch(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

react_prompt =hub.pull("hwchase17/react")

react_agent_runnable=create_react_agent(tools=tools,llm=llm,prompt=react_prompt)