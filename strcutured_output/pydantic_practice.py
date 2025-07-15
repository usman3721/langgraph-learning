from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

import os   
load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))


# class Country(BaseModel):
#     '''Information about a country'''
#     name: str = Field(..., description="The name of the country")
#     capital: str = Field(..., description="The capital city of the country")
#     language: str = Field(..., description="The primary language spoken in the country")


# structured_llm=llm.with_structured_output(Country)

# response = structured_llm.invoke("Tell me about France?")

# print(response)

from typing_extensions import Annotated,TypedDict
from typing import Optional, List

class Joke(TypedDict):
    """Joke to tell user"""
    setup:Annotated[str,...,"Setup of the joke"]

    punchline:Annotated[str,...,"Punchline of the joke"]
    rating:Annotated[Optional[int],None,"How funny is the joke? 1-10, 10 being the funniest"]

structured_llm=llm.with_structured_output(Joke)
response = structured_llm.invoke("Tell me a joke about Cat?")

print(response)
