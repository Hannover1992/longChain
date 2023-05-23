import os
from langchain.llms import OpenAI
import streamlit as st
from dotenv import load_dotenv


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


print("Loading OpenAI API key from environment variable: OPENAI_API_KEY" + OPENAI_API_KEY)