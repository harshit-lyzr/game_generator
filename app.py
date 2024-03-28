import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent,Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
api = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Lyzr Game Generator",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr Game Generator")
st.markdown("### Welcome to the Lyzr Game Generator!")

query=st.text_input("Enter your Game Name: ")

open_ai_text_completion_model = OpenAIModel(
    api_key=api,
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)

def game_generator(query):
    engineer_agent = Agent(
        role="Senior Software Engineer",
        prompt_persona="""You are a Senior Software Engineer at a leading tech think tank.
                Your expertise in programming in python. and do your best to
                produce perfect code"""
    )

    qa_agent=Agent(
        role="Software Quality Control Engineer",
        prompt_persona="""You are a software engineer that specializes in checking code
            for errors. You have an eye for detail and a knack for finding
                hidden bugs.
            You check for missing imports, variable declarations, mismatched
                brackets and syntax errors.
            You also check for security vulnerabilities, and logic errors"""
    )

    senior_qa_engineer=Agent(
        role="Senior Software Quality Control Engineer",
        prompt_persona="""\
                You feel that programmers always do only half the job, so you are
                super dedicate to make high quality code."""
    )

    prompt=f"""You will create a game using python, these are the instructions:
    
            Instructions
            ------------
        {query} game
    
            Your Final answer must be the full python code, only the python code and simple documentation how to play up to 30 words and nothing else.
            """

    qa_prompt=f"""You are helping create a game using python, these are the instructions:
    
            Instructions
            ------------
            {query} game
    
            Using the code you got, check for errors. Check for logic errors,
            syntax errors, missing imports, variable declarations, mismatched brackets,
            and security vulnerabilities.
    
            Your Final answer must be the full python code, only the python code and nothing else.
            """

    senior_qa_prompt=f"""\
            You are helping create a game using python, these are the instructions:
    
            Instructions
            ------------
            {query} game
    
            You will look over the code to insure that it is complete and
            does the job that it is supposed to do.
    
            Your Final answer must be the full python code, only the python code and nothing else.
            """

    code_task  =  Task(
        name="Code Generation",
        model=open_ai_text_completion_model,
        agent=engineer_agent,
        instructions=prompt,
    )

    review_task=Task(
        name="QA Testing",
        model=open_ai_text_completion_model,
        agent=qa_agent,
        instructions=qa_prompt,
    )

    evaluate_task =Task(
        name="Senior QA Testing",
        model=open_ai_text_completion_model,
        agent=senior_qa_engineer,
        instructions=senior_qa_prompt,
    )

    output = LinearSyncPipeline(
        name="Game Pipline",
        completion_message="pipeline completed",
        tasks=[
              code_task,
              review_task,
              evaluate_task
        ],
    ).run()

    return output[0]['task_output']

if st.button("Generate"):
    game=game_generator(query)
    st.markdown(game)

