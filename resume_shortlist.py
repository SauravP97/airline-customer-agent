import numpy as np
from langchain_core.tools import tool
import pandas as pd
import re
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from typing_extensions import TypedDict, Annotated
from IPython.display import Image, display

def custom_reducer(obj1, obj2):
  return obj2


class SharedState(TypedDict):
  """
  Represents the state of our graph.

  Attributes:
      question: question
      generation: LLM generation
      documents: list of documents
  """
  resume_content: Annotated[str, custom_reducer]
  jd_content: Annotated[str, custom_reducer]
  
  preferred_qual: Annotated[str, custom_reducer]
  minimum_qual: Annotated[str, custom_reducer]
  responsibilities: Annotated[str, custom_reducer]

  preferred_qual_feedback: Annotated[str, custom_reducer]
  minimum_qual_feedback: Annotated[str, custom_reducer]
  responsibilities_feedback: Annotated[str, custom_reducer]


def read_job_description(shared_state):
    """ Reads the job description. """
    print("Reading job description...")
    jd_doc = open('./knowledge-base/wse_jd.md', 'r')
    jd_content = jd_doc.read()
    
    shared_state['jd_content'] = jd_content
    return shared_state


def get_preferred_qualification(shared_state):
    """ Returns the preferred qualifications from the job description. """
    print("Extracting preferred qualifications from job description...")
    query = "What are the preferred qualifications for this job?"
    response = openai.Client().chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {shared_state['jd_content']} \n\n Question: {query}"}
        ]
    )
    
    shared_state["preferred_qual"] = response.choices[0].message.content.strip()
    return shared_state


def get_minimum_qualification(shared_state):
    """ Returns the minimum qualifications from the job description. """
    print("Extracting minimum qualifications from job description...")
    query = "What are the minimum qualifications for this job?"
    response = openai.Client().chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {shared_state['jd_content']} \n\n Question: {query}"}
        ]
    )
    
    shared_state["minimum_qual"] = response.choices[0].message.content.strip()
    return shared_state


def get_job_responsibilities(shared_state):
    """ Returns the job responsibilities from the job description. """
    print("Extracting job responsibilities from job description...")
    query = "What are the job responsibilities for this position?"
    response = openai.Client().chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {shared_state['jd_content']} \n\n Question: {query}"}
        ]
    )
    
    shared_state["responsibilities"] = response.choices[0].message.content.strip()
    return shared_state


def get_swe_resume_content(shared_state):
    """ Reads the SWE resume content. """
    print("Reading SWE resume content...")
    resume_doc = open('./knowledge-base/resumes/swe_resume.md', 'r')
    resume_content = resume_doc.read()
    
    shared_state['resume_content'] = resume_content
    return shared_state


def provide_feedback_for_min_qual(shared_state):
    """ Provides feedback on the minimum qualifications based on the resume content. """
    print("Providing feedback for minimum qualifications...")
    query = """
    Does this resume meet the minimum qualifications? If not, provide feedback like 
    What else can be improved in the resume ?
    What technologies or frameworks can be learned?
    What projects can be added to the resume?
    What skills can be improved or added?

    Keep your suggestions concise within 100 words and in bullet points.
    """
    response = openai.Client().chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {shared_state['resume_content']} \n\n Minimum Qualifications: {shared_state['minimum_qual']} \n\n Question: {query}"}
        ]
    )

    shared_state['minimum_qual_feedback'] = response.choices[0].message.content.strip()
    return shared_state

def provide_feedback_for_preferred_qual(shared_state):
    """ Provides feedback on the preferred qualifications based on the resume content. """
    print("Providing feedback for preferred qualifications...")
    query = """
    Does this resume meet the preferred qualifications? If not, provide feedback like 
    What else can be improved in the resume ?
    What technologies or frameworks can be learned?
    What projects can be added to the resume?
    What skills can be improved or added?

    Keep your suggestions concise within 100 words and in bullet points.
    """
    response = openai.Client().chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {shared_state['resume_content']} \n\n Preferred Qualifications: {shared_state['preferred_qual']} \n\n Question: {query}"}
        ]
    )

    shared_state['preferred_qual_feedback'] = response.choices[0].message.content.strip()
    return shared_state


def provide_feedback_for_responsibilities(shared_state):
    """ Provides feedback on the job responsibilities based on the resume content. """
    print("Providing feedback for job responsibilities...")
    query = """
    Does this resume meet the job responsibilities? If not, provide feedback like 
    What responsibilities can be improved or added?

    Keep your suggestions concise within 100 words and in bullet points.
    """
    response = openai.Client().chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {shared_state['resume_content']} \n\n Job Responsibilities: {shared_state['responsibilities_feedback']} \n\n Question: {query}"}
        ]
    )
    
    shared_state['responsibilities_feedback'] = response.choices[0].message.content.strip()
    return shared_state


def build_graph():
  # Building a Graph
  # State of the Graph that will be shared among nodes.
  workflow = StateGraph(SharedState)

  # Add nodes.
  workflow.add_node("read_job_description", read_job_description)
  workflow.add_node("get_preferred_qualification", get_preferred_qualification)
  workflow.add_node("get_minimum_qualification", get_minimum_qualification)
  workflow.add_node("get_job_responsibilities", get_job_responsibilities)
  workflow.add_node("get_swe_resume_content", get_swe_resume_content)
  workflow.add_node("provide_feedback_for_min_qual", provide_feedback_for_min_qual)
  workflow.add_node("provide_feedback_for_preferred_qual", provide_feedback_for_preferred_qual)
  workflow.add_node("provide_feedback_for_responsibilities", provide_feedback_for_responsibilities)

  workflow.add_edge(START, "read_job_description")
  workflow.add_edge("read_job_description", "get_preferred_qualification")
  workflow.add_edge("read_job_description", "get_minimum_qualification")
  workflow.add_edge("read_job_description", "get_job_responsibilities")

  workflow.add_edge("get_preferred_qualification", "get_swe_resume_content")
  workflow.add_edge("get_minimum_qualification", "get_swe_resume_content")
  workflow.add_edge("get_job_responsibilities", "get_swe_resume_content")

  workflow.add_edge("get_swe_resume_content", "provide_feedback_for_min_qual")
  workflow.add_edge("get_swe_resume_content", "provide_feedback_for_preferred_qual")
  workflow.add_edge("get_swe_resume_content", "provide_feedback_for_responsibilities")

  workflow.add_edge("provide_feedback_for_min_qual", END)
  workflow.add_edge("provide_feedback_for_preferred_qual", END)
  workflow.add_edge("provide_feedback_for_responsibilities", END)

  graph = workflow.compile()

  response = graph.invoke({})

  # print(graph.get_graph().draw_mermaid())

  return response


agent_response = build_graph()
print(agent_response.keys)