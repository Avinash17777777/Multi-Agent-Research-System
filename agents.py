from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search , scrape_url 
from dotenv import load_dotenv

load_dotenv()

#model setup 
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=2048,
)

# Backup model - has its own, much larger free-tier daily quota (500K tokens/day
# vs 100K for the model above), used automatically if the main model's daily
# quota runs out mid-pipeline. See build_writer_chain/build_critic_chain below.
fallback_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=2048,
)


#1st agent 
SEARCH_SYSTEM_PROMPT = (
    "You are a research search agent. The only tool available to you is web_search. "
    "Use it to gather information about the topic you're given. Never attempt to call "
    "any tool other than web_search - no other tool exists in this system, and trying "
    "to call one will crash the pipeline."
)

def build_search_agent():
    return create_agent(
        model = llm,
        tools= [web_search],
        system_prompt = SEARCH_SYSTEM_PROMPT,
    )

#2nd agent 

READER_SYSTEM_PROMPT = (
    "You are a research reader agent. The only tool available to you is scrape_url. "
    "You will be given a list of URLs - always call scrape_url on the single URL most "
    "relevant to the topic. Never attempt to call any other tool, such as a search tool "
    "- no other tool exists in this system, and trying to call one will crash the "
    "pipeline. If none of the URLs look ideal, still pick the closest match and scrape "
    "it rather than asking to search again."
)

def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url],
        system_prompt = READER_SYSTEM_PROMPT,
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()
writer_chain_fallback = writer_prompt | fallback_llm | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
critic_chain_fallback = critic_prompt | fallback_llm | StrOutputParser()