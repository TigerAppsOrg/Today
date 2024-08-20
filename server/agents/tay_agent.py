import os

from chains.crawl_hybrid_search_chain import crawl_hybrid_search_chain
from chains.email_hybrid_search_chain import email_hybrid_search_chain
from langchain import hub
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain_openai import ChatOpenAI
from models.tool_inputs import SingleTextInput
from langchain.agents import AgentType, initialize_agent
from langchain.schema import LLMResult
from typing import Any

AGENT_MODEL = os.getenv("AGENT_MODEL")

agent_prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [
    StructuredTool(
        name="Crawl",
        func=crawl_hybrid_search_chain.invoke,
        description="""This tool accesses a crawl of all Princeton
        and Princeton-related webpages. Useful when you need to answer
        questions about the university, academic requirements, professors,
        various academic programs, general information about campus life,
        and other general things that would be listed on a university 
        webpage. 
        
        Not useful for answering questions that involve real time
        information about campus life, clubs, events, job opportunity 
        postings, and other similar kinds of information.

        Should be used as a default fallback when other tools don't 
        give a good response.
        
        Use the entire prompt as input to the tool. For instance, if 
        the prompt is "Who is Professor Arvind Narayanan?", the input 
        should be "Who is Professor Arvind Narayanan?".
        """,
        args_schema=SingleTextInput
    ),
    StructuredTool(
        name="Emails",
        func=email_hybrid_search_chain.invoke,
        description="""This tool accesses the latest Princeton listserv
        emails. Useful when you need to answer question about real time
        events, clubs, job opportunity postings, deadlines for auditions,
        and things going on in campus life.
        
        Not useful for answering questions about academic facts or 
        professors.
        
        Use the entire prompt as input to the tool. For instance, if 
        the prompt is "What dance shows are coming up?", the input 
        should be "What dance shows are coming up?".
        """,
        args_schema=SingleTextInput
    )
]

chat_model = ChatOpenAI(
    model=AGENT_MODEL,
    temperature=0,
    streaming=True
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    output_key="output"
)

tay_agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=chat_model,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    return_intermediate_steps=False
)

class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False
    
    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""

async def run_call(query: str, stream_it: AsyncCallbackHandler):
    tay_agent.agent.llm_chain.llm.callbacks = [stream_it]
    await tay_agent.acall(inputs={"input": query})
