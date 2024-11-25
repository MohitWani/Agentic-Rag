from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition

from dotenv import load_dotenv
import os 

load_dotenv()

Groq_api_key = os.environ['Groq_API_key']


from langchain.tools.retriever import create_retriever_tool


embedding = GPT4AllEmbeddings()
db = FAISS.load_local('faiss_index', embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_Mutual_Fund_information",
    "Search and return information about HDFC Mutual Fund on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]
   



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]




### Edges


def grade_documents(state) -> Literal["generate", "rewrite"]:
    
    
    class grade(BaseModel):

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    
    model = ChatGroq(groq_api_key=Groq_api_key,
         model_name="mixtral-8x7b-32768")

    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Nodes


def agent(state):
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatGroq(groq_api_key=Groq_api_key,
         model_name="mixtral-8x7b-32768")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatGroq(groq_api_key=Groq_api_key,
         model_name="mixtral-8x7b-32768")
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):

    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatGroq(groq_api_key=Groq_api_key,
         model_name="mixtral-8x7b-32768")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}



# Define a new graph
def graph_workflow():
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    return workflow.compile()


if __name__=="__main__":

    graph = graph_workflow()

    import pprint

    inputs = {
        "messages": [
            ("user", "How is the trend for NAV across the three months?"),
        ]
    }
    for output in graph.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")