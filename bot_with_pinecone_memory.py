# app.py
import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.schema import Document

# Load environment variables
load_dotenv()

def serialize_message(msg: BaseMessage) -> dict:
    """Serialize a LangChain message to a dictionary."""
    return {
        "type": msg.__class__.__name__,
        "content": msg.content,
    }

def deserialize_message(msg_dict: dict) -> BaseMessage:
    """Deserialize a dictionary to a LangChain message."""
    msg_type = msg_dict["type"]
    if msg_type == "HumanMessage":
        return HumanMessage(content=msg_dict["content"])
    elif msg_type == "AIMessage":
        return AIMessage(content=msg_dict["content"])
    else:
        raise ValueError(f"Unknown message type: {msg_type}")

class PineconeMemory:
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("chat-memory")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text",
            namespace="chat_history"
        )
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        try:
            # Search for session history
            results = self.vector_store.similarity_search(
                session_id,
                filter={"session_id": session_id},
                k=3  # Get last 3 relevant histories
            )
            
            history = ChatMessageHistory()
            if results:
                # Process results from newest to oldest
                for result in reversed(results):
                    history_data = json.loads(result.page_content)
                    if "messages" in history_data:
                        for msg_dict in history_data["messages"]:
                            msg = deserialize_message(msg_dict)
                            history.messages.append(msg)
            
            return history
            
        except Exception as e:
            print(f"Error retrieving history: {e}")
            return ChatMessageHistory()
    
    def save_history(self, session_id: str, history: ChatMessageHistory):
        try:
            # Serialize the chat history
            history_data = {
                "messages": [serialize_message(msg) for msg in history.messages],
                "session_id": session_id
            }
            
            # Create document for vector store
            document = Document(
                page_content=json.dumps(history_data),
                metadata={"session_id": session_id}
            )
            
            # Save to Pinecone
            self.vector_store.add_documents([document])
            
        except Exception as e:
            print(f"Error saving history: {e}")

# Initialize Pinecone memory
@st.cache_resource
def init_pinecone_memory():
    return PineconeMemory()

# Initialize LLM and chat chain
@st.cache_resource
def init_chat_chain():
    llm = ChatGroq(
        model="gemma2-9b-it",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant please answer the question."),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    memory = init_pinecone_memory()
    
    return RunnableWithMessageHistory(
        chain,
        memory.get_session_history,
    ), memory

def main():
    st.title("💬 Chatbot with Pinecone Memory")
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.session_id = str(hash(str(st.session_state)))
    
    # Initialize chat chain and memory
    chat_chain, memory = init_chat_chain()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            config = {"configurable": {"session_id": st.session_state.session_id}}
            with st.spinner("Thinking..."):
                response = chat_chain.invoke(
                    {"input": prompt},
                    config=config,
                )
            
            # Display bot response
            st.markdown(response.content)
            
            # Add bot response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response.content}
            )
            
            # Save chat history to Pinecone
            history = memory.get_session_history(st.session_state.session_id)
            history.add_user_message(prompt)
            history.add_ai_message(response.content)
            memory.save_history(st.session_state.session_id, history)

if __name__ == "__main__":
    main()