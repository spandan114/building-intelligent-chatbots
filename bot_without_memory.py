import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

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
    
    return chain

def main():
    st.title("💬 Chatbot with Memory")
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.session_id = str(hash(str(st.session_state)))
    
    # Initialize chat chain
    chat_chain = init_chat_chain()
    
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

if __name__ == "__main__":
    main()