import streamlit as st
import company_backend as brain

st.set_page_config(page_title= "Company Policy Q&A Bot with RAG")

new_title = '<p style="font-family: sans-serif; color: White; font-size: 50px;">Company Policy Q&A Bot with RAG </p>'
st.markdown(new_title, unsafe_allow_html= True)

if 'vector_index' not in st.session_state:
    with st.spinner("Stay Tuned! I'm getting your company Policy Document"):
        st.session_state.vector_index = brain.company_pdf()

input_text = st.text_area("Input text", label_visibility= "collapsed")
go_button = st.button("Ask your Queries!", type = "primary")

if go_button:
    with st.spinner("I'm getting answers to your Queries!"):
        response_content = brain.company_rag_response(index= st.session_state.vector_index, question= input_text)
        st.write(response_content)
