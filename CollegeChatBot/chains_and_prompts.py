from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.schema import StrOutputParser
from operator import itemgetter


class Templates:
    """Define and return prompt templates for question answering and refinement."""
    qa_template = ChatPromptTemplate.from_messages([
        ("system", """ 
            You will be given a question and context, and you must understand the context and the question to answer in natural language.
           If the conversation history is available, use it to understand follow-up questions.  
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "question: {question}\n context: {context}"),
    ])

    refine_template = ChatPromptTemplate.from_messages([
        ("system",
         "Rephrase the student's question for clarity while preserving its meaning and if there something mispealing correct it. Make sure the output is in English and only one question."),
        ("human", "Question to refine: {question}")
    ])

    cluster_template = ChatPromptTemplate.from_messages([
        ("system", """ You are an intelligent query analysis assistant for a college information system. Your primary task is to determine the optimal number of documents to retrieve from each of four distinct data sources to best answer a student's question.
        The four data sources are, in this specific order:
        1. Abstracts: Contains summaries and details of student graduation projects and research.
        2. Emails: Contains contact information (email addresses, departments) for college doctors and staff.
        3. Finals: Contains links to past final exam papers for various courses.
        4. Rules: Contains official college regulations, academic rules, course prerequisites, and program structures.
        You will be given:
        1. The User's Question.
        2. the chat history . 
        3. A Sample Snippet from each of the four data sources to help you understand the type of information each contains.
        Your goal is to output a Python list of 4 integers. Each integer represents the estimated number of relevant documents that should be retrieved from the corresponding data source to answer the question. The order in the list MUST match the order of the data sources listed above: [number_for_Abstracts, number_for_Emails, number_for_Finals, number_for_Rules]
        Guidelines for determining the number of documents:
        * Relevance is Key: If a data source is highly relevant and likely to contain the core answer, suggest 1-3 documents. For broader queries requiring exploration (e.g., "find projects about AI"), you might suggest up to 10 from 'Abstracts'.
        * Specificity:
            * 
            * If the question asks for a single piece of specific information (e.g., "email of Dr. Smith"), 1 or 2  document from the 'Emails' source is likely sufficient.
            * If the question asks about multiple specific items (e.g., "final exams for CS101 for 2020, 2021, and 2022"), you might suggest 1-3 documents from 'Finals', assuming each year/course might be a separate document or closely related ones.
        * Irrelevance: If a data source is clearly irrelevant to the question, use 0 for that source.
        * Conciseness: Prefer fewer documents if the answer is likely to be found in a small number of highly relevant chunks. Avoid requesting many documents unnecessarily.
        * No Information: If you believe none of the data sources can answer the question, you can output [0, 0, 0, 0].
        Example Scenarios:
        * Question: "What is Dr. Eman Hamdi's email address?"
            * 
            * Expected Output Reasoning: This is a direct request for an email. 'Emails' is the only relevant source.
            * Expected Output: [0, 1, 0, 0]
        * Question: "Can you show me some graduation project abstracts on machine learning?"
            * 
            * Expected Output Reasoning: This is about project abstracts. 'Abstracts' is primary. A few examples would be good.
            * Expected Output: [3, 0, 0, 0] (Suggesting 3 abstracts)
        * Question: "Where can I find the final exam for Algorithms from 2021 and what are the prerequisites for the advanced algorithms course?"
            * 
            * Expected Output Reasoning: Needs 'Finals' for the exam link and 'Rules' for course prerequisites.
            * Expected Output: [0, 0, 1, 1] (or [0, 0, 1, 2] if prerequisites are complex and might span multiple rule chunks)
        * Question: "What are the college's rules regarding plagiarism and what's the email for the head of the CS department?"
            * 
            * Expected Output Reasoning: Needs 'Rules' for plagiarism and 'Emails' for the department head.
            * Expected Output: [0, 1, 0, 1] 
        """),
        MessagesPlaceholder(variable_name="chat_history")
        ,
        ("human", """
                Now, analyze the following:
        User's Question:
              {question}
            
        Data Source Samples:
        1. Sample from 'Abstracts' Data Source:
              **Faculty Information**
        
        Faculty of Computer Science and Information
        Scientific Computing Department
        
        **Abstract**
        
        During the last few decades, with the rise of Youtube, Amazon, Netflix and many other such web services, recommender systems have taken more and more place in our lives. From e-commerce (suggest to buyers articles that could interest them) to online advertisement (suggest to users the right contents, matching their preferences), recommender systems are today unavoidable in our daily online journeys.
        
        **Recommender Systems as Algorithms**
        
        In a very general way, recommender systems are algorithms aimed at suggesting relevant items to users (items being movies to watch, text to read, products to buy or anything else depending on industries).
            
        2. Sample from 'Emails' Data Source:
              Dr. Eman Hamdi is part of the Unknown department and can be reached at emanhamdi@cis.asu.edu.eg.
        3. Sample from 'Finals' Data Source:
              The final exam for Algorithms Analysis & Design course, offered by the scientific computing department, from 2021, is available at the following link: [https://drive.google.com/file/d/1mT21jrptv4w2IdFd5G-oVxI_6lR_SrS9/view
        The final exam for Algorithms Analysis & Design course, offered by the scientific computing department, from 2020, is available at the following link: [https://drive.google.com/file/d/1W_X01ASI0yyo6fCG4SwZYlvoE-Sh_C0g/view?usp=drive_link
        4. Sample from 'Rules' Data Source:
              Topic: Scientific Computing Fourth Level Courses
        Summary: Outlines the courses, credit hours, and prerequisites for the seventh and eighth semesters of the fourth level Scientific Computing program.
        Chunk: ""Fourth Level Courses For Scientific Computing (SC4) 
        Seventh Semester  
        • SCO422: Computational Geometry (3 Credit Hours) – Prerequisite: SCO311 (Computer 
        Graphics) 
        • SCO411: Neural Networks & Deep Learning (3 Credit Hours) – Prerequisite: BSC225 (Linear 
        Algebra) 
        • SCO421: Computer Vision (3 Credit Hours) – Prerequisite: CIS243 (Artificial Intelligence) 
        • Total Credit Hours: 18 
        Eighth Semester 
        • CSY410: Computer and Network Security (3 Credit Hours) – Prerequisite: CIS365
            
        Your Output (Python list of 4 integers only):
                """)
    ])


def get_chains(models, idx, memory):
    """Build LangChain pipelines for question refinement, retrieval, and answering."""
    chain_question_refine = Templates.refine_template | models[idx] | StrOutputParser()
    chain_answer_question = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
            )
            | Templates.qa_template | models[idx] | StrOutputParser()
    )
    chain_cluster_question = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
            )
            | Templates.cluster_template | models[idx] | StrOutputParser()
    )
    return chain_question_refine, chain_answer_question, chain_cluster_question
