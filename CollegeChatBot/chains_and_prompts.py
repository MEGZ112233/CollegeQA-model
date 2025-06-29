from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.schema import StrOutputParser
from operator import itemgetter

SCHEDULE_INFO = """
LEVEL 4 COMPUTER SCIENCE (CS) SCHEDULE
=======================================

THEORY OF COMPUTATION
---------------------
  Lecture: Dr. Ghada Farouk - Sunday, 4:00 PM (Hall 2)

  Tutorials (Electronics Lab):
    - Dr. Zeina Rayan:
        - Saturday: Section 1 (8:00 AM), Section 2 (10:00 AM)
        - Tuesday: Section 6 (4:00 PM), Section 7 (6:00 PM)
    - Dr. Michael Elamir:
        - Saturday: Section 3 (2:00 PM)
        - Tuesday: Section 4 (10:00 AM), Section 5 (12:00 PM)

CYBER SECURITY
--------------
  Lecture: Dr. Donia Gamal - Saturday, 12:00 PM (Hall 2)

  Labs:
    - Dr. Mostafa Shawki & Dr. Amira Fekry: Section 4 - Sunday, 10:00 AM (Class 1)
    - Dr. Shamia Magdy & Dr. Amira Fekry (HP Lab):
        - Wednesday: Section 5 (8:00 AM)
        - Thursday: Section 1 (8:00 AM), Section 2 (10:00 AM), Section 3 (4:00 PM)
    - Dr. Shamia Magdy & Dr. Mostafa Shawki: Section 6 - Wednesday, 10:00 AM (HP Lab)
    - Dr. Amira Fekry & Dr. Mostafa Shawki: Section 7 - Wednesday, 4:00 PM (HP Lab)

DISTRIBUTED COMPUTING
---------------------
  Lecture: Dr. Abeer Mahmoud - Sunday, 12:00 PM (Hall 2)

  Labs (Taught by Dr. Salma Maher & Dr. Hasnaa Ehab in CIS Lab 7):
    - Wednesday: Section 6 (8:00 AM), Section 5 (10:00 AM), Section 7 (12:00 PM)
    - Thursday: Section 2 (8:00 AM), Section 1 (10:00 AM), Section 4 (12:00 PM), Section 3 (2:00 PM)

SOFTWARE QUALITY ASSURANCE
--------------------------
  Lecture: Dr. Yasmin Afify - Tuesday, 2:00 PM (Hall 2)

  Labs:
    - Dr. Salma Elkady & Dr. Oliver Ayman: Section 5 - Tuesday, 8:00 AM (SC Lab)
    - Dr. Salma Elkady & Dr. Yasmine Shabaan: Section 6 - Tuesday, 12:00 PM (Class 4)
    - Dr. Salma Elkady & Dr. Sahar Saber: Section 7 - Wednesday, 10:00 AM (CIS Lab 2)
    - Dr. Oliver Ayman & Dr. Yasmine Shabaan (CIS Lab 2):
        - Thursday: Section 4 (8:00 AM), Section 3 (10:00 AM)
    - Dr. Sahar Saber & Dr. Yasmine Shabaan: Section 2 - Thursday, 12:00 PM (CIS Lab 2)
    - Dr. Sahar Saber & Dr. Oliver Ayman: Section 1 - Thursday, 2:00 PM (CIS Lab 2)

COMPUTER ANIMATION
------------------
  Labs (Taught by Dr. Heeba G Saleh & Dr. Mohamed Mosa):
    - Wednesday (CSys Lab): Section 7 (8:00 AM), Section 5 (12:00 PM), Section 6 (2:00 PM)
    - Thursday (Robot Lab): Section 3 (8:00 AM), Section 4 (10:00 AM), Section 1 (12:00 PM), Section 2 (2:00 PM)

GAME DESIGN AND IMPLEMENTATION
------------------------------
  Lecture: Dr. Mariam Nabil - Sunday, 2:00 PM (Hall 2)
"""
class Templates:
    """Define and return prompt templates for question answering and refinement."""
    qa_template = ChatPromptTemplate.from_messages([
        ("system", """ 
            You will be given a question and context, and you must understand the context and the question to answer in natural language.
           If the conversation history is available, use it to understand follow-up questions. , also if you can't have a uncertainty ask the user to be more specfic and try to ask him about the question you need to know from him 
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
        ("system", """ You are an intelligent query analysis assistant for a college information system. Your primary task is to determine the optimal number of documents to retrieve from each of four distinct data sources, and to decide if a specific, fixed piece of information about course schedules should be included.
            The five data sources are, in this specific order:
            1. Abstracts: Contains summaries and details of student graduation projects and research.
            2. Emails: Contains contact information (email addresses, departments) for college doctors and staff.
            3. Finals: Contains links to past final exam papers for various courses.
            4. Rules: Contains official college regulations, academic rules, course prerequisites, and program structures.
            5. Schedules: A fixed text containing the weekly schedule for Level 4 Computer Science courses, including lecture times, tutorial sections , doctors , instructors, and labs .

            You will be given:
            1. The User's Question.
            2. The chat history.
            3. A Sample Snippet from each of the five data sources to help you understand the type of information each contains.

            Your goal is to output a Python list of 5 integers. The first four integers represent the estimated number of relevant documents to retrieve. The fifth integer is a binary flag (0 or 1).
            The order in the list MUST match the order of the data sources listed above:
            [number_for_Abstracts, number_for_Emails, number_for_Finals, number_for_Rules, flag_for_Schedules]

            Guidelines for determining the numbers:
            * Relevance is Key: For sources 1-4, if a source is highly relevant, suggest 1-3 documents. For broader queries (e.g., "find projects about AI"), you might suggest up to 10 from 'Abstracts'.
            * Schedule Flag: For source 5, use `1` if the question is about Level  course timings, locations, instructors, lectures ,  doctors, or labs. Otherwise, use `0`.
            * Irrelevance: If a data source is clearly irrelevant, use 0 for that source.
            * Conciseness: Prefer fewer documents if the answer is likely to be found in a small number of highly relevant chunks.
            * No Information: If you believe none of the data sources can answer the question, output [0, 0, 0, 0, 0].

            Example Scenarios:
            * Question: "What is Dr. Eman Hamdi's email address?"
                * Reasoning: This is a direct request for an email. 'Emails' is the only relevant source.
                * Expected Output: [0, 1, 0, 0, 0]
            * Question: "When is the Cyber Security lecture?"
                * Reasoning: This is a direct question about a schedule. 'Schedules' is the only relevant source.
                * Expected Output: [0, 0, 0, 0, 1]
            * Question: "What are the college's rules regarding plagiarism and what's the email for the head of the CS department?"
                * Reasoning: Needs 'Rules' for plagiarism and 'Emails' for the department head.
                * Expected Output: [0, 1, 0, 1, 0]
            """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """
                    Now, analyze the following:
            User's Question:
                  {question}

            Data Source Samples:
            1. Sample from 'Abstracts' Data Source:
                  **Faculty Information**
                  Faculty of Computer Science and Information
                  **Abstract**
                  During the last few decades, with the rise of Youtube, Amazon, Netflix... recommender systems have taken more and more place in our lives.

            2. Sample from 'Emails' Data Source:
                  Dr. Eman Hamdi is part of the Unknown department and can be reached at emanhamdi@cis.asu.edu.eg.

            3. Sample from 'Finals' Data Source:
                  The final exam for Algorithms Analysis & Design course, offered by the scientific computing department, from 2021, is available at the following link: [https://drive.google.com/...]

            4. Sample from 'Rules' Data Source:
                  Topic: Scientific Computing Fourth Level Courses
                  Summary: Outlines the courses, credit hours, and prerequisites for the seventh and eighth semesters...

            5. Sample from 'Schedules' Data Source:
                  LEVEL 4 COMPUTER SCIENCE (CS) SCHEDULE
                  =======================================
                  THEORY OF COMPUTATION
                  ---------------------
                    Lecture: Dr. Ghada Farouk - Sunday, 4:00 PM (Hall 2)
                  CYBER SECURITY
                  --------------
                    Lecture: Dr. Donia Gamal - Saturday, 12:00 PM (Hall 2)

            Your Output (Python list of 5 integers only):
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
