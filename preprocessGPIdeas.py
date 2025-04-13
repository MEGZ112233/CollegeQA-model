import fitz  # PyMuPDF
import io
from PIL import Image
import os
from typing import List
from dotenv import load_dotenv
import psycopg2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
import time
load_dotenv() 
models = [
    ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", api_key=os.getenv("G1")),
    ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", api_key=os.getenv("G2")),
    ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", api_key=os.getenv("G3")),
    ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", api_key=os.getenv("G4")),
    ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", api_key=os.getenv("G5")),
]
def setup_dataBase():
    DB_HOST = os.getenv("DB_HOST")  
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT")

    try:
    # Establish connection
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn 
    # Close the connection
       

    except Exception as e:
        print("Error connecting to database:", e)

def encode_image(image):
    """Convert PIL Image to base64 string"""
    import base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_response (message) :
    idx = 0 
    modelCount = len(models)
    while True : 
        try : 
            response = models[idx%modelCount].invoke([message])
            return response.content
        except Exception as e : 
            print('Error at Generating response from gemeni wait 3 second to try again')
            time.sleep(3) 
            idx+=1 

def extract_text_pdf (pdf_path ,conn , cursor ):
    """
    Extract text from each page of a PDF using LangChain with Google's Gemini Pro Vision
    """
    template = """
    *"You are an AI assistant that extracts and formats lecture and lab schedules from structured tables. Given the following schedule, provide a well-structured output with clear details about each course, including:

Course Name

Type (Lecture/Lab/Tutorial)

Time Slot

Day

Instructor(s)

Location

Format the response as follows:
[Course Name]

Type: Lecture/Lab/Tutorial

Day: [Day]

Time: [Start Time] - [End Time]

Location: [Room/Lab Name]

Instructor(s): [List of names]

For example, if the input is:

'Cyber Security Lab CS4, Section (5), Wednesday, HP Lab, 5 PM - 6 PM, Shamia Magdy - Amira Fekry'

Then the expected output should be:

Cyber Security Lab

Type: Lab

Day: Wednesday

Time: 5:00 PM - 6:00 PM

Location: HP Lab

Instructor(s): Shamia Magdy, Amira Fekry

Now, extract and format the schedule from the following input:
    """
    pdf_document = fitz.open(pdf_path)
    for pageIdx in range(len(pdf_document)):
        page = pdf_document[pageIdx]
        pix =  page.get_pixmap(matrix = fitz.Matrix(300/72 , 300/72))
        img_data  = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        message =  HumanMessage(
            content = [
                {"type" : "text" , "text" : template}, 
                   {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image)}"}}
            ]
        )
        response =  get_response(message)
        print(response)
        # try : 
        #     cursor.execute("INSERT INTO gp_ideas (content) VALUES (%s)", (response,))
        #     conn.commit()
        #     print("inserted new row!!!")
        # except Exception as e  : 
        #     print(e)    
        


if __name__ == "__main__":
    conn = setup_dataBase()
    cursor = conn.cursor()
    extract_text_pdf("gadwal.pdf" , conn , cursor)
    cursor.close()
    conn.close()
          