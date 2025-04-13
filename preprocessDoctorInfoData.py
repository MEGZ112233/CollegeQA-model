import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import time

# Setup environment
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
models = [
    ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("G1")),
    ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("G2")),
    ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("G3")),
    ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("G4")),
    ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("G5")),
]



RefineText = ChatPromptTemplate.from_messages([
    ("human", """
     Your task is to clean up the following text:
    1. Remove any non-English text (like Arabic or other foreign languages)
    2. Remove special characters and unusual formatting (like extra periods, pipes, etc.)
    3. Make the text sound natural and professional in English
    4. Preserve the important information (name, department, email, position)

    IMPORTANT: Output ONLY the cleaned text with no additional words, phrases, or explanations.
    Do not include phrases like "Here is" or "The cleaned text is".
    
    Here are some examples:
    
    Example 1:
    Input: Dr. Dr .Hala Mousher Ebied | .  - is part of the Unknown department and can be reached at halam@cis.asu.edu.eg. وكيلة الكلية لشؤون الطلاب
    Output: Dr. Hala Mousher Ebied is part of the Unknown department and can be reached at halam@cis.asu.edu.eg.
    
    Example 2:
    Input: Dr. Dr. Howida Shedeed | Howida Rais | .   - is part of the SC department and can be reached at dr_howida@cis.asu.edu.eg. رئيسة قسم SC
    Output: Dr. Howida Shedeed (Howida Rais) is part of the SC department and can be reached at dr_howida@cis.asu.edu.eg.
    
    Example 3:
    Input: Prof ! Dr... Ahmad Hussein | العميد .| - is part of the CS department and can be reached at ahmad.hussein@faculty.edu. عميد الكلية
    Output: Prof. Dr. Ahmad Hussein is part of the CS department and can be reached at ahmad.hussein@faculty.edu.
    
    Now, clean this text:
    Input text: {input_text} """)
])



# Chains

def get_query (input_text) : 
   modelCount=len(models)
   idx = 0  
   while True : 
    try : 
        chain =  RefineText | models[idx%modelCount] |StrOutputParser()
        restult =  chain.invoke({"input_text" : input_text})
        return restult
    except Exception as e : 
           idx+=1 
           time.sleep(3) 
    
def preprocessDoctors():
    """fucntion to preprocess the data of doctors"""
    try:
       with open('doctorInfo.txt', 'w',encoding='utf-8') as fileout:
            with open('doctors_info.txt' , encoding='utf-8') as file:
              for line in file:
                line = line.strip()
                result  = get_query(line)
                fileout.write(result + '\n')
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
preprocessDoctors()
