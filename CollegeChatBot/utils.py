import csv
import json

from langsmith import expect
from numpy.f2py.auxfuncs import throw_error


def upload_chunks(data_path):
    chunks = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0]:
                chunks.append(row[0])
    return chunks
def parse_list(response) :
    response = response.strip()
    if response.startswith("```python"):
        response = response.replace("```python", "")
    elif response.startswith("```"):
        response = response.replace("```", "")
    response = response.strip('\n')
    try:
      jsonList  = json.loads(response)
      if isinstance(jsonList, list) and len(jsonList) == 4 and all(isinstance(x, int) for x in jsonList):
            return jsonList
      else :
          raise Exception(f"Invalid response from API: {response}")


    except json.JSONDecodeError as e:
        parse_error_message = f"JSON decoding error: {e}. Output was not valid JSON. and the output was {response}"
        print(parse_error_message)
        return [0,2,2,2]

    except Exception as e:
        parse_error_message = f"An unexpected error occurred during parsing: {e} and the output was {response}"
        print(parse_error_message)
        return [0,2,2,2]