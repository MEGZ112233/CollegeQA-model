import pandas as pd
import os
# Load your file
base_dir = os.path.dirname(os.path.dirname(__file__))
chunk_path =   os.path.join(base_dir, "merged.csv")
df = pd.read_csv(chunk_path)

startAbstract = 0
endAbstract = 230
df.iloc[startAbstract:endAbstract].to_csv(f'abstracts.csv', index=False)
startEmail = 230
endEmail = 410
df.iloc[startEmail:endEmail].to_csv(f'emails.csv', index=False)
startRules = 410
endRules = 438
df.iloc[startRules:endRules].to_csv(f'rules.csv', index=False)
startFinal = 439
endFinals = 498
df.iloc[startFinal:endFinals].to_csv(f'finals.csv', index=False)

