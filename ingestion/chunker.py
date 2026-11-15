import json 

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNKS_PATH , CHUNK_OVERLAP , CHUNK_SIZE , SECTIONS
from ingestion.clean import load_and_clean

splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE , chunk_overlap = CHUNK_OVERLAP)

def make_chunks(df): 
    chunks = []
    for _ , row in df.iterrows(): 
        for column , section in SECTIONS.items(): 
            text = row[column]
            if not text: 
                continue

            for i , piece in enumerate(splitter.split_text(text)): 
                header = f"Scheme: {row['Scheme Name']} ({row['State / UT / Ministry']})\nSection: {section}\n"

                chunks.append({
                    "id": f"{row['Scheme Slug']}__{section}__{i}",
                    "text" : header + piece, 
                    "metadata" : {
                        "scheme_name" : row['Scheme Name'] , 
                        "level" : row["Level"] , 
                        "state" : row["State / UT / Ministry"] , 
                        "section" : section, 
                        "url" : row["MyScheme URL"]
                    }
                })
    return chunks

def save_chunks(chunks): 
    CHUNKS_PATH.parent.mkdir(parents=True , exist_ok= True)
    with open(CHUNKS_PATH , "w" , encoding="utf-8") as f : 
        for chunk in chunks: 
            f.write(json.dumps(chunk , ensure_ascii= False) + "\n") 


if __name__ == "__main__": 
    chunks = make_chunks(load_and_clean())
    save_chunks(chunks)
    print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")
