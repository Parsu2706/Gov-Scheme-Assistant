""" Load the CSV and clean the text"""

import re 
import pandas as pd 

from config import DATA_PATH , SECTIONS , MINISTRY_FIXES , HEALTH_KEYWORDS

def fix_ministry(row):
    name = row["State / UT / Ministry"]
    if name in MINISTRY_FIXES:
        return MINISTRY_FIXES[name]
    if name == "Ministry Of H":
        text = (row["Scheme Name"] + " " + str(row["Description"])).lower()
        if any(word in text for word in HEALTH_KEYWORDS):
            return "Ministry of Health and Family Welfare"
        return "Ministry of Housing and Urban Affairs"
    return name


def clean_text(text): 
    """ remove extra spaces and blank lines"""
    if pd.isna(text):
        return ""
    text = str(text).replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_documents(text): 
    text = re.sub(r"\s*[•\n]\s*", " ", text)
    return text.strip()

def clean_process(text): 
    return re.sub(r"\s*(Step\s*\d+\s*:)", r"\n\1", text).strip()

def load_and_clean(): 
    df = pd.read_csv(DATA_PATH)
    df['State / UT / Ministry'] = df.apply(fix_ministry , axis=1)

    for column in SECTIONS: 
        df[column] = df[column].apply(clean_text)

    df["Documents Required"] = df["Documents Required"].apply(clean_documents)
    df["Application Process"] = df["Application Process"].apply(clean_process)

    same = df["Eligibility (General)"] == df["Eligibility Criteria"]
    df.loc[same, "Eligibility (General)"] = ""
 
    return df


if __name__ == "__main__": 
    data = load_and_clean()
    print(data.shape)
    print(data["State / UT / Ministry"].value_counts().head(10))
