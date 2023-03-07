import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json
import pandas as pd

@st.cache_resource
def load():
    model = SentenceTransformer('all-mpnet-base-v2')
    index = faiss.read_index('/content/drive/MyDrive/vector.index')
    return model, index

@st.cache_data
def load_data():
    with open('/content/drive/MyDrive/key2.json','r') as f:
        db = f.read()
    with open('/content/drive/MyDrive/rest.json','r') as f:
        file_record = json.loads(f.read())
    df=pd.read_csv("/content/drive/MyDrive/firstlarge.csv",names=["Sequence", "st","score"])
    data=df.Sequence.to_list() 
    
    return db, file_record,data



def search2(query,model, index3, data, keys, file_record):
    query_vector = model.encode([query])
    k = 7
    top_k = index3.search(query_vector, k)
    resultss = []
  #  print(top_k)
    for _id in top_k[1].tolist()[0]:
        if data[_id] in keys:
            kk = keys[data[_id]]
                   
            if kk[1] >= 4:
                pre = " ".join( file_record[ kk[0]][kk[1]-4:kk[1]])
            else:
                pre = " ".join( file_record[kk[0]][:kk[1]])
      #  print(pre)
            if kk[1]+4<len(file_record[ kk[0]]):
             post = " ".join( file_record[ kk[0]][kk[1]+1:kk[1]+4])
            else:
             post =  " ".join( file_record[ kk[0]][kk[1]+1:])

            resultss.append(
                    {
                'file':kk[0],
                'pre': pre ,
                'sentence':data[_id],
                'post': post,

            }
                )

    return resultss



def app():
    st.set_page_config(page_title="Search App")
    st.title("Search App")

    # Load data and model
    model, index = load()
    db, file_record, data = load_data()
    keys = json.loads(db)
    index3 = faiss.IndexFlatIP(768)
    index3.add(index)

    # Add search bar
    query = st.text_input("Enter a sentence to search")

    # Perform search when user clicks button
    if st.button("Search"):
        results = search2(query, model, index3, data, keys, file_record)
        if len(results) == 0:
            st.write("No results found.")
        else:
            # Display results as a table
            df = pd.DataFrame(results)
            st.dataframe(df)