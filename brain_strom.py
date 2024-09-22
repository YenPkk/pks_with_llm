### 這樣其實也算是以一種對答的方式，為了更精確撈出我們需要的資料

import os
import re
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma


def llm_retriver(embeddings, vector_store):

    llm = ChatOllama(model='llama3.1')
    retriever = vector_store.as_retriever(search_kwargs={'k':5}) 

    # 這是langchain的特規設計，目的是為了使用者體驗?!，反正設計者希望，將對話紀錄與當前問題整合成"a single standalone question."(一個獨立的問題)，簡單來說就是把Input跟對話紀錄合成一個單一個query
    prompt_step1 = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    # 另外，最重要的是對話紀錄也需要做成可以被 檢索的向量，所以需要用 create_history_aware_retriever() 建立 1 個可以檢索對話紀錄的 chain： (也沒錯)
    # “Conversation Retrieval Chain” 的 chat history 與新的使用者訊息會先經過一次語言模型產(llm)生 1 個新的 search query, 
    # 若是要丟給llm結合那就很合理了，丟給llm用prompt的格式再適合不過了，這也是為什麼這邊也是寫成prompt
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt_step1)


    # 這邊這個步驟就跟上面的Retrival Chain一樣，把retrieve出來的東西塞到模板prompt template當中
    # 這一份才是最終送進要吐出回答給使用者的模型當中
    prompt_step2 = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions in traditional Chinese, based on the below context:\n\n{context}'),
    MessagesPlaceholder(variable_name="chat_history"), # 這行程式一定要加 # 你可能會覺得上面不是把對話紀錄跟當前的使用者輸入整合成一個問題了嗎?這樣這邊prompt不就只要input就好了嗎???但，並沒有，人家還是拆開的!?反正要加
    ('user', '{input}'),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt_step2)


    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    # 對話紀錄
    chat_history = []

    context = []


    input_text = input('>>> ')
    while input_text.lower() != 'bye':
        output = {} # 每次都要刷新，這樣邏輯才會對 # 用來裝streaming的字串，整理好之後再加進chat_history之中
        for chunk in retrieval_chain.stream({
                'input': input_text,
                'chat_history': chat_history,
                'context': context,
            }):

            for key in chunk:
                # print(chunk)
                # print(key)
                if key not in output:
                    output[key] = chunk[key]    # 看不太懂這在幹嘛....乾...這不就dict基本操作...你被那個"key"字搞混了，這邊就是再說如果output中沒有某一個key，那麼就建一個key其對應值是chunk[key]的那個值
                else:                           
                    output[key] += chunk[key]   # 如果output已經有這個key了，那麼就把此key對應到的value全部接起來變成一個字串，output[key]對應到的值，就會變一個很長的字串
                # if key != curr_key:
                #     print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
                # else:
                #     print(chunk[key], end="", flush=True)
                if key == 'answer':     # 只要print answer就好，chunk包含全部的內容(input, 從db撈出來的資料....)
                    print(chunk[key], end="", flush=True)
                # curr_key = key

        chat_history.append(HumanMessage(content=input_text))
        chat_history.append(AIMessage(content=output['answer']))
        need_file_context = output['context']

        print()
        input_text = input('>>> ')
    return need_file_context

def llm_summary(need_file_context):

    summary_llm = ChatOllama(model='llama3.1')

    # 可能要提供對話紀錄給模型這樣比較好....做摘要，不然都亂回答
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'Summary in traditional Chinese with about 800 word, based on the below user input content :'), 
        ('user', "{input}")

    ])
    ## 嘗試多個腳色多個觀點的prompt??!


    # print(prompt.invoke({'input': need_file_context, 'question':chat_history,}))  # context(List of Document是有崁進來的，但是就是不讀....) # 這樣可以看prompt長什麼樣子

    chain = prompt | summary_llm

    result = chain.invoke({'input': need_file_context[0:3], # 這邊可以直接給list of Document!!!!!
                            })

    summary_result = result.content
    return summary_result

def write_into_file(need_file_context, summary_result):
    internal_link_list = []
    for each_file in need_file_context:
        internal_link_list.append(each_file.metadata['file_name'])
    # print(internal_link_list)

    output_format_string = str(summary_result) + "\n\n---\n\n"
    for each in internal_link_list:
        output_format_string += f"[[{each}|{each.split('/')[-1]}]]\n" # Obsidain internal link format [[檔案完整路徑|別名(顯示名稱)]]
    # regex_file_name_format = r'[\?*"\\\/<>:|]' # 處理一下問題中可能有不符合檔名的符號
    # we_need_this_query = re.sub(regex_file_name_format, '', we_need_this_query)
    inbox_dir = "C:\\Users\\Tony\\WORKING_DIR\\PKs\\Inbox\\"
    query_file_name = "$$" + "test" + ".md"

    output_file_route = os.path.join(inbox_dir, query_file_name)
    with open(output_file_route, 'w', encoding='utf-8') as f:
        f.write(output_format_string)

if __name__ == '__main__':
    embeddings = OllamaEmbeddings(model='nomic-embed-text')

    vector_store = Chroma(
        collection_name="brain_strom",
        embedding_function=embeddings,
        persist_directory="./brain_strom",  # Where to save data locally, remove if not neccesary
    )
    need_file_context = llm_retriver(embeddings, vector_store)
    summary_result = llm_summary(need_file_context)
    write_into_file(need_file_context, summary_result)