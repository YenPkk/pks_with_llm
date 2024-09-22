from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import re
import os
from langchain_core.documents import Document
from uuid import uuid4

def write_into_chroma_db(path, pks_root_dir, vector_store):
    
    docs = []
    uni_id = 0
    source = os.walk(path)
    for dir, do_need, files in source:
        # 組合所有的dir根他下面的files
        for each_file in files:
            full_path = os.path.join(dir, each_file) # # 檔案名稱，因為筆記的重點也會寫在這裡，所以一定要留，只是看到時候如何寫入vector db
            internal_link = full_path.split('.md')[0].replace(pks_root_dir, '') # 這樣才不會切到當前目錄的那個.或前一個目錄.. # 且去除PKs 的根目錄名稱
            file_name = each_file.split('.')[0] # 檔案名稱當中.一定可以區分檔名與副檔名
            with open(full_path, 'r', encoding='utf-8') as f:
                x = f.read() # function return要用變數接

                regex = r'---\n(([\w0-9_\- ]*:[\w0-9\[\]\"\-\|:_+=\-%\/\\.@ ]*\n)*)---'

                # 文字部分(document)
                plain_text = re.sub(regex, "", x) # 把metadata去除
                final_text = re.sub(r'\s', '', plain_text) # 把空白與換行刪除，併成一行
                full_content = f"{file_name}>>{final_text}"

                # metadata部分
                modi_internal_link = internal_link.replace('\\', '/') # 修改windows目錄格式
                property_dict = {}
                property_dict['file_name'] = modi_internal_link
                try:
                    response = re.match(regex, x)
                    tmp_list = response[1].split('\n')
                    tmp_list.remove('') # 因為最後一行會有\n所以會切出一個空字串，要把它刪掉
                    for each in tmp_list:
                        key, value = each.split(':', 1) # 可以指定只切第一個:
                        property_dict[key] = value.strip()
                        

                except TypeError:
                    property_dict['no_metadata'] = 'no_metadata' # 如果筆記沒有metadata就給他一個表示沒有的dict，因為dict不能是空的
            print(file_name)
            print(property_dict)
            print('-'*50)
            each_doc = Document(page_content=full_content,  metadata=property_dict, id=uni_id)
            docs.append(each_doc)
            uni_id += 1
    uuids = [str(uuid4()) for _ in range(len(docs))]
    # 寫入chroma vector db, 指定persist_directory參數，可以將db資料寫到磁碟(local)中
    vector_store.add_documents(documents=docs, ids=uuids)


if __name__ == '__main__':

    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vector_store = Chroma(
        collection_name="brain_strom",
        embedding_function=embeddings,
        persist_directory="./brain_strom",  # Where to save data locally, remove if not neccesary
    )

    pks_root_dir = "C:\\Users\\Tony\\WORKING_DIR\\PKs\\"
    need_dir_list = ["Inbox\\", "Project\\", "Resource\\", "Area\\"]
    for each in need_dir_list:
        wrtie_dir = os.path.join(pks_root_dir, each)
        # print(wrtie_dir, pks_root_dir)
        write_into_chroma_db(wrtie_dir, pks_root_dir=pks_root_dir, vector_store=vector_store)