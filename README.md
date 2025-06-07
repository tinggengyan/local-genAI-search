# Local-GenAI Search - Local generative search

Local GenAI Search is your local generative search engine 
based on Llama3 model that can run localy on 32GB 
laptop or computer (developed with MacBookPro M2 with 32BG RAM).

The main goal of the project is that it lets user ask questions 
about content of their local files, which it answers in 
concise manner with referencing relevant documents that can be 
then opened. 

![img.png](img.png)

The engine is using MS MARCO embeddings for semantic search,
with top documents being passed to  Llama 3 model. 

By default, it would work with NVIDIA API, and use 70B parameter Llama 3 
model. However, if you used all your NVIDIA API credits or 
do not want to use API for searching your local documents, 
it can also run locally, using 8B parameter model. 


## How to run

In order to run your Local Generative AI Search (given you have sufficiently string machine to run Llama3), you need to 
download the repository:

````
git clone https://github.com/nikolamilosevic86/local-gen-search.git
````
You will need to install all the requirements:
```commandline
pip install -r requirements.txt
```

You need to create a file called ``config.py``, and put there
your HuggingFace API key. 
You can copy ``config_example.py`` and modify it.

API key for HuggingFace can be retrieved at ``https://huggingface.co/settings/tokens``.
In order to run generative component, you need to request
access to Llama3 model at ```https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct```

API key for Nvidia NIM API Endpoint can be retrieved at ```https://build.nvidia.com/explore/discover```

The next step is to index a folder and its subfolders containing
documents that you would like to search. You can do it using
the ``index.py`` file. Run

```commandline
python index.py path/to/folder
```
As example, you can run it with TestFolder provided:
```commandline
python index.py ~/Document/studyfiles
```
This will create a qdrant client index locally and index all the files
in this folder and its subfolders with extensions ```.pdf```,```.txt```,```.docx```,```.pptx```

The next step would be to run the generative search service.
For this you can run:

```commandline
python uvicorn_start.py
```

This will start a local server, that you can query using postman, 
or send POST requests. Loading of models (including 
downloading from Huggingface, may take few minutes, 
especially for the first time). There are two interfaces:
```commandline
http://127.0.0.1:8000/search
```

```commandline
http://127.0.0.1:8000/ask_localai
```

Both interfaces need body in a format:

```commandline
{"query":"What are knowledge graphs?"}
```
and headers for Accept and Content-Type set to ``application/json``.

Here is a code example:

```python
import requests
import json

url = "http://127.0.0.1:8000/ask_localai"

payload = json.dumps({
  "query": "What are knowledge graphs?"
})
headers = {
  'Accept': 'application/json',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
```
Finally, streamlit user interface can be started in the following way:
```commandline
streamlit run user_interface.py
```

Now you can use the user interface and ask question that will be 
answered based on the files on your file system.

## Technology used

- Llama3 8B
- NVIDIA NIM API Endpoints (For Llama 3 70B)
- Langchain
- Transformers
- MSMarco IR embedding models
- PyPDF2

## Towards Data Science article
If you want to see more details on development of this tool, you can read 
[How to Build a Generative Search Engine for Your Local Files Using Llama 3 | Towards Data Science](https://towardsdatascience.com/how-to-build-a-generative-search-engine-for-your-local-files-using-llama-3-399551786965)

Also, you can check the following papers:
```
@article{kovsprdic2024verif,
  title={Verif.ai: Towards an Open-Source Scientific Generative Question-Answering System with Referenced and Verifiable Answers},
  author={Ko{\v{s}}prdi{\'c}, Milo{\v{s}} and Ljaji{\'c}, Adela and Ba{\v{s}}aragin, Bojana and Medvecki, Darija and Milo{\v{s}}evi{\'c}, Nikola},
  journal={arXiv preprint arXiv:2402.18589},
  year={2024}
}
```


## Contributors

* [Nikola Milosevic](https://github.com/nikolamilosevic86)



# command
1. docker run -p 6333:6333 qdrant/qdrant
2. python index.py ~/Document/studyfiles
3. python uvicorn_start.py
4. streamlit run user_interface.py