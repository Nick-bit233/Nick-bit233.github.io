---
title: LangChain速成文档
date: 2024-01-23 17:37:30
tags:
---

> 注意，本篇基于langchain官方文档编写，众所周知LangChain的更新频率很高，因此本文不保证具有时效性
> 
> 最后更新时间：2024-01-23， LangChain版本0.1.2

## LangChain基础：使用链和LLM对话
### 安装langchain
使用pip：
```python
pip install langchain
pip install langchain-core # 正常来说，上面一行命令也会同时安装langchain-core
pip install langchain-community # 安装第三方贡献的langchain库
pip install langchain-openai # 使用OPENAI的模型和api需要的集成包
```
如果你使用OPENAI 的 api，可能需要按照下面的方法配置api：
```python
"""配置部分：采用环境变量，但所有变量在实例化类对象时均可以传入参数的形式覆盖"""

# API_KEY
api_key = "sk-xxx" 
# 默认调用模型名称
model_name = "gpt-3.5-turbo"
# 默认调用模型的最大token数
max_tokens = 4096
# 默认调用模型的温度
temperature = 0
# 其他配置

"""
如果想把 OPENAI API 的请求根路由修改成自己或第三方的代理地址，可以通过设置环境变量 “OPENAI_API_BASE” 来进行修改。
12/14备注：实测openai的python库无法正确使用http规则代理（我不知道为什么，也许要去看源码），
          如果使用官方接口，建议开启全局代理，然后不再设置代理环境变量
"""
# set openai api key
os.environ["OPENAI_API_KEY"] = api_key
# set openai proxy
# os.environ["OPENAI_PROXY"] = "http://127.0.0.1:7890"
# set openai base url
os.environ["OPENAI_API_BASE"] = "https://aigptx.top/v1"
```
### 基本使用
#### 基础LLM查询
LangChain包括两种使用语言模型的方法：

- LangChain LLMs 类（接收字符串、输出字符串）
- LangChain 聊天模型（接收消息、输出消息，其中消息是一个langChain定义的类），类名通常以Chat开头
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# 导入消息类型
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

#####
# 直接invoke llm 对象
llm = OpenAI(model_name=model_name)
result = llm.invoke("怎么评价人工智能")
print(result)

#####

#####
chat = ChatOpenAI(model_name=model_name)  # 实例化一个chat对象
# 消息类型：AIMessage, HumanMessage, SystemMessage（分别对应AI回复，用户输入，系统提示）
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate this sentence from English to French. I love programming.")
]

# invoke chat 对象，返回“AIMessage”类型的对象
result = chat.invoke(messages)
print(type(result))  # -> AIMessage
print(result)
#####
```

#### 使用Prompt模板
帮助你通过简单输入构建提示模板：
```python
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

if chat_prompt:
    chat_prompt = ChatPromptTemplate.from_template("What is a good name for a company that makes {product}?")
    print(chat_prompt.format(product="colorful socks"))
    return chat_prompt

# 通过 PromptTemplate 类来创建提示模板
prompt = PromptTemplate(
    input_variables=["product"],  # 这是一个可以修改的参数，会被填入下面的template字符串中
    template="What is a good name for a company that makes {product}?",
)
# 类似string.format()的用法，传入参数填充模板中的变量
print(prompt.format(product="colorful socks"))
return prompt
```

#### 使用链式调用连接模板、模型和解析输出
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def basic_chain_usage_test(use_chain_pipe=True):
    # 基本结构：prompt + model + output parser
    prompt = ChatPromptTemplate.from_template(
        "tell me a short joke about {topic}, use {language}"
    )
    model = ChatOpenAI(model_name=model_name, temperature=0.9)
    output_parser = StrOutputParser()

    if not use_chain_pipe:
        # langchain的各种类型都具有invoke函数
        # prompt类型的invoke函数会将输入的参数填充到prompt模板中，返回一个带有message后的prompt对象
        prompt_value = prompt.invoke({"topic": "ice cream",
                                      "language": "Chinese"})
        print(f"prompt: {prompt_value}")

        # model类型的invoke函数会将prompt value传递给已实例化的model对象，这里model是一个LLM聊天模型，返回一个message对象
        message = model.invoke(prompt_value)
        print(type(message))
        print(f"message: {message}")

        # output parser类型的invoke函数会处理message对象中返回的信息。这里的StrOutputParser是一个基础输出器，将结果转换为字符串
        result = output_parser.invoke(message.content)
        print(type(result))
        print(f"output: {result}")
        return

    # piece together then different components into a single chain using LCEL
    # use | (unix pipe operator) as the operator to connect the components
    # 使用 | 作为连接符号，将不同的组件连接起来，以形成一个链式管道
    chain = prompt | model | output_parser

    result = chain.invoke({"topic": "pigeons",
                           "language": "Chinese"})
    print(f"reply: {result}")
```

### LCEL和可运行管道
上述例子中，类似`chain = prompt | model | output_parser`的语法糖被称为LCEL（[LangChain Expression Language](https://python.langchain.com/docs/expression_language/)）。
每个支持LCEL语法糖的对象，底层都被抽象为了一个“可运行”类：`Runnable`，多个可运行对象的输入输出相连接，组成“可运行管道”，即`RunnableParallel`。它允许接受来自`invoke`函数得到的输入（通常是一个字典类型），然后根据其上子类的实现不同，处理数据并得到输出。
记住下面的几点：

- 使用`RunnableParallel()`可以实例化一个可运行管道对象，该对象存储的数据结构被视为一个字典，可以通过参数赋值来设置字典的键值对，或是直接传入一个字典。
   - 使用`RunnablePassthrough()`函数可以指代`invoke`函数的输入，返回类型是字典
   - 使用`RunnablePassthrough.assign()`函数可以修改`invoke`函数的输入的字典，包括添加新的键值对，返回类型是字典
   - 使用任何**lambda表达式**可以**直接处理**invoke函数的输入，返回的则是lambad表达式的结果
```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

runnable.invoke({"num": 1})

# 输出结果：
{
    'passed': {'num': 1}, 
    'extra': {'num': 1, 'mult': 3}, 
    'modified': 2
}
```

- 在LCEL的链式语法中，如果两个对象之间以`|`连接了，实例化`RunnableParallel()`的步骤可以直接简写为定义一个字典：
```python

retrieval_chain = (
    # 下面的三种写法等价：
    # RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    # RunnableParallel(context=retriever, question=RunnablePassthrough())
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

- 可以使用`itemgetter`函数快速检索一个特定的输入键值（需要线导入名称）
```python
from operator import itemgetter

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke({"question": "where did harrison work", "language": "italian"})
```

- 可以使用`RunnableLambda()`函数，允许调用任何已定义的外部函数来处理输入数据，并返回你自定义的结果。但是，这些函数都必须有且只有一个类型为字典的参数。
```python
def multiple_length_function(_dict):
    return len(_dict["text1"]) * len(_dict["text2"])
    
chain = (
    {
        "x": {"text1": itemgetter("foo"), "text2": itemgetter("bar")}
        | RunnableLambda(multiple_length_function),
    }
    | prompt
    | model
)

chain.invoke({"foo": "abc", "bar": "gah"})
# x的内容是9 (3*3=9)
```

- **可运行管道组成的链是可以随意嵌套的**，你可以将其视为一个带有特定功能的字典，允许多层套娃：
```python

prompt1 = ChatPromptTemplate.from_template(
    "generate a {attribute} color. Return the name of the color and nothing else:"
)
prompt2 = ChatPromptTemplate.from_template(
    "what is a fruit of color: {color}. Return the name of the fruit and nothing else:"
)
prompt3 = ChatPromptTemplate.from_template(
    "what is a country with a flag that has the color: {color}. Return the name of the country and nothing else:"
)
prompt4 = ChatPromptTemplate.from_template(
    "What is the color of {fruit} and the flag of {country}?"
)

color_generator = (
    {
        "attribute": RunnablePassthrough()} 
    	| prompt1 
    	| {
            "color": model | StrOutputParser()
        }
)
color_to_fruit = prompt2 | model | StrOutputParser()
color_to_country = prompt3 | model | StrOutputParser()
question_generator = (
    color_generator 
    | {
        "fruit": color_to_fruit, 
        "country": color_to_country
    } 
    | prompt4
)
```
### 外部参数绑定
`Runnable.bind()`的函数允许你在外部添加可运行管道对象的参数。这里我们以为模型对象绑定参数为例。
最常用的功能是，为模型绑定`stop`参数，可以指定LangChain在匹配到到LLM的某些输出后终止LLM的后续输出，直接返回已输出的内容：
```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write out the following equation using algebraic symbols then solve it. Use the format\n\nEQUATION:...\nSOLUTION:...\n\n",
        ),
        ("human", "{equation_statement}"),
    ]
)

runnable = (
    {"equation_statement": RunnablePassthrough()}
    | prompt
    | model.bind(stop="SOLUTION")
    | StrOutputParser()
)
print(runnable.invoke("x raised to the third plus seven equals 12"))

### 不加bind(stop="SOLUTION")的结果：
EQUATION: x^3 + 7 = 12

SOLUTION:
Subtracting 7 from both sides of the equation, we get:
x^3 = 12 - 7
x^3 = 5
### 加上bind(stop="SOLUTION")的结果：
EQUATION: x^3 + 7 = 12

```
#### 附加函数调用功能
如果你使用的是OPENAI等支持函数调用的模型，你还可以定义函数（这要求LLM以固定的格式输出结果），并使用`bind`函数将其绑定到LLM对象或聊天模型对象中。
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

def prompt_chain_with_input_and_output_handle():

    prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
    model = ChatOpenAI(model_name=model_name, temperature=0.9)

    # 为model定义函数调用，这回要求model返回一个符合该function定义的，json格式的结果
    functions = [
        {
            "name": "joke",
            "description": "A joke",
            "parameters": {
                "type": "object",
                "properties": {
                    "setup": {"type": "string", "description": "The setup for the joke"},
                    "punchline": {
                        "type": "string",
                        "description": "The punchline for the joke",
                    },
                },
                "required": ["setup", "punchline"],
            },
        }
    ]
    # 这个管道的定义使得当用户输入"ears"时，map_对象中实际得到的数据是{"foo": "ears"}
    map_ = RunnableParallel(foo=RunnablePassthrough())
    # chain = (
    #         map_
    #         | prompt
    #         | model.bind(function_call={"name": "joke"}, functions=functions)
    #         | JsonKeyOutputFunctionsParser(key_name="setup")  # 使用Parser从json中提取想要输出key（如：setup）字段
    # )
    chain = (
            map_
            | prompt
            | model.bind(function_call={"name": "joke"}, functions=functions)  # 绑定函数和指定函数调用
            | JsonOutputFunctionsParser()  # 使用Parser提取整个json
    )

    result = chain.invoke("ears")
    print(result)
```
#### 


### 模型运行时配置修改
如果你不想在某些模型运行时使用全局的语言模型配置参数（模型名称，温度，最大token限制等），可以使用以下方法：
首先，在实例化model时指定可以覆盖修改的参数，使用`ConfigurableField`，其中id字段要与你使用的模型后端匹配，而名称和描述可以自定义。
```python
from langchain.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)
```
使用模型时（包括在LCEL中使用时），用函数即可覆盖配置：
```python
model.with_config(
    configurable={
        "llm_temperature": 0.9
    }
).invoke("pick a random number")
```
还可以使用`configurable_alternatives`指定一套可替代的配置方案，详细请参考：[Configure chain internals at runtime | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/how_to/configure#configurable-alternatives)

### 加入对话记忆
如果需要进行多轮对话，则有必要将每次对话的内容记录下来，并在下一次提示时提供历史消息。
可以通过在定义prompt时添加关于历史的`MessagesPlaceholder`占位符来实现：
```python
def test_memory_chain():
    from operator import itemgetter

    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    model = ChatOpenAI(model_name=model_name)  # 使用聊天LLM模型和聊天提示模板
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot"),  # 这是系统提示
            MessagesPlaceholder(variable_name="history"),  # 这是一个占位符，用于将对话历史填充到模板中
            ("human", "{input}"),  # 这是用户输入
        ]
    )
    memory = ConversationBufferMemory(return_messages=True)  # 添加一个ConversationBufferMemory对象，用于存储对话历史
    mem = memory.load_memory_variables({})  # 初始化memory，返回值为这样的字典：{"history": []}
    print(f"init memory: {mem}")
    chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            )  # 从memory中加载键值为"history"的变量，然后将其通过管道输入给下一步的prompt
            | prompt
            | model
    )

    inputs = {"input": "hi im bob"}
    response = chain.invoke(inputs)
    print(f"first response: {response}")

    memory.save_context(inputs, {"output": response.content})  # 一次chain invoke后，需要更新memory中的context
    mem = memory.load_memory_variables({})  # 重新加载memory，返回值中会有已经存储的对话历史
    print(f"memory after first invoke: {mem}")

    # 再次进行对话，这次会将对话历史填充到prompt中
    inputs = {"input": "can you tell what is my name?"}
    response = chain.invoke(inputs)
    print(f"second response: {response}")
```
记住，`MessagesPlaceholder`占位符不止用于记忆历史，可以记录任何外部内容，当你需要从外部添加信息时也可以使用。
## LangChain实现检索增强（RAG）
检索增强的含义是，LLM可能不知道用户需要的特定数据，因此需要一个程序检索外部数据，然后在执行生成步骤时将其传递给LLM。
检索增强的步骤通常是将外部源数据（通常是各种文档）进行加载、拆分处理，然后转化为向量（被称为“嵌入”），这将保存数据中的语义信息，去除干扰，以方便语言模型理解，最后存储在一种介质中（如向量数据库），当模型需要时，通过一种方法从介质中“检索”数据。
### 加载文档
LangChain支持各种文档格式的读取，如果你的数据只是简单的文本格式（如markdown和txt），可以直接使用`TextLoader`获得一份存储在内存的文档对象。
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./index.md")
loader.load()

# 文档对象的数据结构示例：
[
    Document(
        page_content='---\nsidebar_position: 0\n---\n# Document loader ... ',  
        metadata={'source': '../docs/docs/modules/data_connection/document_loaders/index.md'}
    )
]
# 每个文档包含两个对象，page_content是文本形式存储的内容，metadata包含文本的元数据，比如存储路径source
```
关于所有支持的文档格式和相关库，参考：[https://python.langchain.com/docs/integrations/document_loaders/](https://python.langchain.com/docs/integrations/document_loaders/)
下面我们会介绍其他针对常用格式的Loader：
#### CSV表格
```python
from langchain_community.document_loaders.csv_loader import CSVLoader

# 不同的CSV文档可能有不同的分隔符，读取时可以指定，并且可以指定表头的名称
loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['MLB Team', 'Payroll in millions', 'Wins']
})
data = loader.load()

# 读取CSV文件时，可以指定source_column参数将文本的元数据从默认的“文件路径”转化为每一列的表头名称
# 在文本-问答任务中这种处理也许有用。
loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv', source_column="Team")
```
#### HTML网页
```python
from langchain_community.document_loaders import UnstructuredHTMLLoader

# 第一种方法是使用UnstructuredHTMLLoader，这将直接提取html中的纯文本内容到page_content中，而忽略其他的所有内容。
loader = UnstructuredHTMLLoader("example_data/fake-content.html")
data = loader.load()

# 另一种方法是使用BeautifulSoup4
from langchain_community.document_loaders import BSHTMLLoader

# 这将使得html中的纯文本内容被提取到page_content中，而网页标题（如果有）被写入元数据中的'title'字段
loader = BSHTMLLoader("example_data/fake-content.html")
data = loader.load()
data
```
#### JSON
`JSONLoader`类定义了加载JSON格式文件的方法。LangChain允许你从一个json文件中读取出多个不同的文档，并且可以指定读取哪些键对应的值的内容。这是通过一个名为`jq`的python库实现的。
在实例化`JSONLoader`的时候，可以传入`jq_schema`参数，其中指定了要从json文件的哪个键中提取对象。`jq_schema`的语法与lua字典类似，直接使用`.`可以访问子级键值对，如果对象是一个列表，使用`[]`可以提取整个列表中的每个对象，这会让`JSONLoader`返回多个`Document`对象组成的列表，其在列表中的索引序号将被写入元数据中。
```python
loader = JSONLoader(
    file_path='./example_data/facebook_chat.json',
    jq_schema='.messages[].content',
    text_content=False)

data = loader.load()

# 示例：如果json长这这样：
    {
     'joinable_mode': {'link': '', 'mode': 1},
     'messages': [{'content': 'Bye!',
                   'sender_name': 'User 2',
                   'timestamp_ms': 1675597571851},
                  ...
                  {'content': 'Goodmorning! $50 is too low.',
                   'sender_name': 'User 2',
                   'timestamp_ms': 1675577876645}],
     'title': 'User 1 and User 2 chat'
    }
# data中的内容将会是：
    [
        Document(
            page_content='Bye!',
            metadata={
                'source': '/Users/avsolatorio/WBG/langchain/docs/modules/indexes/document_loaders/examples/example_data/facebook_chat.json', 
                'seq_num': 1
            }
        ),
        ...
        Document(
            page_content='Goodmorning! $50 is too low.', 
            metadata={
                'source': '/Users/avsolatorio/WBG/langchain/docs/modules/indexes/document_loaders/examples/example_data/facebook_chat.json', 
                'seq_num': 10
            }
        ),
    ]
```
常见的json格式对应的`jq_schema`语法如下：
```json
JSON        -> [{"text": ...}, {"text": ...}, {"text": ...}]
jq_schema   -> ".[].text"

JSON        -> {"key": [{"text": ...}, {"text": ...}, {"text": ...}]}
jq_schema   -> ".key[].text"

JSON        -> ["...", "...", "..."]
jq_schema   -> ".[]"
```
如果你的json文档格式是一个后缀名为`.jsonl`的 JSON Lines文件，这种文件中的每一行是一个符合json格式的字符串，可以在一个文件中存储多个json对象。对于这种文件，在加载时`json_lines=True`参数即可。指定的`jq_schema`索引会对每一行的json对象起作用。
```python
loader = JSONLoader(
    file_path='./example_data/facebook_chat_messages.jsonl',
    jq_schema='.content',
    text_content=False,
    json_lines=True)

data = loader.load()
```
你还可以自定义要将哪些json文件的内容写入文档的metadata，即元数据字段中，这需要通过传入一个`metadata_func`来实现：
```python
# metadata_func有两个参数，第一个参数是源数据的json对象，第二个参数对应其在文档中的元数据对象。
def metadata_func(record: dict, metadata: dict) -> dict:

    # 将原json的"sender_name"字段的值写入文档元数据的"sender_name"字段
    metadata["sender_name"] = record.get("sender_name")
    # 你也可以修改已有元数据的内容，如这里修改了所有文档中元数据的路径，
    # 将它们路径中 /langchain 前的部分都删除了。
    if "source" in metadata:
        source = metadata["source"].split("/")
        source = source[source.index("langchain"):]
        metadata["source"] = "/".join(source)

    return metadata


loader = JSONLoader(
    file_path='./example_data/facebook_chat.json',
    jq_schema='.messages[]',
    content_key="content",
    metadata_func=metadata_func
)

data = loader.load()
```
#### Markdown
```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

markdown_path = "../README.md"

# 如果加入参数mode="elements"，将会保留所有markdown语法的符号（如```)，否则只会提取文本
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
data = loader.load()
```

#### PDF
有多种方法提取PDF文件的内容（基于底层的不同），详情参见：[https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)
这里介绍使用`PyPDFLoader`读取文件，它返回一个`Document`的列表，每个对象都是pdf中一页的内容对应的文档，页码信息会被存储到元数据中。
```python
# 确保先安装了pypdf
# pip install pypdf

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()
```
如果想从读取图像中的文字，可以使用 `rapidocr-onnxruntime` ,这是一个OCR包，可以识别图像中的文本并存储到文档中。安装完成后加入参数`extract_images=True`即可
```python
# pip install rapidocr-onnxruntime

loader = PyPDFLoader("https://arxiv.org/pdf/2103.15348.pdf", extract_images=True)
pages = loader.load()

```
如果对pdf文档的结构无所谓，使用非结构化数据可以只提取所有的纯编码数据到文档。
```python
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader("example_data/layout-parser-paper.pdf") # 同样，加入mode="elements"可以保留元素
data = loader.load()
```
#### Python源码
可以使用`PythonLoader`类直接加载python代码为文档。
```python
from langchain_community.document_loaders import PythonLoader
```
#### 批量加载文档
从一个目录下加载目录中的所有文件。默认情况下，所有文件都以非结构化的方式（直接读取文本）加载到文档对象中，返回一个列表。
`DirectoryLoader`相关参数：

- `glob` : 用来控制加载符合哪些路径名和扩展名的文件
- `show_progress=True`：显示进度条，需要先安装`tqdm`库
- `use_multithreading=True`：使用多线程加快速度
- `loader_cls=TextLoader`：指定所有文件使用某个特定的文档加载器类
- `silent_errors=True`：如果加载过程中出现错误（如编码问题），忽略错误并继续加载（默认将会终止加载，并不会返回已加载的内容到内存）
- `loader_kwargs={}` ：其他加载时的参数，以字典形式传入
   - 要在加载过程中自动检测文件编码，传入`{'autodetect_encoding':True}`
```python
from langchain_community.document_loaders import DirectoryLoader

text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()
```

### 分割文档
处理长文本时，有必要将文本分割成块。尽管这听起来很简单，但这里有很多潜在的复杂性。理想情况下，您希望将语义相关的文本片段放在一起。“语义相关”的含义可能取决于文本的类型。
关于LangChain支持的所有分割器类型，参考：[Text Splitters | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
这里只介绍两种：基于字符（单词）拆分和基于语义拆分。
#### 按照字符拆分文档
最简单的方法，按照字符数切分文档，一般只适用于西文。
```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# 这里传入的是纯文本的列表对象，将会分别拆分每个对象。
# metadata是可选的，会一一对应地写入被拆分的文档中的元数据。
# 返回值是Document类型
documents = text_splitter.create_documents(
    [doc1, doc2, ...], 
    metadatas=[meta1, meta2, ...]
)

print(documents[0])
```

#### 按照单词拆分
推荐普遍情况下有一堆长文本时使用，希望尽可能的将相近的段落和语句放在一个分块中，适用于各种语言的编码。
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([doc])
print(texts[0])
print(texts[1])
```

#### 根据语义拆分
这将根据文本中的语义相似性拆分文本，语义更相似的，更有可能放在一起。需要文本嵌入编码模型的介入。
这里使用OPENAI的文本嵌入api作为语义相似度检查的例子，使用前需要先配置好相关api环境变量:
```python
!pip install --quiet langchain_experimental langchain_openai

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

text_splitter = SemanticChunker(OpenAIEmbeddings())

docs = text_splitter.create_documents([doc])

print(docs[0].page_content)
```

### 编码文档为向量嵌入
编码文档到向量嵌入非常简单，只需制定好模型，然后传入字符串格式的文本即可。前面小节中加载的文档对象，可以从`page_content`字段中取出字符串值。
可以使用的文本-向量编码模型及说明文档参考：[Text embedding models | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/text_embedding/)
下面会使用在线调用OPENAI的文档编码模型为例子。
```python
from langchain_openai import OpenAIEmbeddings

# 如果你在全局设置了环境变量，则无需传入api key的参数
embeddings_model = OpenAIEmbeddings(openai_api_key="...")

# 这里直接使用字符串作为例子
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
len(embeddings), len(embeddings[0])
```
可以使用文本-向量编码模型快速比较不同文本直接的相似度，如：
```python
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
embedded_query[:5] # 输出embed_query传入的文本与embeddings_model中已编码的5个文本之间（各个）的相似度。

# 输出举例：
[0.0053587136790156364,
 -0.0004999046213924885,
 0.038883671164512634,
 -0.003001077566295862,
 -0.00900818221271038]
```
### 创建向量存储器
下面的例子使用 FAISS 向量数据库，该数据库利用了Facebook AI相似性搜索（FAISS）库。
也可以选择其他不同的向量存储器库，参考：[Vector stores | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/vectorstores)
先安装相关依赖：
```python
pip install faiss-cpu
```
相关步骤如下：

- 加载一个示例文档
- 使用分割器将文档分割为固定长度的字符串
- 对于每个字符串切片，进行向量嵌入编码（使用`OpenAIEmbeddings`），然后使用`FAISS.from_documents`存储到数据库对象。
```python
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 读取源文件
raw_documents = TextLoader('../../../state_of_the_union.txt').load()
# 分割文本为若干块，并转换为对应的文档对象
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
# 创建向量数据库，传入两个参数：文档对象列表 和 嵌入模型的实例化对象
db = FAISS.from_documents(documents, OpenAIEmbeddings())
```
#### 相似性搜索
创建好向量数据库后，就可以根据相似性，搜索出需要的文本内容，然后解码为可读的文本。
`similarity_search`函数将会查询输入文本，找出最相关的多个文档，并返回其解码后的文档内容。
```python
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
```
如果输入不是可读文本，而是一个已经编码好的向量，使用函数`similarity_search_by_vector `
```python
embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)
```
#### 嵌入缓存
使用带有缓存的文本嵌入类，可以使得从重复的文本创建向量数据库时，不再调用模型编码而直接查询缓存中的结果。这可能可以减少模型的调用量。
通常，嵌入缓存和向量数据库一起使用，当查询缓存命中时，LangChain直接从缓存中构建向量数据库，而不是从原始文本编码开始。下面是一个例子：

- 为了使用缓存，首先需要引入名称`CacheBackedEmbeddings`，并指定一个用于底层编码嵌入向量的模型，这里仍然使用`OpenAIEmbeddings`
- 创建一个缓存的存储位置，下面的例子中，缓存位于本地文件目录`"./cache/"`中。
   - 你也可以将缓存设为存储在内存中（如果你的内存足够），使用如下的语句：
   - `from langchain.storage import InMemoryByteStore`
   - `store = InMemoryByteStore()`
- 指定一个命名空间参数`namespace`，命名空间比较重要，它保证了你使用 不同的嵌入模型 编码 同一文本 时不会触发缓存命中（否则这容易导致不同模型间的冲突）。一般来说，将其指定为嵌入模型的名称即可。
```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

underlying_embeddings = OpenAIEmbeddings()

store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)
```
完成后，你可以使用`cached_embedder`代替`OpenAIEmbeddings()`等底层嵌入模型创建向量数据库，它们会自动在需要时使用缓存。
使用`list(store.yield_keys())`可以查看缓存中的主键。
### 检索器
上面的内容介绍了简单的向量数据库语义搜索，它们直接返回相应的文本。但实际使用时，往往需要结构化的数据，因此需要包装检索器。
LangChain提供了许多内置的检索器对象，可以方便地完成多查询、排序、时间加权等功能。关于所有的检索器类，参考：[Retrievers | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/data_connection/retrievers/)
#### 基础检索器 - 基于向量数据库
可以直接从向量数据库对象构建检索器
```python
# 省略了前置步骤……
db = FAISS.from_documents(docs, embeddings)

# 创建检索器
retriever = db.as_retriever()

# 查询相关的文档(默认以相似性搜索)
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```
在创建检索器时，可以指定其使用的检索方法， 如最大边缘相关性搜索（MMR）,前提是对应的向量数据库对象支持这种检索。
在`search_kwargs`参数中可以使用字典传入其他参数，如相似性分数阈值（只返回分数高于该阈值的文档），或在指定top k（只返回k个最相关的文档）等。
```python
# 使用MMR检索
retriever = db.as_retriever(
    search_type="mmr"
)
# 使用带有阈值的相似性搜索
retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)
# 指定Top k
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)
```
#### 多查询检索器 MultiQueryRetriever
这种查询器的目标时为了提高语义检索的鲁棒性。由于自然语言存在一定的冗余性和歧义，当查询措辞发生细微变化，或者嵌入不能很好地捕捉数据的语义时，检索可能会产生不同的结果。这通常会使得检索到的文档范围缩小（即同样的问题，只是改了一个措辞，就查不到相关文档了）。为了解决这样的问题，可以使用多查询：即每次查询时，使用多个语义相近的字符串代替单一字符串，进行多次查询，并统合最终的结果。
在LangChain中，自然地使用LLM来生成“多个语义相近的查询”，这样用户只需像往常一样输入一个查询语句，但可以达到多查询的效果。
```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbedding

# 构建一个向量数据库（这里使用的存储后端是Chroma）
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=docs, embedding=embeddings)

llm = ChatOpenAI(temperature=0) # 实例化一个LLM对象，作为多查询生成器

# 构建多查询检索器
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

import logging
# 增加一个logging对象，以打印生成出的多查询内容
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

question = "What are the approaches to Task Decomposition?"
unique_docs = retriever_from_llm.get_relevant_documents(query=question)

# 输出示例：这里是log输出，即LLM为查询question生成的相近语义问题
INFO:langchain.retrievers.multi_query:Generated queries: 
[
 '1. How can Task Decomposition be approached?', 
 '2. What are the different methods for Task Decomposition?', 
 '3. What are the various approaches to decomposing tasks?'
]
```
作为生成多查询的基础，你可以修改底层LLM的配置。上述例子默认使用一个基本的LLM对话。你可以使用基本的LangChain链来将其自定义为带有特定prompt的LLM对话：
> 注意：为了使得检索器能正确找到LLM输出的多查询内容，在自定义LLM chain时需要将其输出格式化解析为键值对的形式，参考下面的例子。

```python
# 定义一个LLM输出格式解析器，将输出安装换行符分割，存储到一个lines的字典里
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

output_parser = LineListOutputParser()

# 自定义撰写提示，要求LLM输出5个多查询语句
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
llm = ChatOpenAI(temperature=0)
# 构造LLM Chain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

# 使用自定义的LLM Chain 创建多查询检索器，其中，parser_key函数指定了
# 检索器从输出数据的哪个key中获取最终的多查询语句列表
retriever = MultiQueryRetriever(
    retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
) 
```

#### 上下文压缩检索器
有的时候，知识库中的每个文档都是长文本，而当检索时，用户只需要文档的概括内容，或是其中与查询最相关的内容，而非整个文档。通常这需要对查询结果进行后处理来实现，不过LangChain为此包装了压缩检索器，可以一并完成查询结果过滤的过程。
上下文压缩检索器基于基本检索器，在使用前，先从向量数据库构建一个基本检索器。

- 可以使用一个LLM实例来决定过滤掉结果文档中的哪些内容，这被称为`LLMChainFilter`：
```python
from langchain.retrievers.document_compressors import LLMChainFilter

... # 省略知识库数据处理的过程
# 创建基本检索器
retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()

# 需要一个LLM作为过滤器的后端
llm = OpenAI(temperature=0)
# 创建LLM过滤器
_filter = LLMChainFilter.from_llm(llm)

# 构建上下文压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, 
    base_retriever=retriever
)

# 进行查询，并返回压缩后的查询结果
compressed_docs = compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
```

- 但是，使用LLM作为压缩和过滤的后端，会让每次查询时都增加许多额外的LLM调用，如果想要节约成本，可以直接使用嵌入模型，查询结果文档内部语句的相似度，从而返回结果。这种情况下，使用`EmbeddingsFilter`
```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

... # 省略知识库数据处理的过程
# 创建基本检索器
retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()

embeddings = OpenAIEmbeddings()
# 构建向量相似度过滤器
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

# 构建上下文压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, 
    base_retriever=retriever
)

# 进行查询，并返回压缩后的查询结果
compressed_docs = compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
```
LangChain还提供多种过滤器，如去除冗余的向量嵌入等。并且，分割文本的工具类也可以视为过滤器。
当你需要对输出文本进行一些流水线处理时（如先分割，在去除冗余，最后提取高相似度文本），可以使用`ocumentCompressorPipeline`构建一个**流水线压缩器**，LangChain将会自动按步骤处理中间文档，只输出最终结果。
```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ") # 文本分割
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings) # 去除冗余
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76) # 相似度比较

# 构建流水线压缩器
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
# 构建上下文压缩检索器
)compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
```

#### 多向量检索器 MultiVector Retriever
注意：这与多查询不同，多向量的意思时，每个源文档在存入知识库（向量数据库）时，具有多个对应的向量嵌入，而它们的特性不同（如分别描述了摘要，重点问题。解决方案等文档内容的不同方面）。然后检索时，按照查询的语义偏好查找同一文档的不同向量嵌入。
```python
TODO
```
#### 带有时间加权的检索器
在检索语义时，也考虑同样的文档或知识内容 _上一次被访问的时间_。
`语义相似度 + (1.0 - decay_rate) ^ 距离上次检索过去的时间（小时）`
`decay_rate`被设定为0-1之间的值，越接近0，意味着在检索时，新鲜时间对文档检索结果的影响越大，反之亦然。因此`decay_rate`为1时，时间加权相当于不存在；`decay_rate`为0时，则无论文档上一次访问的时间距离多久，文档都有相同的新鲜度。
```python
from datetime import datetime, timedelta

import faiss
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 定义嵌入模型
embeddings_model = OpenAIEmbeddings()
# 初始化一个空的向量数据库
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

# 修改decay_rate来测试时间加权的检索效果
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.01, k=1
)

yesterday = datetime.now() - timedelta(days=1)
# 添加一个字符串作为文档，设置上次检索时间为昨天
retriever.add_documents(
    [Document(page_content="hello world", metadata={"last_accessed_at": yesterday})]
)
# 再添加一个字符串，不设定上次检索时间
retriever.add_documents([Document(page_content="hello foo")])

retriever.get_relevant_documents("hello world")
# 在decay_rate接近0的时候会返回"hello world"，而接近1时会返回"hello foo"
```
#### 将检索器包装为LLM Chain
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain

... # 省略知识库数据处理的过程
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

... # 有两种方式构建带有检索器的链，一是在已有链的基础上使用create_retrieval相关函数
chain = prompt | llm | output_parser
retrieval_chain = create_retrieval_chain(retriever, chain)

# 二是直接在链的RunnableParalle管道中传入retriever对象（也是Runnable的）
retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)
```
#### 将检索器包装为智能体工具
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

... # 省略知识库数据处理的过程
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

# 创建检索器工具
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
```


## LangChain Agent
### 核心概念

- `AgentAction`：`dataclass`，表示Agent采取的一个动作，通常是使用工具
> 属性：
> - `tool` 应该调用的工具的名称
> - `tool_input` 该工具的输入
> 
实际使用时通常为带Log的`AgentActionMessageLog`

- `AgentFinish`：智能体准备返回给用户时的最终结果
> 属性：
> - `return_values` 字典集，包含所有最终输出
>    - 通常有一个名为`output`的key，表示回复给用户的字符串

- Intermediate Steps：记录所有智能体的历史动作，和与本次运行相关的输出
   - 通常被保存为一个列表数据类型：`List[Tuple(AgentAction, Observation)]`
      - 为了最大化框架的灵活性，`Observation`的类型可以是“任意”（Any），但通常是字符串

### 智能体链基类：Agent
这是一个在LangChain的链 的基础上构建的类，功能是让大模型决策下一步的行动。
> 再次提醒：记住 _LangChain的链_ 的概念其实只是**包装好**一个提示大模型并解析输出的过程，并允许在这个过程中间插入额外的数据处理步骤。

既然Agent基于Chain类型构建，因此底层上可能使用两种LangChain提示策略包装，即 **纯LLM** 或 **聊天模型**（Chat Models），根据该类型的不同，Agent具有不同的 “预期类型” （Intended Model Type）
同时，LangChain为常用功能包装好了若干智能体类，它们支持的功能各不相同。
### 智能体执行器类：AgentExecutor
程序运行时，实际执行智能体功能的躯壳（运行时），进行实际调用智能体、执行它选择的动作、将动作输出传递回代理的操作，并重复这些操作。
AgentExecutor在运行时可以处理的问题：

- 如果Agent选择了不存在的工具
- 如果Agent的输出无法解析为合法的工具调用或最终输出
- 如果Agent选择的工具在运行时发生错误
- 记录Agent在所有级别上的运行日志（可以输出到控制台或使用[LangSmith](https://python.langchain.com/docs/langsmith)记录）

### 预置智能体的使用
#### 预期类型是Chat Models的内置智能体

- [OpenAI Tools](https://python.langchain.com/docs/modules/agents/agent_types/openai_tools)
   - 使用OPENAI支持“工具调用”的模型，这些模型经过了微调，增强了检测何时调用函数，并输出 **应该传递给该函数的参数** 作为响应的能力。这使得LangChain可用利用这一特性调用多个函数，以此来选择AgentAction，进行动作规划
      - 和下面的“函数调用”的区别是，“函数调用”只支持推理单个函数，而“工具调用”支持一个或多个函数。
      - 只在OPENAI最新的模型（`gpt-3.5-turbo-1106`或`gpt-4-1106`）以后支持。
      - 因为“工具调用”是“函数调用”的上位替代，因此OPENAI认为“函数调用”是废旧的功能，不建议再使用旧的OpenAI Functions，而是统一使用OpenAI Tools构建智能体。
   - 需要导入的引用：
```python
# 支持OPENAI的agent
from langchain.agents import AgentExecutor, create_openai_tools_agent
# 底层的聊天模型
from langchain_openai import ChatOpenAI

# 其他适用于创建工具，网络通信等的包都省略了……
```

   - 创建智能体
```python
# 定义可用的工具列表，这里是一个示例，导入Langchain内置的其中一个网络搜索工具
from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results=1)] # 通常工具定义为全局变量

def new_openai_tools_agent():
    # 定义一个提示词，可自定义，这里从一个langchain仓库中获取
    from langchain import hub
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    # 实例化一个聊天模型对象
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
    
    # 实例化智能体对象
    agent = create_openai_tools_agent(llm, tools, prompt)

	return agent
```

   - 实例化运行时并执行智能体
```python
agent = new_openai_tools_agent()
# 创建智能体执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# 输入内容
input = {"input": "what is LangChain?"}
agent_executor.invoke(input)
```

   - 默认Input只包含用户的一个字符串键值对，但鉴于OpenAI Tools智能体是基于聊天模型的，可再输入中添加任意的对话历史，这和聊天模型的链调用一样：
```python
from langchain_core.messages import AIMessage, HumanMessage

agent_executor.invoke(
    {
        "input": "what's my name? Don't use tools to look this up unless you NEED to",
        "chat_history": [
            HumanMessage(content="hi! my name is bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
        ],
    }
)
```

- [OpenAI Functions](https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent)	
   - 与OpenAI Tools一致，使用OPENAI模型内部的“函数调用”（Function calling）功能
   - 使用时，只需修改上述导入的函数名`create_openai_tools_agent`为`create_openai_functions_agent`即可，其他配置完全相同。
   - 也适用于其他开源模型提供的与OpenAI Functions兼容格式的API接口，因为这些模型通常没有随着OPENAI而更新api，所以使用时选择旧版的`create_openai_tools_agent`
   - 需要导入的引用：
```python
# 支持OPENAI模型的agent
from langchain.agents import AgentExecutor, create_openai_functions_agent
# 底层的聊天模型
from langchain_openai import ChatOpenAI

prompt = hub.pull("hwchase17/openai-functions-agent")
# 其他的部分省略，与OpenAI Tools 基本一致
```

- [Structured Chat](https://python.langchain.com/docs/modules/agents/agent_types/structured_chat)
   - Structured Chat（结构化对话）Agent与OpenAI Tools的原理基本相同，唯一的区别是，因为不依赖模型自身经过微调而输出函数调用的能力，结构化对话Agent通过**_ 引导大模型输出有效的 格式化（或称序列化）函数参数_** 来解析并调用工具，再将结果反馈给大模型，由此完成一步的AgentAction。
   - 默认的序列化方法是JSON。
   - 需要导入的引用：
```python
from langchain.agents import AgentExecutor, create_structured_chat_agent
# 底层的聊天模型仍然可用使用OPENAI模型
from langchain_openai import ChatOpenAI
```

   - 实例化Agent
```python
# 定义可用的工具列表，这里是一个示例，导入Langchain内置的其中一个网络搜索工具
from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results=1)] # 通常工具定义为全局变量

def new_structured_chat_agent():
    # 定义一个提示词，可自定义，这里从一个langchain仓库中获取
    from langchain import hub
    prompt = hub.pull("hwchase17/structured-chat-agent")
    
    # 实例化一个聊天模型对象
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # 实例化智能体对象
    agent = create_structured_chat_agent(llm, tools, prompt)

	return agent
```

   - 因为非“工具调用”的大模型可能不按照格式化要求输出，在实例化Agent时，可以通过参数控制是否自动处理可能的解析错误。
```python
agent = new_structured_chat_agent()
# 创建智能体执行器
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
# 输入内容
input = {"input": "what is LangChain?"}
agent_executor.invoke(input)
```

   - Structured Chat Agent内部将会以下面的方式输出函数调用的过程，但最后返回用户的结果output仍然是自然语言。
```python
> Entering new AgentExecutor chain...
Action:
```
{
  "action": "tavily_search_results_json",
  "action_input": {"query": "LangChain"}
}
```
# 这是工具调用返回的结果，这行注释是方便你理解内容的，并不在实际的输出中出现
[{'url': 'https://www.ibm.com/topics/langchain', 'content': 'LangChain is essentially a library of abstractions for Python and Javascript...'}]
Action:
```
{
  "action": "Final Answer",
  "action_input": "LangChain is an open source orchestration framework ..."
}
```

> Finished chain.

{'input': 'what is LangChain?',
 'output': 'LangChain is an open source orchestration framework ....'}

```

- [JSON Chat](https://python.langchain.com/docs/modules/agents/agent_types/json_agent)
   - 与Structured Chat Agent基本一致，唯一的区别是 **_JSON Chat Agent仅支持 一个输入参数 的工具调用情况_**，如果你的Agent需要调用**含多个参数的**工具，请使用Structured Chat Agent
   - 强制模型使用JSON格式输出函数调用参数。适用于专门对JSON格式进行微调或引导的模型。
   - 使用时，只需将上述步骤中导入的名称`create_structured_chat_agent`替换为`create_json_chat_agent`
   - 如果观察中间输出，会发现JSON Chat与Structured Chat的区别是在输出序列化文本时不再加入markdown标记代码段的符号` ``` ``` `，整个对话的全局都视文本为JSON格式。
```python
from langchain.agents import AgentExecutor, create_json_chat_agent
# 底层的聊天模型仍然可用使用OPENAI模型
from langchain_openai import ChatOpenAI
```
> 别忘了以上所有基于对话模型的Agent都可以在输入中简单地加入对话历史。

#### 预期类型是纯LLM的内置智能体

- [XML](https://python.langchain.com/docs/modules/agents/agent_types/xml_agent)
   - 对XML特攻，要求模型以XML格式输出函数调用信息，适合于擅长XML处理的模型。与**_JSON Chat Agent一样，仅支持单一参数的工具调用_**
   - 需要导入的引用：
```python
from langchain.agents import AgentExecutor, create_xml_agent
# 使用 Anthropic’s Claude 模型
from langchain_community.chat_models import ChatAnthropic
```

   - 实例化并运行Agent
```python
# 定义可用的工具列表，这里是一个示例，导入Langchain内置的其中一个网络搜索工具
from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results=1)] # 通常工具定义为全局变量

def new_xml_agent():
    # 定义一个提示词，可自定义，这里从一个langchain仓库中获取
    from langchain import hub
    prompt = hub.pull("hwchase17/xml-agent-convo")
    
    # 实例化一个claude-2模型查询对象
	llm = ChatAnthropic(model="claude-2")
    
    # 实例化智能体对象
    aagent = create_xml_agent(llm, tools, prompt)

	return agent

agent = new_xml_agent()
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "what is LangChain?"})
```

   - 中间输出将会类似下面这样：
```xml
> Entering new AgentExecutor chain...

<tool>tavily_search_results_json</tool>
<tool_input>what is LangChain?
  
<!--这是工具调用返回的结果，这行注释是方便你理解内容的，并不在实际的输出中出现-->
[{'url': 'https://aws.amazon.com/what-is/langchain/', 'content': 'What Is LangChain? ......'}] 
  
<final_answer>LangChain is an open source framework .....</final_answer>

> Finished chain.
```

   - 底层不是对话模型的接口，因此模型不会认为Agent的一次调用是一场对话的一部分。但是，你仍然可以加入“历史消息”，只不过需要将历史的输入 从使用对话模型的Message类 改为 输入一个简单的字符串：
```python
agent_executor.invoke(
    {
        "input": "what's my name? Only use a tool if needed, otherwise respond with Final Answer",
        # Notice that chat_history is a string, since this prompt is aimed at LLMs, not chat models
        "chat_history": "Human: Hi! My name is Bob\nAI: Hello Bob! Nice to meet you",
    }
)
```

- [ReAct](https://python.langchain.com/docs/modules/agents/agent_types/react)
   - 使用“ReAct"策略，而不是某种格式化文本来引导Agent执行函数调用，具体参见：[ReAct论文](https://react-lm.github.io/)
   - 使用方法和JSON Chat Agent以及 XML Agent一致，同样，让ReAct**_调用的每个工具只能含有单一参数_**
   - 需要导入的引用
```python
from langchain.agents import AgentExecutor, create_react_agent
# 注意使用非聊天Chat的模型
from langchain_openai import OpenAI
```

   - 实例化并运行Agent
```python
# 定义可用的工具列表，这里是一个示例，导入Langchain内置的其中一个网络搜索工具
from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results=1)] # 通常工具定义为全局变量

def new_react_agent():
    # 定义一个提示词，可自定义，这里从一个langchain仓库中获取
    from langchain import hub
    prompt = hub.pull("hwchase17/react")
    
    # 实例化一个模型对象
	llm = OpenAI()
    
    # 实例化智能体对象
    agent = create_react_agent(llm, tools, prompt)

	return agent

agent = new_react_agent()
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "what is LangChain?"})
```

   - 中间输出将会类似下面这样：
```python

> Entering new AgentExecutor chain...

I should research LangChain to learn more about it.
Action: tavily_search_results_json
Action Input: "LangChain"
# 这是工具调用返回的结果，这行注释是方便你理解内容的，并不在实际的输出中出现
[{'url': 'https://www.ibm.com/topics/langchain', 'content': 'LangChain is essentially a library of .....'}] 

I should read the summary and look at the different features and integrations of LangChain.
Action: tavily_search_results_json
Action Input: "LangChain features and integrations"[{'url': 'https://www.ibm.com/topics/langchain', 'content': "LangChain provides integrations for over 25 different embedding methods ....."}] 

I should take note of the launch date and popularity of LangChain.
Action: tavily_search_results_json
Action Input: "LangChain launch date and popularity"[{'url': 'https://www.ibm.com/topics/langchain', 'content': "LangChain is an open source orchestration framework for ....."}] 

I now know the final answer.
Final Answer: LangChain is an open source orchestration framework for building applications using large language models (LLMs) like chatbots and virtual agents .....

> Finished chain.
```
> 以防你没有去看ReAct的论文，简单解释是，这里Agent的每个动作要求LLM输出三行文本：
> 第一行：解释你要采取的行动
> 第二行：选择一个要使用的工具函数的名称
> 第三行：给出使用这个工具的输入参数

- [Self Ask With Search](https://python.langchain.com/docs/modules/agents/agent_types/self_ask_with_search)
   - 这是一个简单的包装，仅仅让模型执行一个简单的网络搜索步骤。
   - 这个类用来帮助开发者熟悉如何自定义构建一个Agent，也适用于非常小的模型，执行简单和轻量化的搜索任务时使用。
   - 使用此Agent时，**_只允许定义唯一的工具_**，该工具的名称（name字段）必须是"Intermediate Answer"，返回尽可能精简的搜索结果，Agent负责从中寻找和总结出一个最佳的答案。
```python
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.llms import Fireworks
from langchain_community.tools.tavily_search import TavilyAnswer

# TavilyAnswer工具类只会返回精简的搜索结果，记得将其命名为"Intermediate Answer"
# tools数组里只能有一个这样的工具
tools = [TavilyAnswer(max_results=1, name="Intermediate Answer")]

def new_self_ask_with_search_agent():
    # 定义一个提示词，可自定义，这里从一个langchain仓库中获取
    from langchain import hub
    prompt = hub.pull("hwchase17/self-ask-with-search")
    
    # 实例化一个模型对象
	llm = Fireworks()
    
    # 实例化智能体对象
    agent = create_self_ask_with_search_agent(llm, tools, prompt)

	return agent

agent = new_self_ask_with_search_agent()
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "what is LangChain?"})
```
### 工具类：Tool
工具是智能体可以调用的函数。 Tool 被抽象为两个组件：

- input schema：告诉LLM调用该工具需要哪些参数，需要为每个参数定义合理的命名及描述，它们将被LangChain嵌入进Agent的prompt中。
- function：实际要运行的函数，通常类型即为一个python函数。
#### 工具的使用和实现
##### Langchain内置工具
**在这里查看目前的内置工具和它们的使用方法：**[**https://python.langchain.com/docs/integrations/tools**](https://python.langchain.com/docs/integrations/tools)
步骤：【以使用Wikipedia查询工具为例】

1. 首先导入Python名称，部分工具依赖于第三方python库或插件，需要先安装它们：
   1. `%pip install --upgrade --quiet  wikipedia`
2. 导入名称时，请参考各工具的文档，确保从正确的库路径导入。
> 部分工具的使用可能还需要导入为了调用该工具而编写的额外模块（如下面的例子所示）
> 更为繁琐的可能需要在本地计算机上配置服务，或前往第三方平台配置密钥等，请参考具体的内置工具使用文档

   1. `from langchain.tools import WikipediaQueryRun`
   2. `from langchain_community.utilities import WikipediaAPIWrapper` 这里就需要一个包装维基百科API调用的模块
3. 做好了这些以后，可以实例化生成工具对象
   1. `wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())`

如果你不知道一个内置工具的具体定义，可以调试如下**基本属性**（下面以`tool`为例）：

- `tool.name`名称：字符串，定义工具的名称，可以在实例化时指定
- `tool.description`描述：字符串，描述工具的功能，可以在实例化时指定
- `tool.args`参数：默认使用JSON格式（即python字典对象），每一条参数的形式为`<参数名>：{'title': <参数标题>，'description':<参数描述>, 'type': <参数类型>}`，其中参数名是唯一的key，参数标题和描述是可选的，提高开发时的可读性，例如：
   - `{'query': {'title': 'Query', 'type': 'string'}}`
- `return_direct`：布尔变量，定义函数的输出是否是直接返回给用户的内容

如果一个工具只有一个参数，可以直接使用`tool.run(<参数>)`调用工具，否则，调用工具时传入完整的参数字典，格式为`{<参数名>: <参数值>}`，例如：

   - `tool.run({"query":"langchain"})`
##### 自定义工具
创建自定义工具有多种方法，可能会需要导入如下的包：
```python
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
```

- 使用@tool 装饰器
   - 最简单的方法，只需要在函数定义前加上`@tool`修饰符
   - 会默认使用函数名称作为工具名称，但可以通过传递字符串作为第一个参数来覆盖此名称。
   - 函数需要具有一个python风格的docstring，这将成为`@tool`修饰符生成的工具的描述。
   - 例如：
```python
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""  # 这个就是docstring
    return a * b

print(multiply.name)
print(multiply.description)
print(multiply.args)

## 输出如下：
multiply
multiply(a: int, b: int) -> int - Multiply two numbers.
{'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
```

   - 覆盖定义工具属性的例子：
```python
# 这个类定义了一个参数输入的基本模式，用来覆盖原函数定义的参数
class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")

# 覆盖了工具的名称，参数（使用自定义的SearchInput）和return_direct属性
@tool("search-tool", args_schema=SearchInput, return_direct=True)
def search(query: str) -> str:
    """Look up things online."""
    return "LangChain"

print(search.name)
print(search.description)
print(search.args)
print(search.return_direct)

## 输出如下：
search-tool
search-tool(query: str) -> str - Look up things online.
{'query': {'title': 'Query', 'description': 'should be a search query', 'type': 'string'}}
True
```

- 继承BaseTool子类
   - 子类化BaseTool类来显式定义自定义工具
   - 自由度最高，但需要更多的代码量
   - 需要先继承`BaseModel`定义工具需要传入的参数，然后继承`BaseTool`定义工具的其他属性
   - 举个例子：还是乘法计算器
```python
from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

# 定义Input Model时，参数名称为变量名，参数类型显示指定，使用一个Field对象定义可选的属性（标题，描述等）
class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

class CustomCalculatorTool(BaseTool):
    # 定义工具的基本属性
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput # 设置参数时需要显示指定类型
    return_direct: bool = True

    # 重写run函数，实现工具的执行
    def _run(
        self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return a * b

    # 重写_arun函数，实现工具的异步执行
    # 如果工具不需要异步执行，可以直接返回异常
    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")

# 实例化一个自定义工具
search = CustomSearchTool()
print(search.name)
print(search.args)

## 输出如下：
Calculator
{'a': {'title': 'A', 'description': 'first number', 'type': 'integer'}, 'b': {'title': 'B', 'description': 'second number', 'type': 'integer'}}
True
```

- 继承 StructuredTool 数据类
   - 相当于混合了前两种方法。
   - 有一定的自由度，同时写法简单。
   - 先定义好一个函数，然后使用`StructuredTool.from_function`方法，传入工具的属性即可。
   - 举例：
```python
# 工具要执行的函数
def search_function(query: str):
    return "LangChain"

# 工具参数会默认设为函数参数
search = StructuredTool.from_function(
    func=search_function,
    name="Search",
    description="useful for when you need to answer questions about current events",
    # coroutine= ... <- 同样可以指定异步执行的方法
)
```

   - 如果需要自定义参数覆盖函数的参数，可以指定`args_schema`:
```python
# 自定义参数类
class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

# 工具要执行的函数
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

calculator = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="multiply numbers",
    args_schema=CalculatorInput,
    return_direct=True,
    # coroutine= ... <- 同样可以指定异步执行的方法
)
```
#### 错误处理
如果工具执行函数的过程种遇到异常并返回，**正常情况下智能体会终止当前任务**。
如果希望智能体在遇到某些影响不大的异常后也可以继续执行，需要在函数执行体内抛出`ToolException` 并相应地设置 `handle_tool_error `。
**当工具抛出 **`ToolException`** 时，代理不会停止工作，而是根据工具的异常处理函数**`handle_tool_error`**处理异常，处理结果将作为观察返回给代理，并以红色打印。**
```python
from langchain_core.tools import ToolException

def search_tool1(s: str):
    raise ToolException("The search tool1 is not available.")

search = StructuredTool.from_function(
    func=search_tool1,
    name="Search_tool1",
    description="A bad tool",
    handle_tool_error=True, # 添加异常处理属性标记
)

search.run("test") 
### 输出：
'The search tool1 is not available.'
```

- 如果只抛出了`ToolException` 异常而没有定义 `handle_tool_error`属性，则LangChain仍然不会让智能体处理异常。
- 当`handle_tool_error`属性仅定义为`True`而不是异常处理函数时，LangChain会将异常字符串作为处理结果返回给智能体。
- `handle_tool_error`属性定义为一个函数时，其需要接受`ToolException`类型的异常对象，并返回字符串。
```python
def _handle_error(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Please try another tool."
    )


search = StructuredTool.from_function(
    func=search_tool1,
    name="Search_tool1",
    description="A bad tool",
    handle_tool_error=_handle_error,
)

search.run("test")
### 输出：
'''
The following errors occurred during tool execution:
The search tool1 is not available. Please try another tool.
'''
```

#### 在OPENAI聊天模型中使用工具
如果你只是需要一个工具函数，或是觉得LangChain中的内置工具很好用，但不需要一个智能体来决定工具的调用，可以将工具调用显式地嵌入ChatOpenAI聊天模型的提示中：
```python
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import MoveFileTool, format_tool_to_openai_function

model = ChatOpenAI(model="gpt-3.5-turbo-0613")
tools = [MoveFileTool()]
functions = [format_tool_to_openai_function(t) for t in tools]

message = model.predict_messages(
    [HumanMessage(content="move file foo to bar")], functions=functions
)
```

### 自定义智能体
智能体其实只是一个带有 工具调用和记忆的 LLM查询或对话系统，在底层完全可以使用LangChain的ICLE语法（即“链”）来实现。在下面的例子中，使用ChatOpenAI作为基础对话模型来实现自定义智能体。

- 先定义好要使用的模型和工具
- 在对LLM提示的时候，记得加上`MessagesPlaceholder`,以实现和智能体的对话具有历史记忆能力。
```python
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length] # 定义工具
chat_history = [] # 记录历史

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # 定义llm
# 将工具绑定到llm
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words.",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
```

- 然后，从`langchain.agents.format_scratchpad`包中导入自定义创建智能体的基础类型。
   - 这里，因为我们需要将函数调用转化为OPENAI api的形式，所以导入`format_to_openai_function_messages`函数，并同时导入`OpenAIFunctionsAgentOutputParser`来解析OPENAI 模型返回的输出。
- 使用链式方法创建智能体，通常一个链条的构成是：
   - 输入：即核心概念中的AgentAction，将“用户输入”，“中间步骤”和“历史”三个部分按照字典键值对的格式配置。
      - 必要的时候（如使用OPENAI api的工具调用），要将输入转化为模型接口支持的格式。
   - 提示：按照已定义好的提示模板
   - 已绑定好工具的LLM对象或聊天模型对象
   - 解析输出的对象，根据你查询的模型决定。
```python
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)
```

- 最后，创建一个智能体执行器，配置好输入和历史即可执行：
```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

input1 = "how many letters in the word educa?"
result = agent_executor.invoke({"input": input1, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ]
)

input2 = "is that a real word?"
agent_executor.invoke({"input": input2, "chat_history": chat_history})

...
```
#### AgentExecutor 的额外参数

- `verbose=True`：是否输出详细执行信息
- `max_iterations=10`：限制最大迭代次数 
- `max_execution_time=10`：限制最大执行时间（秒）
- `handle_parsing_errors=True`：是否自动处理解析错误（见下面的小节)
- `handle_parsing_errors="..."`：处理解析错误，并使用自定义的字符串（见下面的小节)
### 智能体执行的监督和检查
#### 按步骤执行
调用`AgentExecutor`类对象的`iter`函数，获得每一个中间步骤的迭代器。由此，可以自由控制执行哪些步骤，并随时可以插入其他功能的代码。
```python
question = "What is the product of the 998th, 999th and 1000th prime numbers?"

for step in agent_executor.iter({"input": question}):
    # 这种写法等价于:
    # output = step.get("intermediate_step")
    # if output:
    if output := step.get("intermediate_step"):
        action, value = output[0]
        if action.tool == "GetPrime":
            print(f"Used Tool GetPrime，get number： {value} ...")
            assert is_prime(int(value))
        # 询问控制台用户是否继续Agent执行
        _continue = input("Should the agent continue (Y/n)?:\n") or "Y"
        if _continue.lower() != "y":
            break
```

#### 获得中间步骤的日志
智能体执行完毕后，`agent_executor.invoke`会返回一个对象，其中包含所有执行历史的情况，其结构如下：
```python
response = agent_executor.invoke({"input": "What is Leo DiCaprio's middle name?"})

print(response.keys())

# TODO
```

要获得每一个中间步骤的详细情况，获取`response["intermediate_steps"]`即可，这是一个列表，其中每个元素是一个元组，包含两个对象，分别是：

- 一个`AgentActionMessageLog`对象，记录当前步骤调用工具的详细信息（包括工具名称，参数，日志字符串，实际嵌入LLM提示中的`AIMessage`对象等）
   - 如果步骤是“输出结果返回给用户”，则这里会是一个`AgentFinish`对象。
- 一个字符串，为调用工具返回的“Observation”
```python
[(
    AgentActionMessageLog(
        tool='Wikipedia', 
        tool_input='Leo DiCaprio', 
        log='\nInvoking: `Wikipedia` with `Leo DiCaprio`\n\n\n', 
        message_log=[
            AIMessage(
                content='', 
                additional_kwargs={
                    'function_call': {
                        'name': 'Wikipedia', 
                        'arguments': '{\n  "__arg1": "Leo DiCaprio"\n}'
                    }
                })
        ]
    ), 
     'Page: Leonardo DiCaprio\nSummary: Leonardo Wilhelm DiCaprio (; Italian: [diˈkaːprjo]; born November 1'
)]
```
#### 自我检查错误
> 注意：这和之前介绍的工具类的错误处理不同，前者只是处理调用工具时，工具可能发生的错误。而这里是从智能体的层面上，防止LLM在输出对任务的推理时出现幻觉，产生不符合规定格式的字符串（如json格式，ReAct范式等）

在实例化任意 `AgentExecutor`的时候加入参数`handle_parsing_errors=True`，可以使LangChain自动处理智能体的不当输出，这主要包括无效的格式或不完整的回应。
```python
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
```
例如，如果Agent使用“ReAct"策略，但LLM并未遵循提示输出“Action: ”字段时，LangChain将向LLM自动重发一条消息，告知错误格式的位置的应当的输出，以此尝试引导LLM纠正错误。
```
> Entering new AgentExecutor chain...

# LLM的返回，缺少 Action: 字段
Thought: I should search for "Leo DiCaprio" on Wikipedia
Action Input: Leo DiCaprio

# LangChain生成的错误反馈，将作为下一步的Input自动输入给LLM
Invalid Format: Missing 'Action:' after ...

# LLM的下一步返回，这次的格式正确
Thought:I should search for "Leonardo DiCaprio" on Wikipedia
Action: Wikipedia
Action Input: Leonardo DiCaprio

# 正确执行工具调用后，返回给LLM工具调用的输出结果
Page: Leonardo DiCaprio 
Summary: Leonardo Wilhelm DiCaprio (; Italian: [diˈkaːprjo]; born November 1 ...
```
如果要将自动生成的错误反馈修改为自定义的字符串，显示指定参数`handle_parsing_errors`时传入字符串而非布尔值即可。
```python
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True,
    handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",
)
```
#### 返回结构化输出
上面的所有情况，智能体都以字符串输出结果，如果想要智能体也输出结构化的数据，以便结合到更大范围的系统中。
我们可以将“返回给用户”这一步骤也视为一个工具调用，只不过调用的结果是输出结构化的文本。在LangChain种，可以定义Response类来实现这一功能。
【注意：以下示例以基于OPENAI聊天模型的智能体为例】
```python
from typing import List
from pydantic import BaseModel, Field

class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(
        description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information"
    )
```
然后，在定义工具时，使用函数`convert_pydantic_to_openai_function`将`Response`类定义为工具函数:
```python
from langchain_openai import ChatOpenAI
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

llm = ChatOpenAI(temperature=0)
llm_with_tools = llm.bind(
    functions=[
        # 其他工具，例如一个 retriever tool
        format_tool_to_openai_function(retriever_tool),
        ...
        # Response 结构化返回工具
        convert_pydantic_to_openai_function(Response),
    ]
)
```
定义 解析Response工具输出的函数。这里我们没有使用预置智能体，因此引入名称`AgentFinish`和`AgentActionMessageLog`来包装LLM输出的中间步骤数据结构，以便得到可以被LangChain其他模块解析的对象。
```python
import json
from langchain_core.agents import AgentActionMessageLog, AgentFinish

def parse(output):
    # 如果LLM没有调用任何函数，将自定义回答结果直接包装为AgentFinish
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # 如果LLM要调用某个非Response类的函数，将其函数名和参数抽取出来
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # 如果LLM调用了Response函数准备返回结构化数据，将返回包装为AgentFinish。
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # 否则，将返回包装为 agent action，这将使LangChain代替我们执行函数并继续下一步与智能体的交互。
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )
```
最后，使用parse作为链步骤中解析智能体输出的动作：
```python
from langchain.agents.format_scratchpad import format_to_openai_function_messages

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        # Format agent scratchpad from intermediate steps
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | parse
)

agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, verbose=True)

agent_executor.invoke(
    {"input": "..."}, return_only_outputs=True,
)
```

### 工具包：Toolkits
对于许多常见任务，代理将需要一组相关的工具。为此，LangChain提供了工具包的概念——实现特定目标所需的大约3-5个工具的组合。
使用工具包非常方便，只需导入工具包的名称，然后使用`get_tools()`方法即可返回一个列表，包含该工具包中的所有工具，你可以直接将该列表作为参数传入实例化Agent的方法中，轻松愉快！
```python
# Initialize a toolkit
toolkit = ExampleTookit(...)

# Get list of tools
tools = toolkit.get_tools()

# Create agent
agent = create_agent_method(llm, tools, prompt)
```
#### Langchain内置工具包
**所有内置工具包的使用文档：**[https://python.langchain.com/docs/integrations/toolkits](https://python.langchain.com/docs/integrations/toolkits)
