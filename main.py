from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(override=True)


def main():
    print("Hello from langchain-langgraph!")
    information = ''
    with open('.\maximus_the_confessor.txt','r') as file:
        information = file.read()

    summary_template = """Given the information {information}, about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm = ChatOpenAI(model="gpt-4")
    llm = ChatOllama(model="gemma3:270m",base_url="http://localhost:11434")  # NOT http://localhost:11434/v1

    llm2 = ChatOpenAI(model="google/gemma-4-e4b",base_url="http://192.168.0.252:1234/v1", api_key="lm-studio")
    
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

    print("\n \n")
    chain = summary_prompt_template | llm2
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
