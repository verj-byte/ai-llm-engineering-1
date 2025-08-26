# This implementation is an example of using OpenAI and Gemini with ability to select which model to use based on user's preference.

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
# import google.generativeai as genai  
from langchain_google_genai import ChatGoogleGenerativeAI as genai

import os

import chainlit as cl

# model = genai.GenerativeModel("gemma-3-27b-it")

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="OpenAI",
            markdown_description="This is using OpenAI model.",
            icon="https://picsum.photos/id/180/200",
        ),
        cl.ChatProfile(
            name="Gemini",
            markdown_description="This is using Gemini model.",
            icon="https://picsum.photos/id/181/200",
        ),
    ]

# Set the user's environment for the chat session.
# user_env = cl.user_session.get("env")
# os.environ["OPENAI_API_KEY"] = user_env["OPENAI_API_KEY"]

@cl.on_chat_start
async def on_chat_start():

    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"starting chat using the {chat_profile} profile model"
    ).send()

    user_env = cl.user_session.get("env")
    # os.environ["OPENAI_API_KEY"] = user_env["OPENAI_API_KEY"]

    if chat_profile=="OpenAI":
        model = ChatOpenAI(api_key=user_env["OPENAI_API_KEY"],streaming=True)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
                ),
                ("human", "{question}"),
            ]
        )
        runnable = prompt | model | StrOutputParser()
        cl.user_session.set("runnable", runnable)
    else:
        # genai.configure(api_key=user_env["GOOGLE_GEMINI_API_KEY"])  
        # model = genai.GenerativeModel("gemma-3-27b-it")
        model = genai(model="gemini-2.0-flash", api_key=user_env["GOOGLE_GEMINI_API_KEY"], streaming=True)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
                ),
                ("human", "{question}"),
            ]
        )
        runnable = prompt | model | StrOutputParser()
        cl.user_session.set("runnable", runnable)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
                ),
                ("human", "{question}"),
            ]
        )
        runnable = prompt | model | StrOutputParser()
        cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):

    # user_env = cl.user_session.get("env")
    # os.environ["OPENAI_API_KEY"] = user_env["OPENAI_API_KEY"]

    chat_profile = cl.user_session.get("chat_profile")
    

    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()