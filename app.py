from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel

# Load the environment variables from the .env file. and override the existing ones to avoid any conflicts
load_dotenv(override=True)

# create a pydantic model for the evaluation.
# It has two attributes 
# - is_acceptable: bool : To check if the response is acceptable or not
# - feedback: str : To give the feedback on the response
class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str



# This functionn sends a push notification to user's mobile through pushover application along with the test
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


# ----- TOOLS--------------------------------
# This function records the user details and sends a push notification to the user
# If the end user is interested to get in touch then this function is called
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


# This function records the unknown question and sends a push notification to the user
# If the LLM is not able to answer the question then this function will be called
def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}


# Define a JSON schema to tell the LLM how to use tool "record_user_details"
record_user_details_json = {
    "name": "record_user_details", # tool name
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address", # when to use tool
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

# This is a JSON schema to tell LLM how to use tool "record_unknown_question"
record_unknown_question_json = {
    "name": "record_unknown_question", # tool name
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer", # when to use tool
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

# Defines list of tools to be passed to LLM
tools = [{"type": "function", "function": record_user_details_json}, # This says tool is a function and "record_user_details_json" describing the tool "record_user_details"
        {"type": "function", "function": record_unknown_question_json}]   # This says tool is a function and "record_unknown_question_json" describing the tool "record_unknown_questions"


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.gemini = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.name = "Arpan Bandyopadhyay"
        # Prepare the resources for LLM
        reader = PdfReader("me/linkedin_arpan.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    # This function handles the tool calls requested by LLM
    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            # get the tool name from global scope
            tool = globals().get(tool_name)
            # execute the tool if it exists
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    

    # This function defines system prompt for LLM (How to act while providing the answer)
    # System prompt is equipped with the resources - summary and linkedin profile
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    


    # Cobstruct the full message list to send to the LLM -It includes 
    # - System prompt
    # - History of conversation
    # - users message
    def chat(self, message, history):
        # Construct messages list (conversation)
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools )# tools=tools ->list of tools that LLM can use
            # finish_reason is the reason what API tells us that why the LLM stopped generating the response
            if response.choices[0].finish_reason=="tool_calls": # finish_reason is "tool_calls" means LLM requested for tool calls
                message = response.choices[0].message
                tool_calls = message.tool_calls # Extract list of tool calls what LLM wants to invoke
                results = self.handle_tool_call(tool_calls) # Invoke the tool and get the result
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        # reply from LLM against users message        
        reply= response.choices[0].message.content
        # Evaluate the reply from the LLM using evaluator LLM
        evaluation = self.evaluate(reply, message, history)
        # Check if the reply is acceptable or not
        if evaluation.is_acceptable:
            print("Passed evaluation - returning reply")
        else:
            print("Failed evaluation - retrying")
            print(evaluation.feedback)
            # If the reply is not acceptable then retry it again and get the improved reply from the LLM
            reply = self.rerun(reply, message, history, evaluation.feedback)       
        return reply
    


    # This function retries a previous LLM response that was rejected by a quality evaluator. It sends a new prompt to the LLM with:
    # the previous answer,
    # the reason it was rejected,
    # and asks it to try again.
    def rerun(self,reply, message, history, feedback):
        updated_system_prompt = self.system_prompt() + f"\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
        updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n" #Add the previous (bad) answer to the system prompt
        updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"  # Add the reason why it was rejected
        messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]
        response = self.gemini.chat.completions.create(model="gemini-2.0-flash", messages=messages)
        return response.choices[0].message.content
    


    # This function defines evaluator system prompt on how to evaluate the reply from the LLM
    def evaluator_system_prompt(self):
        evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
    You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
    The Agent is playing the role of {self.name} and is representing {self.name} on their website. \
    The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
    The Agent has been provided with context on {self.name} in the form of their summary and LinkedIn details. Here's the information:"

        evaluator_system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."
        return evaluator_system_prompt
    


    # Define a function to construct a user prompt for an evaluator LLM
    def evaluator_user_prompt(self,reply, message, history):
        # Start by including the full conversation history between the user and the agent
        user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        
        # Add the latest message that the user just asked
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        
        # Add the latest reply that the agent gave
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        
        # Ask the evaluator to review the agent's response
        user_prompt += f"Please evaluate the response, replying with whether it is acceptable and your feedback."
        
        # Return the complete constructed prompt for the evaluator LLM
        return user_prompt
    

    
    # This function evaluates the reply from the LLM using evaluator LLM
    # It includes :
    # - evaluator System prompt
    # - history of conversation
    # - LLM's reply
    # - user's original question
    def evaluate(self,reply, message, history) -> Evaluation:
        messages = [{"role": "system", "content": self.evaluator_system_prompt()}] + [{"role": "user", "content": self.evaluator_user_prompt(reply, message, history)}]
        response = self.gemini.beta.chat.completions.parse(model="gemini-2.0-flash", messages=messages, response_format=Evaluation) # This says it should generate response as per Evaludation class formats
        return response.choices[0].message.parsed



if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
    