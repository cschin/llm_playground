import gradio as gr
import random
import time
import requests
import json

with gr.Blocks() as demo:


    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Input your query about Gene Review")
    query_type = gr.Dropdown(choices=["Summary", "Q&A"], value=0, type="index", label="Query Type")
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history, dropdown):
        msg = {"text":message, "topn":3}
        if dropdown == 0:
            r = requests.post('http://127.0.0.1:3000/api/post_query_for_summary_of_a_topic', json = msg)
        else:
            r = requests.post('http://127.0.0.1:3000/api/post_query_for_answer_of_a_question', json = msg)


        bot_message = json.loads(r.text) 
        if bot_message is not None:
            bot_message = bot_message[0]["text"]
            chat_history.append((message, bot_message))
            
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot, query_type], [msg, chatbot])

demo.launch()

