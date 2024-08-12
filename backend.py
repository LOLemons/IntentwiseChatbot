from flask import Flask, render_template, request, jsonify
import fitz
import openai

#text extraction from whitepaper
with fitz.open("whitepaper.pdf", filetype='pdf') as doc:
    text = ""
    for page in doc:
        text += page.get_text()

#prompt for the bot
def make_prompt(question):
        prompt = f"""
SYSTEM: Your purpose is to answer questions about AI, its various implications, factors that must be considered while implementing it, and the responsibilities of corporations and governments when developing and implementing AI.
Follow exactly these 3 steps:\n
1. Read the context below and aggregrate this data \n
Context :\n\n
{text}\n\n
2. Answer the question using only the above context, use NOTHING ELSE\n\n
Remember, you are an honest, helpful, and truthful assistant. Never make up answers on your own. If you don't have any context and are unsure of the answer, or if the question asked by the user is out of context, reply with a witty answer and tell them to ask questions within the context.
\n\n
User Question: {question}
\n\n
Response: """
        return prompt

openai.api_key = '<api key>' #change <api key> to your own key

#parameters for gpt 3.5-turbo
def get_completion(prompt):
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

#running bot as a webapp on flask
app = Flask(__name__)
@app.route("/")
def home():    
    return render_template("index.html")
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')
    model_prompt = make_prompt(userText)
    response = get_completion(model_prompt)  
    return response
if __name__ == "__main__":
      app.run(debug=True)
