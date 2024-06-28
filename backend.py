#import dependencies
from flask import Flask, render_template, request
import fitz
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

#Set up white paper text for LlaMa prompt
with fitz.open("sample.pdf", filetype='pdf') as doc: #change the name of the pdf file later
    text = ""
    for page in doc:
        text += page.get_text()

#model,  tokenizer, and pipeline setup
model_dir = "C://Users//shrim//Llama-2-7b-hf" #change the directory to the model later
model = LlamaForCausalLM.from_pretrained(model_dir)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
pipeline = transformers.pipeline(
"text-generation",
model=model,
tokenizer=tokenizer,
torch_dtype=torch.float16,
device_map="auto"
)

#function to process context and user question with LlaMa
def model_output(context, question):
    sequences = pipeline(
        f"""
Your purpose is to answer questions about AI, its various implications, factors that must be considered while implementing it, and the responsibilities of corporations and governments when developing and implementing AI.
Follow exactly these 3 steps:
1. Read the context below and aggregrate this data
Context : {context}
2. Answer the question using only this context, use NOTHING ELSE
User Question: {question}

Remember, you are an honest, helpful, and truthful assistant. Never make up answers on your own. If you don't have any context and are unsure of the answer, or if the question asked by the user is out of context, reply with a witty answer and tell them to ask questions within the context.
""",
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=400
    )
    return sequences

#frontend flask app
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def welcome_page():
    return render_template("Webpage.html")

@app.route("/modelResponse", methods=['GET', 'POST'])
def question_replace():
    user_question = str(request.form["user_question"])
    output = model_output(text, user_question)
    for seq in output:
        model_response = f"{seq['generated_text']}"
    return render_template("Webpage.html", model_response=model_response)

if __name__== '__main__':
    app.run(debug=True)