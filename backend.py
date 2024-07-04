#import dependencies
from flask import Flask, render_template, request
import fitz
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, BitsAndBytesConfig

#Set up white paper text for LlaMa prompt
with fitz.open("sample.pdf", filetype='pdf') as doc: #change the name of the pdf file later
    text = ""
    for page in doc:
        text += page.get_text()

#model, tokenizer setup
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model_dir = "C://Users//shrim//Llama-2-7b-hf" #change the directory to the model later
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, quantization_config=quantization_config, torch_dtype=torch.float16, device_map="auto"
)

#model generation settings
generation_config = GenerationConfig.from_pretrained(model_dir)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

#pipeline setup
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=generation_config,
)

llm=HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={'temperature':0})

#template and prompt creation
template ="""<s>[INST] <<SYS>>
Your purpose is to answer questions about AI, its various implications, factors that must be considered while implementing it, and the responsibilities of corporations and governments when developing and implementing AI.
Follow exactly these 3 steps:
1. Read the context below and aggregrate this data
Context : {text}
2. Answer the question using only this context, use NOTHING ELSE
User Question: {question}

Remember, you are an honest, helpful, and truthful assistant. Never make up answers on your own. If you don't have any context and are unsure of the answer, or if the question asked by the user is out of context, reply with a witty answer and tell them to ask questions within the context.
<</SYS>>[/INST]"""

prompt = PromptTemplate(
    input_variables=["text", "question"],
    template=template,
)

#function to process context and user question with LlaMa
def model_output(question):
    return llm(prompt.format(text=text, question=question))

#frontend flask app
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def welcome_page():
    return render_template("Webpage.html")

@app.route("/modelResponse", methods=['GET', 'POST'])
def question_replace():
    user_question = str(request.form["user_question"])
    output = model_output(user_question)
    return render_template("Webpage.html", model_response=output)

if __name__== '__main__':
    app.run(debug=True)