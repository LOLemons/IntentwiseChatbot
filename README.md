# Intentwise Chatbot
A Chatbot powered by LLaMa 2 made to answer questions about AI ecosystems and their implications

To get started, clone the repository and install all the dependencies in setup.txt or use the .venv virtual environment provided.
Open the `backend.py` file, change the path of the LlaMa 2 model (which you will have to have gain access to and downloaded from HuggingFace).
Then run the `backend.py` file. It will run a flask app locally on your computer.

A fairly capable Nvidia GPU is needed to run the webpage locally (at least 8gb vram recommended) to make use of the quantization.

Keep in mind that the LLM too is running locally and extensive resources will be used to run the web page.

Report issues on the issues page

*The white paper isn't ready yet so the LLM will produce terrible responses for now*

*Once the white paper is ready, we will have to change the name of the pdf being used. For now, we will use placeholder text on `sample.pdf`*
