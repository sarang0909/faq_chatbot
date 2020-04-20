# FAQ-Chatbot
COVID-19 FAQ chatbot in python along with user interfce

To Run the application:<br>
Install all libraries in requirement.txt <br>
and run: python main.py<br>


GCP AppEngine Deployment:<br>
gcloud config set project <PROJECT_ID> <br>
virtualenv -p python3 chatbot_env<br>
source chatbot_env/bin/activate<br>
gcloud app create<br>
gcloud app deploy<br>
