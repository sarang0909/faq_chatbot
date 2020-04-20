# FAQ-Chatbot
COVID-19 FAQ chatbot in python along with user interfce

To Run the application:
Install all libraries in requirement.txt 
and run: python main.py


GCP AppEngine Deployment:
gcloud config set project <PROJECT_ID>
virtualenv -p python3 chatbot_env
source chatbot_env/bin/activate
gcloud app create
gcloud app deploy
