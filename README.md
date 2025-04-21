<img width="500" alt="image" src="https://raw.githubusercontent.com/htahel09/NortheasternOGS-Chatbot/refs/heads/main/Hybrid_CAG_RAG.jpeg" />

Project Description:
This project aims to develop an AI-powered chatbot that enables international students to efficiently access information from Northeastern’s Office of Global Services website using natural language queries. By leveraging Retrieval-Augmented Generation (RAG) and Cache-Augmented Generation (CAG), the chatbot will provide accurate, context-aware responses with citations.
A key challenge is the website’s complex structure, making it difficult for users to find relevant information. The solution involves automated web scraping to extract and preprocess content, ensuring up-to-date responses without manual intervention. The chatbot will be deployed locally first, with plans for cloud deployment for broader accessibility.
The project also serves as a learning opportunity for web scraping, NLP model fine-tuning, and cloud-based deployment, enhancing the technical skills of the development team

Dataset:
All URLS found by crawling: https://international.northeastern.edu/ogs/
Allowed Domains: https://international.northeastern.edu/

Requirements
bert_score==0.3.13
bleach==4.1.0
datasets==3.3.2
evaluate==0.4.3
faiss_cpu==1.10.0
numpy==2.2.5
packaging==25.0
scikit_learn==1.6.1
sentence_transformers==3.4.1
streamlit==1.44.1
torch==2.6.0+cu124
transformers==4.50.0.dev0

