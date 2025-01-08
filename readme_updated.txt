# Lumenalta: LLM Training with Project Gutenberg Catalog

### Objective
Train an LLM (large language model) using the Project Gutenberg Catalog to create book summaries.

### Important clarifications
- This test is aiming for the candidate to deliver a simple E2E model and query example, by any means, we are not expecting a production ready solution.
- As for the sample data from the [Project Gutenberg Catalog](https://www.gutenberg.org/ebooks/offline_catalogs.html#xmlrdf), you do not need to use the whole catalog, as stated on the Data Collection section, you can choose a diverse set of genres and styles, and train your model with a limited selection.
- As for the documentation, a simple update to this document with your thought processes, approach, decision making and result analysis will be enough, no formal documentation is expected.


### Steps
1. **Data Collection**: 
- Download a selection of books from the [Project Gutenberg Catalog](https://www.gutenberg.org/ebooks/offline_catalogs.html#xmlrdf). Choose a diverse set of genres and styles.
- Preprocess the text (tokenization, removing special characters, etc.).
2. **Baseline LLM Training**: 
- Use a pre-trained LLM (e.g., T5, LLaMA 2, etc) as a baseline.
- Fine-tune the LLM on the collected book data.
- Evaluate the LLM-generated summaries using metrics like ROUGE, BLEU or an alternative of your choosing.
3. **RAG Pipeline Implementation**: 
- Set up a retrieval system to fetch relevant passages from external knowledge bases (e.g., Wikipedia, Project Gutenberg).
- Combine the retrieved passages with the LLM-generated summaries to enhance book summaries.
- Implement the RAG pipeline using Hugging Face Transformers or similar libraries.
- The augmentation can be done with a small set of passages/articles, there's no need to use a large dataset.
4. **Evaluation Metrics**:
- Compare the RAG-enhanced summaries with the LLM baseline.
- Assess the quality, coherence, and informativeness of the summaries.
5. **Documentation and Model Deployment**: 
- Document the entire process, including model architecture, hyperparameters, and any challenges faced.
- Deploy the trained model for book summarization.
- **Optional**: Launch the application through [Streamlit](https://streamlit.io/) or any similar model hosting site.

### Deliverables
- A document outlining your evaluation process, including dataset details, LLM evaluation approach, RAG metrics, and production readiness considerations.
- A zip file with the necessary contents to test your model, and an updated README that will contain instructions to test said model.

### Bonus (Optional)
- Explore domain-specific retrieval sources (e.g., literary analysis articles) for better performance.

### Aditional Instructions
This project is a layout upon which you will build your solution, please update `requirements.txt` with the necesary dependencies.


##########Vandana Singh############

#Development_code folder contains all baseline and rag based code
1.baseline_project_final -run base_test.py
2. rag_project_final-run rag_test.py 
3. Compare:run compare.py