# Style Similarity Search

For Local run 
[download](https://drive.google.com/file/d/10R6SLY4zAaILC1iO02Us6iHNpcrsxkKr/view?usp=sharing) model embeddings for local run
and locate downloaded files in "model" folder 

* cd local-run 
* update docker-compose.yaml file to contain following volumes
  - /{PATH_TO_FOLDER}/model:/usr/src/app/model
* Build images:
    ```
    docker-compose build
    ```
*. Start containers:
    ```
    docker-compose up
   
After these steps you can access services at the following urls:
 - http://localhost - frontend
 - http://localhost:5000/process - backend (accepts POST requests)
 
 Depending on Provided Weights Application can more focus on detetection Style or Content Features 

To enable this feature, set the environment variables for example 
` STYLE_COEFFICIENT:` `0.7`
` CONTENT_COEFFICIENT:` `0.3` 
