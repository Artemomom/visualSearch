# Local testing

1. (Optional) To run backend you need a model and a prebuilt index. 
You can download the ones used on the demo server using the following gsutil command:
    ```
    gsutil cp -r gs://image-search-demo-data/fashion-search-deit data/
    ```
    Alternatively, you can build a new index for any model using the `index.py` script from the root directory.
2. Adjust model and index paths in the `docker-compose.yml` file 
(section `volumes` of  the `style_backend` service)

3. Change the working directory to `local-run`:
    ```
    cd local-run
    ```

4. Build images:
    ```
    docker-compose build
    ```
    
5. Start containers:
    ```
    docker-compose up
    ```
    
After these steps you can access services at the following urls:
 - http://localhost - frontend
 - http://localhost:5000/process - backend (accepts POST requests)