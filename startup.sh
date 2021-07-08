#
gsutil cp -r gs://image-search-demo-data/$project_name/model .
gsutil cp -r gs://image-search-demo-data/$project_name/index .

export PYTHONUNBUFFERED=1
python server.py
