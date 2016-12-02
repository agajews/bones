docker rm -f tf-container
docker create -p 2222:2222 --name tf-container tensorflow/tensorflow:nightly-py3 bash /root/run.sh
docker cp run.sh tf-container:/root
docker cp script.py tf-container:/root
docker cp bones tf-container:/root
docker start tf-container
# docker logs -f tf-container
