docker build -t my_ratsql .
docker run --gpus 2 --rm -m4g -p 6667:6667 -v $(pwd):/mnt --name my_ratsql_dev -it my_ratsql