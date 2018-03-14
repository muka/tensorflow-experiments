
setup-trainer:
	export PYTHONPATH=$PYTHONPATH:${PWD}/models/research:${PWD}/models/research/slim
	mkdir -p data
	./create_tf.py
	cp tmp/images_output/pascal_label_map.pbtxt data/label_map

run-trainer:
	export PYTHONPATH=$PYTHONPATH:${PWD}/models/research:${PWD}/models/research/slim
	cd ./models/research && python3 object_detection/train.py --logtostderr --train_dir=${PWD}/data --pipeline_config_path=${PWD}/data/ssd_mobilenet_v1_coco.config

export-trainer:
	rm -fr ${PWD}/data/melanet/saved_model
	python3 object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ${PWD}/data/pipeline.config --trained_checkpoint_prefix ${PWD}/data/model.ckpt --output_directory ${PWD}/data/melanet 

deps:
	pip3 install tensorflow pandas

fetch-models-src:
	git clone https://github.com/tensorflow/models.git
	ln -s ./models/research/object_detection/ ./

fetch-models:

	mkdir -p data/model

	# wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
	# tar -xvf ssd_inception_v2_coco_2017_11_17.tar.gz
	# mv ssd_inception_v2_coco_2017_11_17 model
	# rm ssd_inception_v2_coco_2017_11_17.tar.gz
	# wget https://github.com/tensorflow/models/raw/master/research/object_detection/samples/configs/ssd_inception_v2_coco.config
	# mv ssd_inception_v2_coco.config data/

	wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
	tar -xvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz
	mv ssd_mobilenet_v1_coco_2017_11_17 data/model
	rm ssd_mobilenet_v1_coco_2017_11_17.tar.gz
	wget https://github.com/tensorflow/models/raw/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config
	mv ssd_mobilenet_v1_coco.config data/

run-tensorflow:
	docker run --name tf1 --rm -it -v ${PWD}/data:/data -p 8888:8888 tensorflow/tensorflow
