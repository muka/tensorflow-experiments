# Apple type recognition

Notes for experiments on tensorflow and apple recognition


## Setup

- Get models `make fetch-models`
- Ensure tensroflow py modules are reachable `export PYTHONPATH=$PYTHONPATH:${PWD}/models/research:${PWD}/models/research/slim`
- Run TF `make run-tensorflow`

Image labelled with MS VoT https://github.com/Microsoft/VoTT.git


#Testing

I used that repo https://github.com/chtorr/go-tensorflow-object-detection to get RT detection out of Webcam

- Clone & install as descripted in the repo
- Edit under `data/` setting 1: to apple
- launch specifing the directory where you exported the frozen model (see `make export-trainer`)

## References

- https://gist.github.com/douglasrizzo/c70e186678f126f1b9005ca83d8bd2ce
- https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-4-training-the-model-68a9e5d5a333
- https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
