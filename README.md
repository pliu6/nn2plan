# nn2plan
A tool to optimize a neural network and generate the plan for TensorRT

## Example1: generate a plan from caffe model
Download the pretrained model and prototxt
```
wget --no-check-certificate 'https://nvidia.box.com/shared/static/3qdg3z5qvl8iwjlds6bw7bwi2laloytu.gz' -O DetectNet-COCO-Dog.tar.gz
tar zxvf DetectNet-COCO-Dog.tar.gz
```

Run the command to optimize and serialize the inference engine to a cache file
```
./nn2plan -t caffe DetectNet-COCO-Dog/deploy.prototxt DetectNet-COCO-Dog/snapshot_iter_38600.caffemodel 2 coverage bboxes
```

The generated plan file will be DetectNet-COCO-Dog/snapshot_iter_38600.caffemodel.2.tensorcache
