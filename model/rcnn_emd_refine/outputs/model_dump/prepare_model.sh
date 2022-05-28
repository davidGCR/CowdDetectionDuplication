
#!/bin/bash

#RWF dataset
echo "===> Download pretrained model..."
id="1y_8_pX_ThhktEofWFbf05rsk0rzIGMv3"
gdown --id $id
filename="rcnn_emd_refine_mge.pth"
src="/content/${filename}"
dst="/content/CowdDetectionDuplication/model/rcnn_emd_refine/model_dump/outputs"
mv $src ${dst}
