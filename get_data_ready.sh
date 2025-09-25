#tar -xvzf ./mydataset/clear.tar.gz -C /workspace/mydataset/
#mv /workspace/mydataset/content/mount/MyDrive/2025fog/clear /workspace/mydataset/
#tar -xvzf ./mydataset/foggy_zurich_direct.tar.gz -C /workspace/mydataset/
#mv /workspace/mydataset/content/mount/MyDrive/2025fog/foggy_zurich_direct /workspace/mydataset/
#wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s.pt
#python ./helpers/merge_zurich_data.py
rm -r "mydataset/foggy_zurich_direct/light_images"
rm -r "mydataset/foggy_zurich_direct/medium_images"
mkdir mydataset/foggy_zurich_direct/train
mkdir mydataset/foggy_zurich_direct/val
mv mydataset/foggy_zurich_direct/images mydataset/foggy_zurich_direct/train/
python ./helpers/restructure_foggy_zurich.py
