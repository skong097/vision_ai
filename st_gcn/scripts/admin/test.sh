cd /home/gjkong/dev_ws/st_gcn/

# Fine-tuned 모델 테스트
python test_stgcn_finetuned.py \
    --video /home/gjkong/dev_ws/st_gcn/video/S001C001P001R001A043_rgb.avi \
    --model /home/gjkong/dev_ws/st_gcn/checkpoints_finetuned/best_model_finetuned.pth
