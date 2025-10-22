
CUDA_VISIBLE_DEVICES=0 python main_vsod.py \
  --datadir $3 \
  --datfile ../dataset/test_video_lst.txt \
  --preddir ../results/$2 \
  --dataset $1 \
  --savedir ../results/$2


