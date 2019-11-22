cd ~/Documents/local_patch_retrieval/
# Nakano Server
python3 lp_retrieval.py --feature_comb add --loss_weight l1 --epoch_num 400 \
--model_prefix lp_rtv_add_l1 --batch_size_per_gpu 4 --normalize_embedding
python3 lp_retrieval.py --feature_comb add --loss_weight l2 --epoch_num 400 \
--model_prefix lp_rtv_add_l2 --batch_size_per_gpu 4 --normalize_embedding